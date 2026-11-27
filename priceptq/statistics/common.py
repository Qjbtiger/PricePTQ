"""Calibration, reduced Hessian sketches, whitening, and forward KL for the main paper."""

from __future__ import annotations

import hashlib

import os

from collections import OrderedDict

from contextlib import nullcontext

from dataclasses import dataclass

from pathlib import Path

from typing import Any

import torch

import torch.nn as nn

import torch.nn.functional as F

from datasets import Dataset, load_dataset

from tqdm.auto import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:
    SDPBackend = None
    sdpa_kernel = None



DEFAULT_DATASET_NAME = "Salesforce/wikitext"



DEFAULT_DATASET_CONFIG = "wikitext-2-raw-v1"



DEFAULT_SPLIT = "validation"



EPS = 1e-12



@dataclass(frozen=True)
class QuantSpec:
    """A weight quantization format and its scale-sharing granularity."""
    quant_type: str
    granularity: str

    @property
    def key(self) -> str:
        """Return the canonical quantization-format and granularity identifier."""
        return f"{self.quant_type}_{self.granularity}"



@dataclass(frozen=True)
class WhiteningTransform:
    """A normalized whitening matrix, inverse, scalar variance, and verification errors."""
    constraint: str
    p: torch.Tensor
    p_inv: torch.Tensor
    sigma_sq: float
    verification: dict[str, float]



@dataclass(frozen=True)
class HessianSketch:
    """Rademacher channel probes and their averaged downstream Hessian-vector products."""
    probes: torch.Tensor
    h_probes: torch.Tensor



def stable_seed(text: str) -> int:
    """Map a string deterministically to a reproducible random seed."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**31)



def parse_csv_list(
    raw_value: str | None,
    allowed_values: list[str],
    field_name: str,
) -> list[str]:
    """Parse a comma-separated list or return the supplied defaults."""
    if raw_value is None:
        return allowed_values.copy()
    values = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not values:
        raise ValueError(f"{field_name} must not be empty")
    invalid = [item for item in values if item not in allowed_values]
    if invalid:
        raise ValueError(
            f"Unsupported {field_name}: {invalid}. Allowed values: {allowed_values}"
        )
    return values



def resolve_device(device: str | None) -> torch.device:
    """Resolve an explicit device or choose CUDA when available."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")



def short_model_name(model_name: str) -> str:
    """Return the last path component of a model identifier."""
    return model_name.split("/")[-1]



def force_math_attention_kernels():
    """Select math attention kernels so second derivatives remain available."""
    if not torch.cuda.is_available():
        return nullcontext()
    if sdpa_kernel is not None and SDPBackend is not None:
        try:
            return sdpa_kernel([SDPBackend.MATH])
        except TypeError:
            return sdpa_kernel(backends=[SDPBackend.MATH])
    if hasattr(torch.backends.cuda, "sdp_kernel"):
        return torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_math=True,
            enable_mem_efficient=False,
        )
    return nullcontext()



def load_model_and_tokenizer(
    model_name: str,
    device: torch.device,
) -> tuple[nn.Module, Any]:
    """Load a reference causal LM and tokenizer in FP32 on the requested device."""
    local_files_only = (
        os.environ.get("HF_HUB_OFFLINE") == "1"
        or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    model.config.use_cache = False
    model.to(device)
    model.eval()
    return model, tokenizer



def build_calibration_batches(
    tokenizer,
    dataset_name: str = DEFAULT_DATASET_NAME,
    dataset_config: str = DEFAULT_DATASET_CONFIG,
    split: str = DEFAULT_SPLIT,
    num_samples: int = 128,
    max_seqlen: int = 1024,
    batch_size: int = 2,
    seed: int = 0,
) -> list[dict[str, torch.Tensor]]:
    """Return deterministic shifted-token validation batches from shuffled WikiText2 documents."""
    dataset_path = Path(dataset_name)
    if dataset_path.is_file() and dataset_path.suffix == ".arrow":
        dataset = Dataset.from_file(str(dataset_path))
    else:
        dataset = load_dataset(dataset_name, dataset_config, split=split)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    order = torch.randperm(len(dataset), generator=generator).tolist()

    eos_token_id = tokenizer.eos_token_id
    chunk_len = max_seqlen + 1
    required_tokens = num_samples * chunk_len
    token_buffer: list[int] = []

    for index in order:
        text = dataset[index]["text"]
        if not text.strip():
            continue
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not token_ids:
            continue
        token_buffer.extend(token_ids)
        if eos_token_id is not None:
            token_buffer.append(eos_token_id)
        if len(token_buffer) >= required_tokens:
            break

    available = len(token_buffer) // chunk_len
    if available < num_samples:
        raise RuntimeError(
            f"Only collected {available} sequences from {dataset_name}/{dataset_config}"
        )

    tokens = torch.tensor(
        token_buffer[: num_samples * chunk_len],
        dtype=torch.long,
    ).view(num_samples, chunk_len)
    input_ids = tokens[:, :-1].contiguous()
    labels = tokens[:, 1:].contiguous()
    loss_mask = torch.ones_like(labels, dtype=torch.bool)

    batches: list[dict[str, torch.Tensor]] = []
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batches.append(
            {
                "input_ids": input_ids[start:end].clone(),
                "labels": labels[start:end].clone(),
                "loss_mask": loss_mask[start:end].clone(),
            }
        )
    return batches



def get_quantizable_linear_layers(
    model: nn.Module,
    layer_pattern: str | None = None,
    max_layers: int | None = None,
) -> OrderedDict[str, nn.Linear]:
    """Return ordered linear modules matching the filter, excluding the language-model head."""
    layers: OrderedDict[str, nn.Linear] = OrderedDict()
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name == "lm_head" or name.endswith("lm_head"):
            continue
        if layer_pattern is not None and layer_pattern not in name:
            continue
        layers[name] = module
        if max_layers is not None and len(layers) >= max_layers:
            break
    return layers



def masked_token_rows(tensor: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """Select supervised token rows from a tensor using the supplied boolean mask."""
    mask = loss_mask.bool()
    if tensor.dim() == mask.dim() + 1 and tuple(tensor.shape[:-1]) == tuple(mask.shape):
        return tensor[mask]
    if tensor.dim() == mask.dim() and tuple(tensor.shape) == tuple(mask.shape):
        return tensor[mask]
    if tensor.dim() >= 2 and tensor.shape[0] == mask.numel():
        return tensor[mask.reshape(-1)]
    raise ValueError(
        f"Cannot apply mask with shape {tuple(mask.shape)} to tensor {tuple(tensor.shape)}"
    )



def make_token_signs_like(
    tensor: torch.Tensor,
    loss_mask: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    """Generate seeded token signs with the layout and mask required by a Hessian probe."""
    mask = loss_mask.bool()
    if tensor.dim() == mask.dim() + 1 and tuple(tensor.shape[:-1]) == tuple(mask.shape):
        signs = sample_rademacher(
            tuple(mask.shape),
            seed=seed,
            device=tensor.device,
            dtype=tensor.dtype,
        )
        return signs * mask.to(device=tensor.device, dtype=tensor.dtype)
    if tensor.dim() >= 2 and tensor.shape[0] == mask.numel():
        flat_mask = mask.reshape(-1)
        signs = sample_rademacher(
            (flat_mask.numel(),),
            seed=seed,
            device=tensor.device,
            dtype=tensor.dtype,
        )
        return signs * flat_mask.to(device=tensor.device, dtype=tensor.dtype)
    raise ValueError(
        f"Cannot build token signs for tensor {tuple(tensor.shape)} with mask {tuple(mask.shape)}"
    )



def compute_masked_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Return summed hard-label language-model loss over supervised tokens."""
    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        labels.reshape(-1),
        reduction="none",
    ).reshape_as(labels)
    return (token_loss * loss_mask.to(dtype=token_loss.dtype)).sum()



def sample_rademacher(
    shape: tuple[int, ...],
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Generate seeded plus-or-minus-one probes with the requested shape, dtype, and device."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    values = torch.randint(0, 2, shape, generator=generator, dtype=torch.int64)
    return (values.to(device=device) * 2 - 1).to(dtype=dtype)



def collect_layer_statistics(
    model: nn.Module,
    layer_name: str,
    layer_module: nn.Linear,
    batches: list[dict[str, torch.Tensor]],
    device: torch.device,
    num_hessian_probes: int,
) -> tuple[HessianSketch, torch.Tensor, dict[str, float]]:
    """Return a reduced Hessian sketch, input covariance, and token counts for a single layer."""
    d_in = layer_module.in_features
    d_out = layer_module.out_features
    stats_device = layer_module.weight.device
    c_x_sum = torch.zeros(d_in, d_in, dtype=torch.float64, device=stats_device)
    h_probe_sum = torch.zeros(num_hessian_probes, d_out, dtype=torch.float64, device=stats_device)
    probe_seed = stable_seed(f"{layer_name}:channel-probes")
    channel_probes = sample_rademacher(
        (num_hessian_probes, d_out),
        probe_seed,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    total_tokens = 0

    progress_bar = tqdm(
        batches,
        total=len(batches),
        desc=f"[stats] {layer_name}",
        leave=False,
        unit="batch",
    )
    for batch_index, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        loss_mask = batch["loss_mask"].to(device).bool()
        if int(loss_mask.sum().item()) == 0:
            continue

        captured: dict[str, torch.Tensor] = {}

        def _hook(
            _module: nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            captured["x"] = inputs[0]
            captured["z"] = output

        with force_math_attention_kernels():
            handle = layer_module.register_forward_hook(_hook)
            logits = model(input_ids=input_ids, use_cache=False).logits
            handle.remove()

            z = captured["z"]
            loss = compute_masked_lm_loss(logits, labels, loss_mask)
            (grad_z,) = torch.autograd.grad(loss, z, create_graph=True)

            x_tokens = masked_token_rows(captured["x"], loss_mask).detach().to(torch.float64)
            c_x_sum += x_tokens.T @ x_tokens
            token_count = x_tokens.shape[0]
            total_tokens += token_count

            for probe_index in range(num_hessian_probes):
                channel_probe = channel_probes[probe_index].to(device=device, dtype=z.dtype)
                token_signs = make_token_signs_like(
                    z,
                    loss_mask,
                    stable_seed(f"{layer_name}:token:{batch_index}:{probe_index}"),
                )
                view_shape = [1] * (z.dim() - 1) + [d_out]
                v = token_signs.unsqueeze(-1) * channel_probe.view(*view_shape)
                scalar = (grad_z * v).sum()
                (hvp,) = torch.autograd.grad(
                    scalar,
                    z,
                    retain_graph=(probe_index < num_hessian_probes - 1),
                )
                weighted_hvp = masked_token_rows(
                    hvp * token_signs.unsqueeze(-1),
                    loss_mask,
                ).sum(dim=0)
                h_probe_sum[probe_index] += weighted_hvp.detach().to(torch.float64)

        model.zero_grad(set_to_none=True)
        progress_bar.set_postfix(tokens=total_tokens)
    progress_bar.close()

    if total_tokens == 0:
        raise RuntimeError(f"No supervised tokens were available for {layer_name}")

    c_x = (c_x_sum / total_tokens).float()
    c_x = 0.5 * (c_x + c_x.T)
    sketch = HessianSketch(
        probes=channel_probes.to(device=stats_device),
        h_probes=(h_probe_sum / total_tokens).float(),
    )
    summary = {
        "input_dim": int(d_in),
        "output_dim": int(d_out),
        "token_count": int(total_tokens),
    }
    return sketch, c_x, summary



def estimate_hessian_trace(sketch: HessianSketch) -> float:
    """Estimate the downstream curvature trace from matching probes and Hessian products."""
    probes = sketch.probes.to(torch.float64)
    h_probes = sketch.h_probes.to(torch.float64)
    values = (probes * h_probes).sum(dim=1)
    return float(values.mean().item())



def estimate_trace_hm(
    sketch: HessianSketch,
    matrix: torch.Tensor,
) -> float:
    """Estimate the trace of a curvature-matrix product using the supplied Hessian sketch."""
    probes = sketch.probes.to(torch.float64)
    h_probes = sketch.h_probes.to(torch.float64)
    m64 = matrix.to(torch.float64)
    values = []
    for probe, h_probe in zip(probes, h_probes, strict=True):
        values.append(torch.dot(h_probe, m64 @ probe).item())
    return float(sum(values) / max(len(values), 1))



def compute_sigma_squared(c_x: torch.Tensor) -> float:
    """Return the input covariance trace normalized by its dimension."""
    return float(torch.trace(c_x.to(torch.float64)).item() / max(c_x.shape[0], 1))



def compute_whitening_transform(
    c_x: torch.Tensor,
    constraint: str,
) -> WhiteningTransform:
    """Construct the requested normalized whitening transform from an input covariance."""
    dim = c_x.shape[0]
    dev = c_x.device
    eye = torch.eye(dim, dtype=torch.float32, device=dev)
    if constraint == "none":
        sigma_sq = compute_sigma_squared(c_x)
        return WhiteningTransform(
            constraint=constraint,
            p=eye,
            p_inv=eye,
            sigma_sq=sigma_sq,
            verification={
                "relative_frobenius_error": 0.0,
                "relative_diag_error": 0.0,
                "max_abs_offdiag": 0.0,
            },
        )

    c64 = c_x.to(torch.float64)
    reg_eps = float((c64.diagonal().abs().mean().clamp(min=1e-12) * 1e-6).item())
    c64 = c64 + reg_eps * torch.eye(dim, dtype=torch.float64, device=dev)
    eigenvalues, u = torch.linalg.eigh(c64)
    eigenvalues = eigenvalues.clamp(min=1e-10)

    if constraint == "vol-preserving":
        sigma_sq = float(torch.exp(eigenvalues.log().mean()).item())
    elif constraint == "trace-preserving":
        sigma_sq = float(eigenvalues.mean().item())
    elif constraint == "frobenius":
        sigma_sq = float(dim / (1.0 / eigenvalues).sum().item())
    else:
        raise ValueError(f"Unknown constraint: {constraint}")

    sigma = sigma_sq**0.5
    lambda_inv_sqrt = torch.diag(1.0 / eigenvalues.sqrt())
    lambda_sqrt = torch.diag(eigenvalues.sqrt())
    p = sigma * (lambda_inv_sqrt @ u.T)
    p_inv = (1.0 / sigma) * (u @ lambda_sqrt)
    p_f32 = p.float()
    p_inv_f32 = p_inv.float()
    return WhiteningTransform(
        constraint=constraint,
        p=p_f32,
        p_inv=p_inv_f32,
        sigma_sq=sigma_sq,
        verification=verify_whitening(p_f32, c_x.float(), sigma_sq),
    )



def verify_whitening(
    p: torch.Tensor,
    c_x: torch.Tensor,
    sigma_sq: float,
) -> dict[str, float]:
    """Return numerical residuals for the supplied covariance and whitening transform."""
    dev = p.device
    p64 = p.to(torch.float64)
    c64 = c_x.to(torch.float64)
    pcpt = p64 @ c64 @ p64.T
    dim = c_x.shape[0]
    target = sigma_sq * torch.eye(dim, dtype=torch.float64, device=dev)
    residual = pcpt - target
    offdiag_mask = ~torch.eye(dim, dtype=torch.bool, device=dev)
    max_offdiag = residual[offdiag_mask].abs().max().item() if dim > 1 else 0.0
    rel_frob = (residual.norm() / (target.norm() + EPS)).item()
    rel_diag = (
        (torch.diag(pcpt) - sigma_sq).abs().mean().item() / (abs(sigma_sq) + EPS)
    )
    return {
        "relative_frobenius_error": float(rel_frob),
        "relative_diag_error": float(rel_diag),
        "max_abs_offdiag": float(max_offdiag),
    }



def compute_masked_forward_kl(
    clean_logits: torch.Tensor,
    quant_logits: torch.Tensor,
    loss_mask: torch.Tensor,
    token_chunk_size: int = 4096,
) -> tuple[float, int]:
    """Return the summed reference-to-quantized KL and the number of supervised tokens."""
    mask = loss_mask.reshape(-1).bool()
    if not mask.any():
        return 0.0, 0

    clean_flat = clean_logits.float().reshape(-1, clean_logits.shape[-1])[mask]
    quant_flat = quant_logits.float().reshape(-1, quant_logits.shape[-1])[mask]
    total_tokens = clean_flat.shape[0]
    kl_sum = 0.0
    for start in range(0, total_tokens, token_chunk_size):
        end = min(start + token_chunk_size, total_tokens)
        clean_log_p = F.log_softmax(clean_flat[start:end], dim=-1)
        quant_log_q = F.log_softmax(quant_flat[start:end], dim=-1)
        kl_sum += float(
            F.kl_div(quant_log_q, clean_log_p, reduction="sum", log_target=True).item()
        )
    return kl_sum, total_tokens

