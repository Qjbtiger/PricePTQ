"""Transformation candidates and selectors used in the main paper."""

from __future__ import annotations

import json

import math

from collections import OrderedDict

from contextlib import contextmanager

from dataclasses import dataclass

from pathlib import Path

from typing import Any

import torch

import torch.nn as nn

import torch.nn.functional as F

from tqdm.auto import tqdm


TRANSFORMATIONS = (
    "none",
    "awq-style-scaling",
    "smoothquant-style-scaling",
    "hadamard-rotation",
    "whitening-normalized-p",
)



def is_power_of_two(value: int) -> bool:
    """Return whether an input dimension is supported by the power-of-two Hadamard transform."""
    return value > 0 and (value & (value - 1)) == 0



def transformations_for_input_dim(input_dim: int) -> tuple[str, ...]:
    """Return feasible paper transformation names for the supplied input dimension."""
    if is_power_of_two(input_dim):
        return TRANSFORMATIONS
    return tuple(name for name in TRANSFORMATIONS if name != "hadamard-rotation")



@dataclass(frozen=True)
class TransformState:
    """A transformation name, diagonal scale or rotation signs, and reproducibility metadata."""
    name: str
    kind: str
    vector: torch.Tensor | None = None
    signs: torch.Tensor | None = None
    metadata: dict[str, Any] | None = None



@dataclass(frozen=True)
class QuantizedLayer:
    """A transformed fake-quant weight, bias, and the corresponding input transformation."""
    transform: TransformState
    quantized_weight: torch.Tensor
    bias: torch.Tensor | None



class TransformedQuantLinear(nn.Module):
    """Apply an input transform followed by a fake-quant linear projection."""
    def __init__(self, quantized: QuantizedLayer):
        super().__init__()
        self.transform = quantized.transform
        self.register_buffer("quantized_weight", quantized.quantized_weight.detach())
        if quantized.bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", quantized.bias.detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the linear projection after applying this module's selected quantization representation."""
        x_tilde = apply_input_transform(x, self.transform)
        return F.linear(x_tilde, self.quantized_weight, self.bias)



def get_quantizable_linear_layers(
    model: nn.Module,
    *,
    layer_pattern: str | None,
    max_layers: int | None,
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



def masked_rows(tensor: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """Select calibration rows at supervised token positions."""
    mask = loss_mask.bool()
    if tensor.dim() == mask.dim() + 1 and tuple(tensor.shape[:-1]) == tuple(mask.shape):
        return tensor[mask]
    raise ValueError(f"Cannot apply mask {tuple(mask.shape)} to tensor {tuple(tensor.shape)}.")



def collect_layer_inputs(
    model: nn.Module,
    layers: OrderedDict[str, nn.Linear],
    batches: list[dict[str, torch.Tensor]],
    *,
    device: torch.device,
    max_price_tokens: int,
) -> dict[str, torch.Tensor]:
    """Capture up to the requested number of supervised input rows for every layer."""
    captured: dict[str, list[torch.Tensor]] = {name: [] for name in layers}
    counts = {name: 0 for name in layers}
    handles = []

    for name, module in layers.items():
        def _make_hook(layer_name: str):
            def _hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...], _output: torch.Tensor) -> None:
                if counts[layer_name] >= max_price_tokens:
                    return
                current_mask = collect_layer_inputs.current_mask
                x = masked_rows(inputs[0].detach(), current_mask)
                remaining = max_price_tokens - counts[layer_name]
                if x.shape[0] > remaining:
                    x = x[:remaining]
                captured[layer_name].append(x.float().cpu())
                counts[layer_name] += int(x.shape[0])
            return _hook

        handles.append(module.register_forward_hook(_make_hook(name)))

    try:
        for batch in tqdm(batches, desc="[price] collect inputs", unit="batch"):
            collect_layer_inputs.current_mask = batch["loss_mask"].to(device)
            with torch.no_grad():
                model(input_ids=batch["input_ids"].to(device), use_cache=False)
            if all(count >= max_price_tokens for count in counts.values()):
                break
    finally:
        for handle in handles:
            handle.remove()

    result = {}
    for name, chunks in captured.items():
        if not chunks:
            raise RuntimeError(f"No calibration inputs collected for {name}.")
        result[name] = torch.cat(chunks, dim=0)
    return result



def groupwise_symmetric_quantize(weight: torch.Tensor, *, bitwidth: int, group_size: int) -> torch.Tensor:
    """Return symmetric INT fake-quant weights with shared scales along input groups."""
    qmin = -(2 ** (bitwidth - 1))
    qmax = 2 ** (bitwidth - 1) - 1
    original_shape = weight.shape
    if group_size <= 0:
        flat = weight.reshape(1, -1)
    else:
        if original_shape[-1] % group_size != 0:
            raise ValueError(f"Input dim {original_shape[-1]} is not divisible by group_size={group_size}.")
        flat = weight.reshape(-1, group_size)
    scale = flat.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / float(qmax)
    quantized = torch.clamp(torch.round(flat / scale), qmin, qmax) * scale
    return quantized.reshape(original_shape)



def normalized_fwht(x: torch.Tensor) -> torch.Tensor:
    """Apply an orthonormal fast Walsh-Hadamard transform along the last dimension."""
    dim = x.shape[-1]
    if dim <= 0 or dim & (dim - 1):
        raise ValueError(f"Hadamard transform requires power-of-two dim, got {dim}.")
    original_shape = x.shape
    y = x.reshape(-1, dim)
    h = 1
    while h < dim:
        y = y.reshape(-1, dim // (2 * h), 2, h)
        left = y[:, :, 0, :]
        right = y[:, :, 1, :]
        y = torch.cat((left + right, left - right), dim=-1)
        y = y.reshape(-1, dim)
        h *= 2
    return y.reshape(original_shape) / math.sqrt(float(dim))



def apply_input_transform(x: torch.Tensor, transform: TransformState) -> torch.Tensor:
    """Apply the given identity, diagonal, or Hadamard transform to input rows."""
    if transform.kind == "identity":
        return x
    if transform.kind == "diagonal":
        if transform.vector is None:
            raise RuntimeError("Diagonal transform is missing vector.")
        return x * transform.vector.to(device=x.device, dtype=x.dtype)
    if transform.kind == "hadamard":
        if transform.signs is None:
            raise RuntimeError("Hadamard transform is missing signs.")
        signed = x * transform.signs.to(device=x.device, dtype=x.dtype)
        return normalized_fwht(signed)
    raise ValueError(f"Unsupported transform kind: {transform.kind}")



def apply_weight_transform(weight: torch.Tensor, transform: TransformState) -> torch.Tensor:
    """Map weights into coordinates equivalent to the supplied input transform."""
    if transform.kind == "identity":
        return weight
    if transform.kind == "diagonal":
        if transform.vector is None:
            raise RuntimeError("Diagonal transform is missing vector.")
        return weight / transform.vector.to(device=weight.device, dtype=weight.dtype).clamp(min=1e-6)
    if transform.kind == "hadamard":
        if transform.signs is None:
            raise RuntimeError("Hadamard transform is missing signs.")
        signed = weight * transform.signs.to(device=weight.device, dtype=weight.dtype)
        return normalized_fwht(signed)
    raise ValueError(f"Unsupported transform kind: {transform.kind}")



def build_transform(
    name: str,
    *,
    layer_name: str,
    x_sample: torch.Tensor,
    weight: torch.Tensor,
    seed: int,
) -> TransformState:
    """Build a paper transformation from calibration inputs, weights, and deterministic signs."""
    input_dim = weight.shape[1]
    if name == "none":
        return TransformState(name=name, kind="identity", metadata={})

    x_abs_mean = x_sample.abs().mean(dim=0).clamp(min=1e-6)
    x_abs_max = x_sample.abs().amax(dim=0).clamp(min=1e-6)
    w_abs_max = weight.detach().float().abs().amax(dim=0).clamp(min=1e-6).cpu()

    if name == "awq-style-scaling":
        vector = _normalize_scale(x_abs_mean.sqrt())
        return TransformState(name=name, kind="diagonal", vector=vector, metadata={"scale_source": "sqrt_mean_abs_activation"})
    if name == "smoothquant-style-scaling":
        alpha = 0.5
        vector = _normalize_scale((x_abs_max**alpha) / (w_abs_max ** (1.0 - alpha)))
        return TransformState(name=name, kind="diagonal", vector=vector, metadata={"alpha": alpha})
    if name == "whitening-normalized-p":
        rms = x_sample.float().pow(2).mean(dim=0).sqrt().clamp(min=1e-6)
        sigma = torch.exp(rms.log().mean())
        vector = (sigma / rms).clamp(min=1e-3, max=1e3)
        return TransformState(
            name=name,
            kind="diagonal",
            vector=vector,
            metadata={"approximation": "diagonal covariance whitening with geometric-mean normalization"},
        )
    if name == "hadamard-rotation":
        if not is_power_of_two(input_dim):
            raise ValueError(f"Hadamard rotation requires power-of-two input dim, got {input_dim} at {layer_name}.")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(stable_seed(f"{layer_name}:{seed}:hadamard"))
        signs = torch.randint(0, 2, (input_dim,), generator=generator, dtype=torch.int8).float()
        signs = signs.mul_(2.0).sub_(1.0)
        return TransformState(name=name, kind="hadamard", signs=signs, metadata={"seed": seed})
    raise ValueError(f"Unsupported transformation: {name}")



def _normalize_scale(vector: torch.Tensor) -> torch.Tensor:
    vector = vector.float().clamp(min=1e-6)
    return (vector / torch.exp(vector.log().mean())).clamp(min=1e-3, max=1e3).cpu()



def stable_seed(text: str) -> int:
    """Map a string deterministically to a reproducible random seed."""
    import hashlib

    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**31)



def quantize_layer(
    module: nn.Linear,
    transform: TransformState,
    *,
    bitwidth: int,
    group_size: int,
    device: torch.device,
) -> QuantizedLayer:
    """Construct the requested layer candidate and its quantized weight representation."""
    weight = module.weight.detach().float().to(device)
    transformed = apply_weight_transform(weight, transform)
    quantized = groupwise_symmetric_quantize(transformed, bitwidth=bitwidth, group_size=group_size)
    return QuantizedLayer(
        transform=transform,
        quantized_weight=quantized.to(device="cpu", dtype=module.weight.dtype),
        bias=None if module.bias is None else module.bias.detach().cpu(),
    )



def price_transformations(
    *,
    layers: OrderedDict[str, nn.Linear],
    layer_inputs: dict[str, torch.Tensor],
    stats_by_layer: dict[str, Any],
    bitwidth: int,
    group_size: int,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, str], dict[str, dict[str, Any]], dict[str, QuantizedLayer], dict[str, dict[str, QuantizedLayer]]]:
    """Return minimum-price choices, candidate scores, selected weights, and the candidate cache."""
    selected: dict[str, str] = {}
    prices: dict[str, dict[str, Any]] = {}
    selected_quantized: dict[str, QuantizedLayer] = {}
    quantized_cache: dict[str, dict[str, QuantizedLayer]] = {}

    for layer_name, module in tqdm(layers.items(), desc="[price] transformations", unit="layer"):
        x = layer_inputs[layer_name].to(device)
        weight = module.weight.detach().float().to(device)
        bias = None if module.bias is None else module.bias.detach().float().to(device)
        y_ref = F.linear(x, weight, bias)
        hessian_trace = _hessian_trace(stats_by_layer.get(layer_name, {}))

        layer_prices: dict[str, Any] = {}
        best_name = None
        best_price = None
        best_quantized = None
        quantized_cache[layer_name] = {}
        for transform_name in transformations_for_input_dim(int(weight.shape[1])):
            transform = build_transform(
                transform_name,
                layer_name=layer_name,
                x_sample=layer_inputs[layer_name],
                weight=weight.detach().cpu(),
                seed=seed,
            )
            quantized = quantize_layer(
                module,
                transform,
                bitwidth=bitwidth,
                group_size=group_size,
                device=device,
            )
            quantized_cache[layer_name][transform_name] = quantized
            candidate_weight = quantized.quantized_weight.to(device=device, dtype=torch.float32)
            y_hat = F.linear(apply_input_transform(x, transform), candidate_weight, bias)
            mse = float((y_hat - y_ref).float().pow(2).mean().detach().cpu().item())
            del candidate_weight
            rho_hat = 0.5 * hessian_trace * mse
            layer_prices[transform_name] = {
                "rho_hat": rho_hat,
                "local_output_mse": mse,
                "hessian_trace": hessian_trace,
                "metadata": transform.metadata or {},
            }
            if best_price is None or rho_hat < best_price:
                best_name = transform_name
                best_price = rho_hat
                best_quantized = quantized

        if best_name is None or best_quantized is None:
            raise RuntimeError(f"No transformation was priced for {layer_name}.")
        selected[layer_name] = best_name
        selected_quantized[layer_name] = best_quantized
        prices[layer_name] = {
            "selected": best_name,
            "candidates": layer_prices,
        }
    return selected, prices, selected_quantized, quantized_cache



def linear_cka_similarity(x: torch.Tensor, y: torch.Tensor, *, eps: float = 1e-12) -> float:
    """Return centered linear CKA between matching reference and quantized output matrices."""
    if x.shape != y.shape:
        raise ValueError(f"CKA inputs must have matching shapes, got {tuple(x.shape)} and {tuple(y.shape)}.")
    if x.numel() == 0:
        return 0.0
    x_flat = x.float().reshape(-1, x.shape[-1])
    y_flat = y.float().reshape(-1, y.shape[-1])
    x_centered = x_flat - x_flat.mean(dim=0, keepdim=True)
    y_centered = y_flat - y_flat.mean(dim=0, keepdim=True)
    x_gram = x_centered @ x_centered.t()
    y_gram = y_centered @ y_centered.t()
    numerator = (x_gram * y_gram).sum()
    denominator = torch.sqrt((x_gram * x_gram).sum() * (y_gram * y_gram).sum())
    if not torch.isfinite(denominator) or float(denominator.detach().cpu().item()) <= eps:
        return 0.0
    score = numerator / denominator.clamp(min=eps)
    return float(torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0).detach().cpu().item())



def calm_cka_selection(
    *,
    layers: OrderedDict[str, nn.Linear],
    layer_inputs: dict[str, torch.Tensor],
    quantized_cache: dict[str, dict[str, QuantizedLayer]],
    device: torch.device,
) -> tuple[dict[str, str], dict[str, dict[str, Any]], dict[str, QuantizedLayer]]:
    """Return maximum-CKA transformation choices, scores, and selected quantized weights."""
    selected: dict[str, str] = {}
    scores: dict[str, dict[str, Any]] = {}
    selected_quantized: dict[str, QuantizedLayer] = {}

    for layer_name, module in tqdm(layers.items(), desc="[calm-cka] transformations", unit="layer"):
        x = layer_inputs[layer_name].to(device)
        weight = module.weight.detach().float().to(device)
        bias = None if module.bias is None else module.bias.detach().float().to(device)
        y_ref = F.linear(x, weight, bias)

        best_name = None
        best_score = None
        best_quantized = None
        layer_scores: dict[str, Any] = {}
        for transform_name, quantized in quantized_cache[layer_name].items():
            candidate_weight = quantized.quantized_weight.to(device=device, dtype=torch.float32)
            y_hat = F.linear(
                apply_input_transform(x, quantized.transform),
                candidate_weight,
                bias,
            )
            score = linear_cka_similarity(y_ref, y_hat)
            del candidate_weight
            layer_scores[transform_name] = {
                "linear_cka": score,
                "metadata": quantized.transform.metadata or {},
            }
            if best_score is None or score > best_score:
                best_name = transform_name
                best_score = score
                best_quantized = quantized

        if best_name is None or best_quantized is None:
            raise RuntimeError(f"No CALM-CKA transformation was scored for {layer_name}.")
        selected[layer_name] = best_name
        selected_quantized[layer_name] = best_quantized
        scores[layer_name] = {
            "selected": best_name,
            "candidates": layer_scores,
        }

    return selected, scores, selected_quantized



def _hessian_trace(entry: dict[str, Any]) -> float:
    for key in ("hessian_trace", "hessian_trace_estimate", "hessian_trace_upper_bound"):
        if key in entry:
            return max(float(entry[key]), 0.0)
    return 1.0



def random_quantized_selection(
    *,
    layer_names: list[str],
    quantized_cache: dict[str, dict[str, QuantizedLayer]],
    seed: int,
) -> tuple[dict[str, str], dict[str, QuantizedLayer]]:
    """Sample one cached candidate per layer with the supplied seed and return choices and weights."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    selected: dict[str, str] = {}
    quantized: dict[str, QuantizedLayer] = {}
    for layer_name in tqdm(layer_names, desc=f"[random] select seed={seed}", unit="layer"):
        candidate_names = tuple(quantized_cache[layer_name].keys())
        transform_name = candidate_names[int(torch.randint(0, len(candidate_names), (1,), generator=generator).item())]
        selected[layer_name] = transform_name
        quantized[layer_name] = quantized_cache[layer_name][transform_name]
    return selected, quantized



def _get_parent_module(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent: nn.Module = model
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]



def fold_transform_into_weight(quantized: QuantizedLayer) -> torch.Tensor:
    """Return a dense weight equivalent to the candidate input transformation and quantizer."""
    transform = quantized.transform
    weight = quantized.quantized_weight.float()
    if transform.kind == "identity":
        return weight
    if transform.kind == "diagonal":
        if transform.vector is None:
            raise RuntimeError("Diagonal transform is missing vector.")
        vector = transform.vector.to(device=weight.device, dtype=weight.dtype)
        return weight * vector
    if transform.kind == "hadamard":
        if transform.signs is None:
            raise RuntimeError("Hadamard transform is missing signs.")
        signs = transform.signs.to(device=weight.device, dtype=weight.dtype)
        return normalized_fwht(weight) * signs
    raise ValueError(f"Unsupported transform kind: {transform.kind}")



def write_folded_fake_quant_weights_to_model(
    model: nn.Module,
    quantized_layers: dict[str, QuantizedLayer],
) -> None:
    """Write selected transformed weights into ordinary linear modules for HF export."""
    with torch.no_grad():
        for layer_name, quantized in tqdm(
            quantized_layers.items(),
            desc="[save] write folded fake-quant weights",
            unit="layer",
        ):
            parent, attr = _get_parent_module(model, layer_name)
            module = getattr(parent, attr)
            if not isinstance(module, nn.Linear):
                raise TypeError(f"Cannot save fake-quant weight for non-Linear module {layer_name}: {type(module)!r}")
            folded_weight = fold_transform_into_weight(quantized)
            module.weight.copy_(folded_weight.to(device=module.weight.device, dtype=module.weight.dtype))
            if module.bias is not None and quantized.bias is not None:
                module.bias.copy_(quantized.bias.to(device=module.bias.device, dtype=module.bias.dtype))



@contextmanager
def quantized_layer_context(model: nn.Module, quantized_layers: dict[str, QuantizedLayer]):
    """Temporarily replace selected linear modules with their fake-quant candidate modules."""
    originals = {}
    try:
        for layer_name, quantized in quantized_layers.items():
            parent, attr = _get_parent_module(model, layer_name)
            originals[layer_name] = (parent, attr, getattr(parent, attr))
            setattr(parent, attr, TransformedQuantLinear(quantized).to(quantized.quantized_weight.device))
        yield
    finally:
        for parent, attr, module in originals.values():
            setattr(parent, attr, module)



def stats_by_raw_layer(stats_path: str | Path) -> dict[str, Any]:
    """Load statistics indexed by the original Hugging Face layer names."""
    with Path(stats_path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {entry["layer_name"]: entry for entry in data.get("layers", [])}


collect_layer_inputs.current_mask = torch.empty(0, dtype=torch.bool)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))
