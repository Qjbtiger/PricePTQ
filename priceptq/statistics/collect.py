"""Joint FP32 statistics collection for configuration selection."""

from __future__ import annotations

import torch

import torch.nn as nn

from tqdm.auto import tqdm

from priceptq.statistics.common import compute_masked_lm_loss, force_math_attention_kernels, masked_token_rows, short_model_name


def collect_weight_only_statistics(
    *,
    model: nn.Module,
    layers: dict[str, nn.Linear],
    batches: list[dict[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """Return input second moments and empirical-Fisher traces collected jointly over the supplied layers."""
    input_sq_sums = {
        name: torch.zeros((), dtype=torch.float64, device=module.weight.device)
        for name, module in layers.items()
    }
    fisher_trace_sums = {
        name: torch.zeros((), dtype=torch.float64, device=module.weight.device)
        for name, module in layers.items()
    }
    captured_outputs: dict[str, torch.Tensor] = {}
    current_mask: torch.Tensor | None = None
    total_tokens = 0
    handles = []

    for name, module in layers.items():
        def _make_hook(layer_name: str):
            def _hook(
                _module: nn.Module,
                inputs: tuple[torch.Tensor, ...],
                output: torch.Tensor,
            ) -> None:
                if current_mask is None:
                    raise RuntimeError("Calibration mask is unavailable while collecting statistics.")
                captured_outputs[layer_name] = output
                x_tokens = masked_token_rows(inputs[0].detach(), current_mask)
                input_sq_sums[layer_name] += x_tokens.to(torch.float64).square().sum()

            return _hook

        handles.append(module.register_forward_hook(_make_hook(name)))

    try:
        for batch in tqdm(
            batches,
            total=len(batches),
            desc="[weight-only] inline all layers",
            unit="batch",
        ):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            current_mask = batch["loss_mask"].to(device).bool()
            token_count = int(current_mask.sum().item())
            if token_count == 0:
                continue

            captured_outputs.clear()
            with force_math_attention_kernels():
                logits = model(input_ids=input_ids, use_cache=False).logits
                loss = compute_masked_lm_loss(logits, labels, current_mask)

            missing_outputs = [name for name in layers if name not in captured_outputs]
            if missing_outputs:
                raise RuntimeError(f"No differentiable layer output captured for: {missing_outputs[:8]}")

            gradients = torch.autograd.grad(
                loss,
                tuple(captured_outputs[name] for name in layers),
                create_graph=False,
            )
            for layer_name, gradient in zip(layers, gradients, strict=True):
                g_tokens = masked_token_rows(gradient, current_mask).detach().to(torch.float64)
                fisher_trace_sums[layer_name] += g_tokens.square().sum()
            total_tokens += token_count

            model.zero_grad(set_to_none=True)
            del input_ids, labels, logits, loss, gradients
            captured_outputs.clear()
    finally:
        for handle in handles:
            handle.remove()

    if total_tokens == 0:
        raise RuntimeError("No supervised tokens were available for statistics collection.")

    statistics = {}
    for name, module in layers.items():
        input_cov_trace = float((input_sq_sums[name] / total_tokens).item())
        statistics[name] = {
            "input_dim": int(module.in_features),
            "output_dim": int(module.out_features),
            "token_count": int(total_tokens),
            "input_cov_trace": input_cov_trace,
            "input_cov_sigma_sq_none": input_cov_trace / float(module.in_features),
            "hessian_trace": max(
                float((fisher_trace_sums[name] / total_tokens).item()),
                0.0,
            ),
        }

    if device.type == "cuda":
        torch.cuda.empty_cache()
    return statistics



def _per_tensor_weight_stats(
    abs_weight: torch.Tensor,
    sigma_sq: float,
    trace_upper_bound: float,
    input_dim: int,
) -> dict[str, float]:
    absmax = float(abs_weight.max().item())
    range_sq = absmax * absmax
    return {
        "absmax": absmax,
        "range_sq_proxy": range_sq,
        "difficulty_proxy": float(sigma_sq * range_sq * trace_upper_bound),
        "input_dim_scaled_difficulty_proxy": float(
            sigma_sq * input_dim * range_sq * trace_upper_bound
        ),
    }



def build_layer_record(
    *,
    model_name: str,
    layer_name: str,
    layer_module: nn.Linear,
    statistics: dict[str, float],
) -> dict[str, object]:
    """Combine a layer weight range and collected statistics into the solver record schema."""
    input_dim = int(statistics["input_dim"])
    sigma_sq = float(statistics["input_cov_sigma_sq_none"])
    trace_upper_bound = float(statistics["hessian_trace"])
    weight_stats = _per_tensor_weight_stats(
        layer_module.weight.detach().float().abs(),
        sigma_sq=sigma_sq,
        trace_upper_bound=trace_upper_bound,
        input_dim=input_dim,
    )
    return {
        "model_name": model_name,
        "model_short_name": short_model_name(model_name),
        "layer_name": layer_name,
        "input_dim": input_dim,
        "output_dim": int(statistics["output_dim"]),
        "token_count": int(statistics["token_count"]),
        "input_cov_trace": float(statistics["input_cov_trace"]),
        "input_cov_sigma_sq_none": sigma_sq,
        "hessian_trace_estimate": trace_upper_bound,
        "hessian_trace_upper_bound": trace_upper_bound,
        "hessian_spectral_norm_upper_bound": trace_upper_bound,
        "hessian_trace_estimator": "gn_fisher_trace_upper_bound",
        "constraints": {
            "none": {
                "sigma_sq": sigma_sq,
                "weight_ranges": {"per-tensor": weight_stats},
            }
        },
    }

