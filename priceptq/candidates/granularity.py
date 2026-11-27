"""Granularity candidates, metadata costs, and price-guided selection."""

from __future__ import annotations

import math

from collections import Counter, OrderedDict

from contextlib import contextmanager

from dataclasses import dataclass

from typing import Any

import torch

import torch.nn as nn

import torch.nn.functional as F

from tqdm.auto import tqdm

from priceptq.candidates.transformation import _hessian_trace, _mean


GRANULARITIES = (
    "per-tensor",
    "per-channel",
    "group-32",
    "group-64",
    "group-128",
    "block-wise",
)



DEPLOY_SUPPORTED_GRANULARITIES = {"group-128"}



BLOCK_ROWS = 64



BLOCK_COLS = 64



PRICE_TIE_TOLERANCE = 0.01



@dataclass(frozen=True)
class GranularityQuantizedLayer:
    """A fake-quant layer candidate and its granularity-dependent deployment metadata."""
    granularity: str
    quantized_weight: torch.Tensor
    bias: torch.Tensor | None
    cost: dict[str, Any]



class GranularityQuantLinear(nn.Module):
    """Evaluate a selected granularity candidate as a linear module."""
    def __init__(self, quantized: GranularityQuantizedLayer, *, device: torch.device):
        super().__init__()
        self.granularity = quantized.granularity
        self.register_buffer("quantized_weight", quantized.quantized_weight.detach().to(device))
        if quantized.bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", quantized.bias.detach().to(device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the linear projection after applying this module's selected quantization representation."""
        return F.linear(x, self.quantized_weight, self.bias)



@dataclass(frozen=True)
class PackedWeight:
    """Integer codes and scales used for the granularity latency proxy."""
    granularity: str
    qweight: torch.Tensor
    scales: torch.Tensor
    input_dim: int
    output_dim: int
    padded_input_dim: int
    padded_output_dim: int
    group_size: int | None = None



def _qrange(bitwidth: int) -> tuple[int, int]:
    if bitwidth <= 0:
        raise ValueError(f"bitwidth must be positive, got {bitwidth}.")
    return -(2 ** (bitwidth - 1)), 2 ** (bitwidth - 1) - 1



def quantize_weight_by_granularity(weight: torch.Tensor, *, bitwidth: int, granularity: str) -> torch.Tensor:
    """Return INT fake-quant weights using the requested scale-sharing scheme."""
    weight = weight.float()
    if granularity == "per-tensor":
        return _quantize_per_tensor(weight, bitwidth=bitwidth)
    if granularity == "per-channel":
        return _quantize_groupwise(weight, bitwidth=bitwidth, group_size=weight.shape[1])
    if granularity.startswith("group-"):
        return _quantize_groupwise(weight, bitwidth=bitwidth, group_size=int(granularity.split("-", 1)[1]))
    if granularity == "block-wise":
        return _quantize_blockwise(weight, bitwidth=bitwidth, block_rows=BLOCK_ROWS, block_cols=BLOCK_COLS)
    raise ValueError(f"Unsupported granularity: {granularity}")



def _quantize_per_tensor(weight: torch.Tensor, *, bitwidth: int) -> torch.Tensor:
    qmin, qmax = _qrange(bitwidth)
    scale = weight.abs().amax().clamp(min=1e-8) / float(qmax)
    return torch.clamp(torch.round(weight / scale), qmin, qmax) * scale



def _quantize_groupwise(weight: torch.Tensor, *, bitwidth: int, group_size: int) -> torch.Tensor:
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}.")
    qmin, qmax = _qrange(bitwidth)
    output_dim, input_dim = weight.shape
    pad_cols = _padding(input_dim, group_size)
    padded = F.pad(weight, (0, pad_cols)) if pad_cols else weight
    groups = padded.reshape(output_dim, -1, group_size)
    scale = groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / float(qmax)
    quantized = torch.clamp(torch.round(groups / scale), qmin, qmax) * scale
    return quantized.reshape(output_dim, input_dim + pad_cols)[:, :input_dim]



def _quantize_blockwise(weight: torch.Tensor, *, bitwidth: int, block_rows: int, block_cols: int) -> torch.Tensor:
    qmin, qmax = _qrange(bitwidth)
    output_dim, input_dim = weight.shape
    pad_rows = _padding(output_dim, block_rows)
    pad_cols = _padding(input_dim, block_cols)
    padded = F.pad(weight, (0, pad_cols, 0, pad_rows)) if pad_rows or pad_cols else weight
    out_blocks = (output_dim + pad_rows) // block_rows
    in_blocks = (input_dim + pad_cols) // block_cols
    blocks = padded.reshape(out_blocks, block_rows, in_blocks, block_cols).permute(0, 2, 1, 3)
    scale = blocks.abs().amax(dim=(-1, -2), keepdim=True).clamp(min=1e-8) / float(qmax)
    quantized = torch.clamp(torch.round(blocks / scale), qmin, qmax) * scale
    restored = quantized.permute(0, 2, 1, 3).reshape(output_dim + pad_rows, input_dim + pad_cols)
    return restored[:output_dim, :input_dim]



def _padding(size: int, block: int) -> int:
    return (block - size % block) % block



def scale_count(granularity: str, *, input_dim: int, output_dim: int) -> int:
    """Count scale values including padding for the specified weight shape and granularity."""
    if granularity == "per-tensor":
        return 1
    if granularity == "per-channel":
        return output_dim
    if granularity.startswith("group-"):
        group_size = int(granularity.split("-", 1)[1])
        return output_dim * math.ceil(input_dim / group_size)
    if granularity == "block-wise":
        return math.ceil(output_dim / BLOCK_ROWS) * math.ceil(input_dim / BLOCK_COLS)
    raise ValueError(f"Unsupported granularity: {granularity}")



def scale_metadata_bits(granularity: str, *, input_dim: int, output_dim: int) -> float:
    """Return the FP32 scale-metadata cost for the supplied weight shape."""
    return 32.0 * float(scale_count(granularity, input_dim=input_dim, output_dim=output_dim)) / float(input_dim * output_dim)



def analytic_latency_proxy(granularity: str, *, input_dim: int, output_dim: int) -> float:
    """Return an analytic fallback latency proxy for a candidate shape and granularity."""
    group128_scales = scale_count("group-128", input_dim=input_dim, output_dim=output_dim)
    density = scale_count(granularity, input_dim=input_dim, output_dim=output_dim) / max(float(group128_scales), 1.0)
    base = {
        "per-tensor": 0.95,
        "per-channel": 0.98,
        "group-32": 1.08,
        "group-64": 1.04,
        "group-128": 1.0,
        "block-wise": 1.12,
    }[granularity]
    return float(base * (1.0 + 0.02 * max(math.log2(max(density, 1.0)), 0.0)))



def candidate_cost(
    granularity: str,
    *,
    input_dim: int,
    output_dim: int,
    bitwidth: int,
    latency_proxy: float,
    latency_source: str,
) -> dict[str, Any]:
    """Return scale metadata, effective weight bits, and the supplied latency estimate."""
    metadata_bits = scale_metadata_bits(granularity, input_dim=input_dim, output_dim=output_dim)
    scales = scale_count(granularity, input_dim=input_dim, output_dim=output_dim)
    current_deploy_supported = granularity in DEPLOY_SUPPORTED_GRANULARITIES
    return {
        "weight_bits_effective": float(bitwidth) + metadata_bits,
        "scale_metadata_bits": metadata_bits,
        "scale_metadata_bytes": 4 * scales,
        "scale_count": scales,
        "latency_proxy": float(latency_proxy),
        "latency_source": latency_source,
        "fake_eval_supported": True,
        "current_deploy_supported": current_deploy_supported,
        "kernel_penalty": 0.0 if current_deploy_supported else 1.0,
    }



def build_latency_table(
    layers: OrderedDict[str, nn.Linear],
    *,
    bitwidth: int,
    device: torch.device,
    dtype: torch.dtype,
    skip_benchmark: bool,
    latency_tokens: int,
    warmup: int,
    iters: int,
    seed: int,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Benchmark each unique layer shape and granularity, returning measured latency proxies."""
    shapes = sorted({(module.out_features, module.in_features) for module in layers.values()})
    table: dict[str, dict[str, dict[str, Any]]] = {}
    for output_dim, input_dim in tqdm(shapes, desc="[cost] latency proxy", unit="shape"):
        shape_key = _shape_key(output_dim, input_dim)
        table[shape_key] = {}
        if skip_benchmark or device.type != "cuda":
            for granularity in GRANULARITIES:
                table[shape_key][granularity] = {
                    "proxy": analytic_latency_proxy(granularity, input_dim=input_dim, output_dim=output_dim),
                    "median_ms": None,
                    "source": "analytic_fallback",
                }
            continue

        medians: dict[str, float] = {}
        for granularity in GRANULARITIES:
            medians[granularity] = _benchmark_synthetic_quantized_linear(
                granularity,
                bitwidth=bitwidth,
                output_dim=output_dim,
                input_dim=input_dim,
                tokens=latency_tokens,
                warmup=warmup,
                iters=iters,
                seed=seed,
                device=device,
                dtype=dtype,
            )
        reference = max(medians["group-128"], 1e-9)
        for granularity in GRANULARITIES:
            table[shape_key][granularity] = {
                "proxy": float(medians[granularity] / reference),
                "median_ms": medians[granularity],
                "source": "synthetic_dequant_cuda",
                "warmup": warmup,
                "iters": iters,
                "tokens": latency_tokens,
            }
    return table



def _benchmark_synthetic_quantized_linear(
    granularity: str,
    *,
    bitwidth: int,
    output_dim: int,
    input_dim: int,
    tokens: int,
    warmup: int,
    iters: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + stable_int(f"{granularity}:{output_dim}:{input_dim}"))
    weight = torch.randn(output_dim, input_dim, generator=generator, dtype=torch.float32)
    x = torch.randn(tokens, input_dim, generator=generator, dtype=torch.float32).to(device=device, dtype=dtype)
    packed = pack_weight(weight.to(device), bitwidth=bitwidth, granularity=granularity)

    with torch.no_grad():
        for _ in range(max(warmup, 0)):
            F.linear(x, dequantize_packed_weight(packed, dtype=dtype))
        torch.cuda.synchronize(device)
        times = []
        for _ in range(max(iters, 1)):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            F.linear(x, dequantize_packed_weight(packed, dtype=dtype))
            end.record()
            torch.cuda.synchronize(device)
            times.append(float(start.elapsed_time(end)))
    times.sort()
    return times[len(times) // 2]



def stable_int(text: str) -> int:
    """Map a string to the deterministic integer used for latency-input generation."""
    import hashlib

    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % (2**31)



def pack_weight(weight: torch.Tensor, *, bitwidth: int, granularity: str) -> PackedWeight:
    """Convert a dense weight into integer codes and scales for the latency proxy."""
    weight = weight.float()
    if granularity == "per-tensor":
        return _pack_per_tensor(weight, bitwidth=bitwidth, granularity=granularity)
    if granularity == "per-channel":
        return _pack_groupwise(weight, bitwidth=bitwidth, granularity=granularity, group_size=weight.shape[1])
    if granularity.startswith("group-"):
        return _pack_groupwise(
            weight,
            bitwidth=bitwidth,
            granularity=granularity,
            group_size=int(granularity.split("-", 1)[1]),
        )
    if granularity == "block-wise":
        return _pack_blockwise(weight, bitwidth=bitwidth, granularity=granularity)
    raise ValueError(f"Unsupported granularity: {granularity}")



def _pack_per_tensor(weight: torch.Tensor, *, bitwidth: int, granularity: str) -> PackedWeight:
    qmin, qmax = _qrange(bitwidth)
    scale = weight.abs().amax().clamp(min=1e-8) / float(qmax)
    qweight = torch.clamp(torch.round(weight / scale), qmin, qmax).to(torch.int8)
    return PackedWeight(
        granularity=granularity,
        qweight=qweight,
        scales=scale.reshape(1),
        input_dim=weight.shape[1],
        output_dim=weight.shape[0],
        padded_input_dim=weight.shape[1],
        padded_output_dim=weight.shape[0],
    )



def _pack_groupwise(weight: torch.Tensor, *, bitwidth: int, granularity: str, group_size: int) -> PackedWeight:
    qmin, qmax = _qrange(bitwidth)
    output_dim, input_dim = weight.shape
    pad_cols = _padding(input_dim, group_size)
    padded = F.pad(weight, (0, pad_cols)) if pad_cols else weight
    groups = padded.reshape(output_dim, -1, group_size)
    scale = groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / float(qmax)
    qweight = torch.clamp(torch.round(groups / scale), qmin, qmax).to(torch.int8)
    return PackedWeight(
        granularity=granularity,
        qweight=qweight,
        scales=scale,
        input_dim=input_dim,
        output_dim=output_dim,
        padded_input_dim=input_dim + pad_cols,
        padded_output_dim=output_dim,
        group_size=group_size,
    )



def _pack_blockwise(weight: torch.Tensor, *, bitwidth: int, granularity: str) -> PackedWeight:
    qmin, qmax = _qrange(bitwidth)
    output_dim, input_dim = weight.shape
    pad_rows = _padding(output_dim, BLOCK_ROWS)
    pad_cols = _padding(input_dim, BLOCK_COLS)
    padded = F.pad(weight, (0, pad_cols, 0, pad_rows)) if pad_rows or pad_cols else weight
    padded_output_dim, padded_input_dim = padded.shape
    blocks = padded.reshape(-1, BLOCK_ROWS, padded_input_dim // BLOCK_COLS, BLOCK_COLS).permute(0, 2, 1, 3)
    scale = blocks.abs().amax(dim=(-1, -2), keepdim=True).clamp(min=1e-8) / float(qmax)
    qweight = torch.clamp(torch.round(blocks / scale), qmin, qmax).to(torch.int8)
    return PackedWeight(
        granularity=granularity,
        qweight=qweight,
        scales=scale,
        input_dim=input_dim,
        output_dim=output_dim,
        padded_input_dim=padded_input_dim,
        padded_output_dim=padded_output_dim,
    )



def dequantize_packed_weight(packed: PackedWeight, *, dtype: torch.dtype) -> torch.Tensor:
    """Reconstruct a dense weight from integer codes and scales at the requested dtype."""
    if packed.granularity == "per-tensor":
        return (packed.qweight.float() * packed.scales.reshape(1, 1)).to(dtype=dtype)
    if packed.granularity == "per-channel" or packed.granularity.startswith("group-"):
        dequantized = packed.qweight.float() * packed.scales
        return dequantized.reshape(packed.output_dim, packed.padded_input_dim)[:, : packed.input_dim].to(dtype=dtype)
    if packed.granularity == "block-wise":
        dequantized = packed.qweight.float() * packed.scales
        restored = dequantized.permute(0, 2, 1, 3).reshape(packed.padded_output_dim, packed.padded_input_dim)
        return restored[: packed.output_dim, : packed.input_dim].to(dtype=dtype)
    raise ValueError(f"Unsupported granularity: {packed.granularity}")



def quantize_layer(
    module: nn.Linear,
    *,
    granularity: str,
    bitwidth: int,
    cost: dict[str, Any],
    device: torch.device,
) -> tuple[GranularityQuantizedLayer, torch.Tensor]:
    """Construct the requested layer candidate and its quantized weight representation."""
    weight = module.weight.detach().float().to(device)
    quantized_weight = quantize_weight_by_granularity(weight, bitwidth=bitwidth, granularity=granularity)
    bias = None if module.bias is None else module.bias.detach().cpu()
    layer = GranularityQuantizedLayer(
        granularity=granularity,
        quantized_weight=quantized_weight.detach().to(device="cpu", dtype=module.weight.dtype),
        bias=bias,
        cost=cost,
    )
    return layer, quantized_weight



def price_granularities(
    *,
    layers: OrderedDict[str, nn.Linear],
    layer_inputs: dict[str, torch.Tensor],
    stats_by_layer: dict[str, Any],
    bitwidth: int,
    latency_table: dict[str, dict[str, dict[str, Any]]],
    device: torch.device,
) -> tuple[dict[str, str], dict[str, dict[str, Any]], dict[str, GranularityQuantizedLayer], dict[str, dict[str, GranularityQuantizedLayer]]]:
    """Return price-guided granularity choices, candidate scores, weights, and candidate caches."""
    selected: dict[str, str] = {}
    prices: dict[str, dict[str, Any]] = {}
    selected_quantized: dict[str, GranularityQuantizedLayer] = {}
    quantized_cache: dict[str, dict[str, GranularityQuantizedLayer]] = {}

    for layer_name, module in tqdm(layers.items(), desc="[price] granularities", unit="layer"):
        x = layer_inputs[layer_name].to(device)
        weight = module.weight.detach().float().to(device)
        bias = None if module.bias is None else module.bias.detach().float().to(device)
        y_ref = F.linear(x, weight, bias)
        hessian_trace = _hessian_trace(stats_by_layer.get(layer_name, {}))
        input_dim = int(module.in_features)
        output_dim = int(module.out_features)
        shape_entry = latency_table[_shape_key(output_dim, input_dim)]

        layer_candidates: dict[str, Any] = {}
        quantized_cache[layer_name] = {}
        for granularity in GRANULARITIES:
            latency_info = shape_entry[granularity]
            cost = candidate_cost(
                granularity,
                input_dim=input_dim,
                output_dim=output_dim,
                bitwidth=bitwidth,
                latency_proxy=float(latency_info["proxy"]),
                latency_source=str(latency_info["source"]),
            )
            quantized, quantized_weight = quantize_layer(
                module,
                granularity=granularity,
                bitwidth=bitwidth,
                cost=cost,
                device=device,
            )
            quantized_cache[layer_name][granularity] = quantized
            y_hat = F.linear(x, quantized_weight.float(), bias)
            mse = float((y_hat - y_ref).float().pow(2).mean().detach().cpu().item())
            rho_hat = 0.5 * hessian_trace * mse
            layer_candidates[granularity] = {
                "rho_hat": rho_hat,
                "local_output_mse": mse,
                "hessian_trace": hessian_trace,
                "cost": cost,
            }

        best_granularity = select_best_granularity(layer_candidates)
        selected[layer_name] = best_granularity
        selected_quantized[layer_name] = quantized_cache[layer_name][best_granularity]
        prices[layer_name] = {
            "selected": best_granularity,
            "selection_rule": "min rho_hat with 1% effective-bit/latency tie-break",
            "price_tie_tolerance": PRICE_TIE_TOLERANCE,
            "candidates": layer_candidates,
        }
    return selected, prices, selected_quantized, quantized_cache



def select_best_granularity(layer_candidates: dict[str, Any]) -> str:
    """Choose within one percent of minimum price by effective bits, latency, and fixed order."""
    min_rho = min(float(item["rho_hat"]) for item in layer_candidates.values())
    threshold = min_rho + max(abs(min_rho) * PRICE_TIE_TOLERANCE, 1e-12)
    eligible = [
        granularity
        for granularity, payload in layer_candidates.items()
        if float(payload["rho_hat"]) <= threshold
    ]
    order = {granularity: index for index, granularity in enumerate(("group-128", "group-64", "per-channel", "group-32", "per-tensor", "block-wise"))}
    return min(
        eligible,
        key=lambda granularity: (
            float(layer_candidates[granularity]["cost"]["weight_bits_effective"]),
            float(layer_candidates[granularity]["cost"]["latency_proxy"]),
            order[granularity],
        ),
    )



def random_quantized_selection(
    *,
    layer_names: list[str],
    quantized_cache: dict[str, dict[str, GranularityQuantizedLayer]],
    seed: int,
) -> tuple[dict[str, str], dict[str, GranularityQuantizedLayer]]:
    """Sample one cached candidate per layer with the supplied seed and return choices and weights."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    selected: dict[str, str] = {}
    quantized: dict[str, GranularityQuantizedLayer] = {}
    for layer_name in tqdm(layer_names, desc=f"[random] select seed={seed}", unit="layer"):
        granularity = GRANULARITIES[int(torch.randint(0, len(GRANULARITIES), (1,), generator=generator).item())]
        selected[layer_name] = granularity
        quantized[layer_name] = quantized_cache[layer_name][granularity]
    return selected, quantized



def fixed_granularity_selection(
    *,
    layer_names: list[str],
    quantized_cache: dict[str, dict[str, GranularityQuantizedLayer]],
    granularity: str,
) -> tuple[dict[str, str], dict[str, GranularityQuantizedLayer]]:
    """Return fixed-granularity assignments and weights from the supplied candidate cache."""
    return (
        {layer_name: granularity for layer_name in layer_names},
        {layer_name: quantized_cache[layer_name][granularity] for layer_name in layer_names},
    )



def selection_summary(selected: dict[str, str], prices: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Summarize selected candidate costs and predicted prices over all layers."""
    counts = Counter(selected.values())
    selected_costs = [
        prices[layer_name]["candidates"][granularity]["cost"]
        for layer_name, granularity in selected.items()
    ]
    total_layers = max(len(selected_costs), 1)
    return {
        "granularity_counts": dict(sorted(counts.items())),
        "mean_effective_bits": _mean([float(cost["weight_bits_effective"]) for cost in selected_costs]),
        "total_scale_metadata_bytes": int(sum(int(cost["scale_metadata_bytes"]) for cost in selected_costs)),
        "mean_scale_metadata_bits": _mean([float(cost["scale_metadata_bits"]) for cost in selected_costs]),
        "mean_latency_proxy": _mean([float(cost["latency_proxy"]) for cost in selected_costs]),
        "current_deploy_supported_layers": int(sum(bool(cost["current_deploy_supported"]) for cost in selected_costs)),
        "current_deploy_supported_ratio": float(sum(bool(cost["current_deploy_supported"]) for cost in selected_costs) / total_layers),
    }



def _get_parent_module(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent: nn.Module = model
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]



def write_fake_quant_weights_to_model(
    model: nn.Module,
    quantized_layers: dict[str, GranularityQuantizedLayer],
) -> None:
    """Write selected fake-quant weights into ordinary linear modules for HF export."""
    with torch.no_grad():
        for layer_name, quantized in tqdm(quantized_layers.items(), desc="[save] write fake-quant weights", unit="layer"):
            parent, attr = _get_parent_module(model, layer_name)
            module = getattr(parent, attr)
            if not isinstance(module, nn.Linear):
                raise TypeError(f"Cannot save fake-quant weight for non-Linear module {layer_name}: {type(module)!r}")
            module.weight.copy_(quantized.quantized_weight.to(device=module.weight.device, dtype=module.weight.dtype))
            if module.bias is not None and quantized.bias is not None:
                module.bias.copy_(quantized.bias.to(device=module.bias.device, dtype=module.bias.dtype))



@contextmanager
def quantized_layer_context(model: nn.Module, quantized_layers: dict[str, GranularityQuantizedLayer]):
    """Temporarily replace selected linear modules with their fake-quant candidate modules."""
    originals = {}
    try:
        for layer_name, quantized in quantized_layers.items():
            parent, attr = _get_parent_module(model, layer_name)
            original = getattr(parent, attr)
            originals[layer_name] = (parent, attr, original)
            device = original.weight.device if isinstance(original, nn.Linear) else quantized.quantized_weight.device
            setattr(parent, attr, GranularityQuantLinear(quantized, device=device))
        yield
    finally:
        for parent, attr, module in originals.values():
            setattr(parent, attr, module)



def _shape_key(output_dim: int, input_dim: int) -> str:
    return f"{output_dim}x{input_dim}"

