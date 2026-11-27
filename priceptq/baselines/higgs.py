"""Paper-specific HIGGS-style sensitivity and budgeted allocation baseline."""

from __future__ import annotations

import argparse

import json

import math

from collections import Counter

from pathlib import Path

from typing import Any

import numpy as np

from scipy.optimize import Bounds, LinearConstraint, milp

from scipy.sparse import lil_matrix

from priceptq.configuration import get_amq_model_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]



MODEL_KEY_TO_AMQ_NAME = {
    "llama-1b": "Llama-3.2-1B",
    "llama-3b": "Llama-3.2-3B",
    "llama-8b": "Llama-3.1-8B",
}



def _default_assets_dir(model_key: str, dataset_key: str) -> Path:
    return PROJECT_ROOT / "results" / "higgs" / f"{model_key}_{dataset_key}" / "assets"



def _default_output_dir(model_key: str, dataset_key: str) -> Path:
    return PROJECT_ROOT / "results" / "higgs" / f"{model_key}_{dataset_key}" / "bit_allocation"



def _torch_load(path: Path) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")



def _torch_save(obj: Any, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)



def _json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2)



def _load_amq_config(config_path: str | Path, model_name: str) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        configs = json.load(handle)
    return get_amq_model_config(configs, model_name)



def _amq_layer_keys(config: dict[str, Any]) -> list[str]:
    return [
        f"{block_idx}_{linear_key}"
        for block_idx in range(int(config["n_block"]))
        for linear_key in config["linear"]
    ]



def _split_qdict_key(qdict_key: str) -> tuple[int, str]:
    block_idx, linear_key = qdict_key.split("_", 1)
    return int(block_idx), linear_key



def _raw_layer_name(config: dict[str, Any], qdict_key: str) -> str:
    block_idx, linear_key = _split_qdict_key(qdict_key)
    return f"{config['layers']}.{block_idx}.{linear_key}"



def _get_submodule(obj: Any, dotted_path: str) -> Any:
    current = obj
    for part in dotted_path.split("."):
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current



def _coerce_scalar(value: Any, *, context: str) -> float:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(f"{context} tensor is not scalar: shape={tuple(value.shape)}")
            return float(value.detach().cpu().item())
    except ImportError:
        pass
    if isinstance(value, np.generic):
        return float(value.item())
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"{context} must be a scalar, got {type(value).__name__}.")



def _load_err_coeffs(path: Path) -> dict[str, float]:
    raw = _torch_load(path)
    if isinstance(raw, dict) and "err_coeffs" in raw:
        raw = raw["err_coeffs"]
    if not isinstance(raw, dict):
        raise ValueError(f"err_coeffs asset must be a dict, got {type(raw).__name__}.")
    return {str(key): _coerce_scalar(value, context=f"err_coeffs[{key!r}]") for key, value in raw.items()}



def _load_quant_err(path: Path) -> dict[str, dict[str, float]]:
    raw = _torch_load(path)
    if isinstance(raw, dict) and "quant_err" in raw:
        raw = raw["quant_err"]
    if not isinstance(raw, dict):
        raise ValueError(f"quant_err asset must be a dict, got {type(raw).__name__}.")

    result: dict[str, dict[str, float]] = {}
    for key, value in raw.items():
        key = str(key)
        if isinstance(value, dict):
            if "mean_error" in value:
                mean_error = _coerce_scalar(value["mean_error"], context=f"quant_err[{key!r}]['mean_error']")
            elif "err" in value:
                mean_error = _coerce_scalar(value["err"], context=f"quant_err[{key!r}]['err']")
            else:
                raise ValueError(f"quant_err[{key!r}] must contain mean_error.")
            result[key] = {
                **{str(k): v for k, v in value.items() if isinstance(v, (int, float, str, bool))},
                "mean_error": float(mean_error),
            }
        else:
            result[key] = {"mean_error": _coerce_scalar(value, context=f"quant_err[{key!r}]")}
    return result



def _candidate_keys(backend: str, bits: list[int]) -> list[str]:
    return [f"{backend}_{int(bit)}" for bit in sorted(set(bits))]



def _candidate_key_to_bit(candidate_key: str) -> int:
    try:
        return int(candidate_key.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid AMQ candidate key {candidate_key!r}; expected backend_bit.") from exc



def _validate_layer_coverage(err_coeffs: dict[str, float], config: dict[str, Any]) -> list[str]:
    ordered_keys = _amq_layer_keys(config)
    missing = sorted(set(ordered_keys) - set(err_coeffs))
    if missing:
        preview = ", ".join(missing[:8])
        suffix = "" if len(missing) <= 8 else f", ... ({len(missing)} missing)"
        raise ValueError(f"err_coeffs.pt is missing AMQ layer keys: {preview}{suffix}")
    return ordered_keys



def _validate_quant_err_coverage(quant_err: dict[str, dict[str, float]], candidate_keys: list[str]) -> None:
    missing = [candidate_key for candidate_key in candidate_keys if candidate_key not in quant_err]
    if missing:
        raise ValueError(
            "quant_err.pt is missing AMQ candidate keys "
            f"{missing}. Recompute it with the current backend/bits."
        )
    for candidate_key in candidate_keys:
        mean_error = float(quant_err[candidate_key]["mean_error"])
        if not math.isfinite(mean_error) or mean_error < 0.0:
            raise ValueError(f"quant_err[{candidate_key!r}]['mean_error'] must be finite and non-negative.")



def _amq_memory_units(config: dict[str, Any], linear_key: str, bit: int, group_size: int) -> float:
    out_dim, in_dim = config["linear_shape"][linear_key]
    c_group_size = in_dim if group_size == -1 else group_size
    effective_bit = float(bit) + (32.0 / float(c_group_size) if bit < 16 else 0.0)
    return float(out_dim) * float(in_dim) * effective_bit



def _architecture_from_selection(
    selected_candidates: dict[str, str],
    config: dict[str, Any],
) -> dict[str, dict[str, list[int]]]:
    arch = {"linear": {linear_key: [] for linear_key in config["linear"]}}
    for block_idx in range(int(config["n_block"])):
        for linear_key in config["linear"]:
            candidate_key = selected_candidates[f"{block_idx}_{linear_key}"]
            arch["linear"][linear_key].append(_candidate_key_to_bit(candidate_key))
    return arch



def _bits_usage(arch: dict[str, Any], config: dict[str, Any], group_size: int) -> float:
    memory_usage = 0.0
    for linear_key, bits in arch["linear"].items():
        for bit in bits:
            memory_usage += _amq_memory_units(config, linear_key, int(bit), group_size)
    return memory_usage / float(config["model_numel"])



def _bits_usage_bounds(
    ordered_keys: list[str],
    config: dict[str, Any],
    bits: list[int],
    group_size: int,
) -> tuple[float, float]:
    candidate_bits = sorted(set(int(bit) for bit in bits))
    min_memory = 0.0
    max_memory = 0.0
    for qdict_key in ordered_keys:
        _, linear_key = _split_qdict_key(qdict_key)
        min_memory += _amq_memory_units(config, linear_key, candidate_bits[0], group_size)
        max_memory += _amq_memory_units(config, linear_key, candidate_bits[-1], group_size)
    model_numel = float(config["model_numel"])
    return min_memory / model_numel, max_memory / model_numel



def _validate_target_window(
    *,
    ordered_keys: list[str],
    config: dict[str, Any],
    bits: list[int],
    group_size: int,
    target_bits: float,
    target_bits_offset: float,
    owq_selection_offset: float,
) -> tuple[float, float, float, float]:
    min_usage, max_usage = _bits_usage_bounds(ordered_keys, config, bits, group_size)
    raw_lower = target_bits - target_bits_offset - owq_selection_offset
    raw_upper = target_bits + target_bits_offset - owq_selection_offset
    if raw_lower > max_usage or raw_upper < min_usage:
        raise ValueError(
            "Requested AMQ target window is outside the reachable bits_usage range. "
            f"target_window_after_owq_adjustment=({raw_lower:.12f}, {raw_upper:.12f}), "
            f"reachable_raw_range=[{min_usage:.12f}, {max_usage:.12f}], "
            f"candidate_bits={sorted(set(int(bit) for bit in bits))}, group_size={group_size}."
        )
    return min_usage, max_usage, raw_lower, raw_upper



def _format_target(value: float) -> str:
    formatted = f"{value:.6f}".rstrip("0").rstrip(".")
    return formatted.replace(".", "p")



def _solve_milp(
    *,
    ordered_keys: list[str],
    err_coeffs: dict[str, float],
    quant_err: dict[str, dict[str, float]],
    candidate_keys: list[str],
    config: dict[str, Any],
    group_size: int,
    raw_lower_bits_usage: float,
    raw_upper_bits_usage: float,
    time_limit: float,
) -> tuple[dict[str, str], Any]:
    n_layers = len(ordered_keys)
    n_candidates = len(candidate_keys)
    n_vars = n_layers * n_candidates

    c = np.zeros(n_vars, dtype=np.float64)
    memory = np.zeros(n_vars, dtype=np.float64)
    var_index: dict[tuple[str, str], int] = {}

    for layer_idx, qdict_key in enumerate(ordered_keys):
        _, linear_key = _split_qdict_key(qdict_key)
        err_coeff = float(err_coeffs[qdict_key])
        for candidate_idx, candidate_key in enumerate(candidate_keys):
            idx = layer_idx * n_candidates + candidate_idx
            bit = _candidate_key_to_bit(candidate_key)
            var_index[(qdict_key, candidate_key)] = idx
            c[idx] = err_coeff * float(quant_err[candidate_key]["mean_error"])
            memory[idx] = _amq_memory_units(config, linear_key, bit, group_size)

    constraints = []
    one_hot = lil_matrix((n_layers, n_vars), dtype=np.float64)
    for layer_idx in range(n_layers):
        for candidate_idx in range(n_candidates):
            one_hot[layer_idx, layer_idx * n_candidates + candidate_idx] = 1.0
    constraints.append(LinearConstraint(one_hot.tocsr(), np.ones(n_layers), np.ones(n_layers)))

    budget = lil_matrix((1, n_vars), dtype=np.float64)
    budget[0, :] = memory
    model_numel = float(config["model_numel"])
    constraints.append(
        LinearConstraint(
            budget.tocsr(),
            np.array([raw_lower_bits_usage * model_numel], dtype=np.float64),
            np.array([raw_upper_bits_usage * model_numel], dtype=np.float64),
        )
    )

    result = milp(
        c=c,
        integrality=np.ones(n_vars, dtype=np.int8),
        bounds=Bounds(np.zeros(n_vars), np.ones(n_vars)),
        constraints=constraints,
        options={"time_limit": time_limit, "mip_rel_gap": 0.0},
    )
    if not result.success:
        raise RuntimeError(f"HIGGS MCKP MILP failed: status={result.status}, message={result.message}")

    selected: dict[str, str] = {}
    for qdict_key in ordered_keys:
        chosen = [
            candidate_key
            for candidate_key in candidate_keys
            if result.x[var_index[(qdict_key, candidate_key)]] > 0.5
        ]
        if len(chosen) != 1:
            raise RuntimeError(f"MILP returned invalid assignment for {qdict_key}: {chosen}")
        selected[qdict_key] = chosen[0]
    return selected, result



def _select_device(requested: str | None):
    import torch

    if requested is not None:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")



def _load_calibration_inputs(
    *,
    model_path: str,
    dataset_name: str,
    dataset_config: str | None,
    dataset_split: str,
    num_samples: int,
    max_seqlen: int,
    seed: int,
    local_files_only: bool,
    trust_remote_code: bool,
) -> dict[str, Any]:
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset_kwargs: dict[str, Any] = {"split": dataset_split}
    if local_files_only:
        dataset_kwargs["download_mode"] = "reuse_dataset_if_exists"
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, **dataset_kwargs)
    else:
        dataset = load_dataset(dataset_name, **dataset_kwargs)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(dataset)).tolist()
    texts = []
    for idx in indices:
        item = dataset[int(idx)]
        text = item.get("text") if isinstance(item, dict) else None
        if isinstance(text, str) and text.strip():
            texts.append(text)
        if len(texts) >= max(64, num_samples * 2):
            break
    if not texts:
        raise ValueError(f"No non-empty text examples found in {dataset_name}/{dataset_config}:{dataset_split}.")

    joined = "\n\n".join(texts)
    tokenized = tokenizer(joined, return_tensors="pt", add_special_tokens=False)
    token_ids = tokenized["input_ids"].flatten()
    required = num_samples * max_seqlen
    while token_ids.numel() < required:
        token_ids = torch.cat([token_ids, token_ids], dim=0)
    token_ids = token_ids[:required].reshape(num_samples, max_seqlen)
    attention_mask = torch.ones_like(token_ids)
    return {
        "tokenizer": tokenizer,
        "input_ids": token_ids,
        "attention_mask": attention_mask,
    }



def _evaluate_loss(
    model: Any,
    input_ids,
    attention_mask,
    *,
    batch_size: int,
    device,
) -> float:
    import torch

    losses: list[float] = []
    with torch.no_grad():
        for start in range(0, input_ids.shape[0], batch_size):
            end = min(start + batch_size, input_ids.shape[0])
            batch_input = input_ids[start:end].to(device)
            batch_mask = attention_mask[start:end].to(device)
            outputs = model(input_ids=batch_input, attention_mask=batch_mask, labels=batch_input)
            losses.append(float(outputs.loss.detach().float().cpu().item()))
    return float(np.mean(losses))



def _masked_kl(clean_logits, noisy_logits, attention_mask) -> tuple[float, int]:
    import torch
    import torch.nn.functional as F

    mask = attention_mask.to(dtype=torch.bool, device=noisy_logits.device).reshape(-1)
    clean_flat = clean_logits.to(device=noisy_logits.device, dtype=torch.float32).reshape(-1, clean_logits.shape[-1])
    noisy_flat = noisy_logits.to(dtype=torch.float32).reshape(-1, noisy_logits.shape[-1])
    if mask.numel() != clean_flat.shape[0]:
        raise ValueError("attention_mask shape does not match logits shape for KL calibration.")
    clean_flat = clean_flat[mask]
    noisy_flat = noisy_flat[mask]
    clean_log_p = clean_flat.log_softmax(dim=-1)
    noisy_log_q = noisy_flat.log_softmax(dim=-1)
    kl = F.kl_div(noisy_log_q, clean_log_p, reduction="sum", log_target=True)
    return float(kl.detach().cpu().item()), int(mask.sum().item())



def _cache_clean_logits(
    model: Any,
    input_ids,
    attention_mask,
    *,
    batch_size: int,
    device,
) -> list[Any]:
    import torch

    cached = []
    with torch.no_grad():
        for start in range(0, input_ids.shape[0], batch_size):
            end = min(start + batch_size, input_ids.shape[0])
            outputs = model(
                input_ids=input_ids[start:end].to(device),
                attention_mask=attention_mask[start:end].to(device),
            )
            cached.append(outputs.logits.detach().float().cpu())
    return cached



def _evaluate_kl(
    model: Any,
    input_ids,
    attention_mask,
    clean_logits: list[Any],
    *,
    batch_size: int,
    device,
) -> float:
    import torch

    total_kl = 0.0
    total_tokens = 0
    batch_id = 0
    with torch.no_grad():
        for start in range(0, input_ids.shape[0], batch_size):
            end = min(start + batch_size, input_ids.shape[0])
            outputs = model(
                input_ids=input_ids[start:end].to(device),
                attention_mask=attention_mask[start:end].to(device),
            )
            kl, n_tokens = _masked_kl(clean_logits[batch_id], outputs.logits, attention_mask[start:end])
            total_kl += kl
            total_tokens += n_tokens
            batch_id += 1
    return total_kl / max(1, total_tokens)



def _fit_sensitivity(xs: list[float], ys: list[float]) -> dict[str, float]:
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    denom = float(np.dot(x, x))
    slope_origin = 0.0 if denom == 0.0 else float(np.dot(x, y) / denom)
    if len(x) >= 2:
        slope, intercept = np.polyfit(x, y, deg=1).tolist()
    else:
        slope, intercept = slope_origin, 0.0
    return {
        "slope_origin": float(slope_origin),
        "slope_with_intercept": float(slope),
        "intercept": float(intercept),
        "used_coefficient": float(max(slope_origin, 0.0)),
    }



def precompute_err_coeffs(
    *,
    err_coeffs_path: Path,
    summary_path: Path,
    config: dict[str, Any],
    model_path: str,
    dataset_name: str,
    dataset_config: str | None,
    dataset_split: str,
    num_samples: int,
    max_seqlen: int,
    batch_size: int,
    seed: int,
    noise_levels: int,
    max_relative_norm: float,
    metric: str,
    device_name: str | None,
    local_files_only: bool,
    trust_remote_code: bool,
) -> dict[str, float]:
    """Fit layer sensitivity coefficients from controlled noise and save their calibration records."""
    import torch
    from transformers import AutoModelForCausalLM

    if model_path is None:
        raise ValueError("model_path is required to compute missing err_coeffs.pt.")
    if noise_levels <= 0:
        raise ValueError("sensitivity-noise-levels must be positive.")
    if max_relative_norm <= 0.0:
        raise ValueError("sensitivity-max-relative-norm must be positive.")

    device = _select_device(device_name)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"[higgs] Computing err_coeffs.pt on {device} from {model_path}.")
    calibration = _load_calibration_inputs(
        model_path=model_path,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        dataset_split=dataset_split,
        num_samples=num_samples,
        max_seqlen=max_seqlen,
        seed=seed,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    input_ids = calibration["input_ids"]
    attention_mask = calibration["attention_mask"]

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    model.config.use_cache = False
    model.eval()
    model.to(device)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    ordered_keys = _amq_layer_keys(config)
    noise_norms = np.linspace(
        max_relative_norm / float(noise_levels),
        max_relative_norm,
        noise_levels,
        dtype=np.float64,
    ).tolist()

    if metric == "loss_increase":
        baseline = _evaluate_loss(model, input_ids, attention_mask, batch_size=batch_size, device=device)
        clean_logits = None
    elif metric == "kl":
        baseline = 0.0
        clean_logits = _cache_clean_logits(model, input_ids, attention_mask, batch_size=batch_size, device=device)
    else:
        raise ValueError(f"Unsupported sensitivity metric: {metric}")

    err_coeffs: dict[str, float] = {}
    layer_summary: dict[str, Any] = {}
    for layer_index, qdict_key in enumerate(ordered_keys, start=1):
        raw_layer_name = _raw_layer_name(config, qdict_key)
        module = _get_submodule(model, raw_layer_name)
        if not hasattr(module, "weight"):
            raise ValueError(f"Expected {raw_layer_name} to have a weight tensor.")
        weight = module.weight
        weight_norm = float(weight.detach().float().norm().cpu().item())
        xs: list[float] = []
        ys: list[float] = []
        print(f"[higgs] Calibrating {layer_index}/{len(ordered_keys)}: {qdict_key}")
        for relative_norm in noise_norms:
            if weight_norm == 0.0:
                xs.append(0.0)
                ys.append(0.0)
                continue
            noise = torch.randn(weight.shape, generator=generator, device=device, dtype=torch.float32)
            noise_norm = noise.norm()
            noise.mul_(float(relative_norm) * weight_norm / (noise_norm + 1e-12))
            injected = noise.to(dtype=weight.dtype)
            actual_x = float((injected.float().norm().item() / weight_norm) ** 2)
            with torch.no_grad():
                weight.add_(injected)
            try:
                if metric == "loss_increase":
                    noisy_value = _evaluate_loss(
                        model,
                        input_ids,
                        attention_mask,
                        batch_size=batch_size,
                        device=device,
                    )
                    delta = noisy_value - baseline
                else:
                    assert clean_logits is not None
                    delta = _evaluate_kl(
                        model,
                        input_ids,
                        attention_mask,
                        clean_logits,
                        batch_size=batch_size,
                        device=device,
                    )
            finally:
                with torch.no_grad():
                    weight.sub_(injected)
            xs.append(actual_x)
            ys.append(float(delta))
        fit = _fit_sensitivity(xs, ys)
        err_coeffs[qdict_key] = fit["used_coefficient"]
        layer_summary[qdict_key] = {
            "raw_layer_name": raw_layer_name,
            "weight_norm": weight_norm,
            "relative_noise_norms": noise_norms,
            "x_relative_noise_norm_sq": xs,
            "delta_metric": ys,
            **fit,
        }

    _torch_save(err_coeffs, err_coeffs_path)
    summary = {
        "artifact": str(err_coeffs_path),
        "method": "HIGGS sensitivity calibration with Gaussian weight noise",
        "metric": metric,
        "baseline_metric": baseline,
        "model_path": model_path,
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "dataset_split": dataset_split,
        "num_samples": num_samples,
        "max_seqlen": max_seqlen,
        "batch_size": batch_size,
        "seed": seed,
        "noise_levels": noise_levels,
        "max_relative_norm": max_relative_norm,
        "coefficients": layer_summary,
    }
    _json_dump(summary, summary_path)
    return err_coeffs



def _fake_quant_groupwise_asym(w, bit: int, group_size: int):
    import torch

    org_shape = w.shape
    if group_size > 0:
        if org_shape[-1] % group_size != 0:
            raise ValueError(f"Last dim {org_shape[-1]} is not divisible by group_size={group_size}.")
        w_2d = w.reshape(-1, group_size)
    elif group_size == -1:
        w_2d = w.reshape(-1, org_shape[-1])
    else:
        raise ValueError("group_size must be -1 or a positive integer.")
    max_val = w_2d.amax(dim=1, keepdim=True)
    min_val = w_2d.amin(dim=1, keepdim=True)
    max_int = 2**bit - 1
    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    zeros = (-torch.round(min_val / scales)).clamp_(0, max_int)
    q = (torch.clamp(torch.round(w_2d / scales) + zeros, 0, max_int) - zeros) * scales
    return q.reshape(org_shape)



def _fake_quantize_for_backend(w, *, backend: str, bit: int, group_size: int):
    if backend in {"gptq", "awq", "owq"}:
        return _fake_quant_groupwise_asym(w, bit=bit, group_size=group_size)
    raise ValueError(f"Unsupported backend: {backend}")



def precompute_quant_err(
    *,
    quant_err_path: Path,
    summary_path: Path,
    backend: str,
    bits: list[int],
    group_size: int,
    shape: list[int],
    trials: int,
    seed: int,
    device_name: str | None,
) -> dict[str, dict[str, float]]:
    """Estimate candidate quantization errors on seeded Gaussian matrices and save the asset table."""
    import torch

    if len(shape) != 2:
        raise ValueError("quant-error-shape must contain two integers.")
    if trials <= 0:
        raise ValueError("quant-error-trials must be positive.")
    rows, cols = int(shape[0]), int(shape[1])
    if group_size > 0 and cols % group_size != 0:
        raise ValueError(f"quant-error-shape cols={cols} must be divisible by group_size={group_size}.")
    device = _select_device(device_name)
    print(f"[higgs] Computing quant_err.pt on {device} for {backend} candidates {sorted(set(bits))}.")

    quant_err: dict[str, dict[str, float]] = {}
    for bit in sorted(set(int(bit) for bit in bits)):
        errors: list[float] = []
        for trial_idx in range(trials):
            generator = torch.Generator(device=device)
            generator.manual_seed(seed + 1009 * bit + trial_idx)
            w = torch.randn((rows, cols), generator=generator, device=device, dtype=torch.float32)
            q = _fake_quantize_for_backend(w, backend=backend, bit=bit, group_size=group_size)
            mse = (q - w).pow(2).mean()
            denom = w.pow(2).mean().clamp(min=1e-12)
            errors.append(float((mse / denom).detach().cpu().item()))
            del w, q
            if device.type == "cuda":
                torch.cuda.empty_cache()
        candidate_key = f"{backend}_{bit}"
        quant_err[candidate_key] = {
            "mean_error": float(np.mean(errors)),
            "std_error": float(np.std(errors)),
            "trials": int(trials),
            "backend": backend,
            "bit": int(bit),
            "group_size": int(group_size),
            "shape": [rows, cols],
            "fake_quantizer": "AMQ-compatible groupwise asymmetric weight quantization",
        }

    _torch_save(quant_err, quant_err_path)
    summary = {
        "artifact": str(quant_err_path),
        "method": "Gaussian normalized MSE under AMQ-compatible fake quantization",
        "backend": backend,
        "bits": sorted(set(int(bit) for bit in bits)),
        "group_size": int(group_size),
        "shape": [rows, cols],
        "trials": int(trials),
        "seed": int(seed),
        "quant_err": quant_err,
    }
    _json_dump(summary, summary_path)
    return quant_err



def _ensure_assets(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    assets_dir: Path,
) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[str, Any]]:
    assets_dir.mkdir(parents=True, exist_ok=True)
    err_coeffs_path = assets_dir / "err_coeffs.pt"
    quant_err_path = assets_dir / "quant_err.pt"
    err_summary_path = assets_dir / "err_coeffs_summary.json"
    quant_summary_path = assets_dir / "quant_err_summary.json"

    recompute_err = args.force_recompute_assets or args.force_recompute_err_coeffs or not err_coeffs_path.exists()
    recompute_quant = args.force_recompute_assets or args.force_recompute_quant_err or not quant_err_path.exists()

    if recompute_err:
        err_coeffs = precompute_err_coeffs(
            err_coeffs_path=err_coeffs_path,
            summary_path=err_summary_path,
            config=config,
            model_path=args.model_path,
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            dataset_split=args.dataset_split,
            num_samples=args.sensitivity_num_samples,
            max_seqlen=args.sensitivity_max_seqlen,
            batch_size=args.sensitivity_batch_size,
            seed=args.sensitivity_seed,
            noise_levels=args.sensitivity_noise_levels,
            max_relative_norm=args.sensitivity_max_relative_norm,
            metric=args.sensitivity_metric,
            device_name=args.sensitivity_device,
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        err_coeffs = _load_err_coeffs(err_coeffs_path)

    if recompute_quant:
        quant_err = precompute_quant_err(
            quant_err_path=quant_err_path,
            summary_path=quant_summary_path,
            backend=args.backend,
            bits=args.bits,
            group_size=args.group_size,
            shape=args.quant_error_shape,
            trials=args.quant_error_trials,
            seed=args.quant_error_seed,
            device_name=args.quant_error_device,
        )
    else:
        quant_err = _load_quant_err(quant_err_path)

    precompute = {
        "assets_dir": str(assets_dir),
        "err_coeffs_path": str(err_coeffs_path),
        "quant_err_path": str(quant_err_path),
        "err_coeffs_summary_path": str(err_summary_path),
        "quant_err_summary_path": str(quant_summary_path),
        "err_coeffs_recomputed": bool(recompute_err),
        "quant_err_recomputed": bool(recompute_quant),
        "sensitivity": {
            "metric": args.sensitivity_metric,
            "num_samples": args.sensitivity_num_samples,
            "max_seqlen": args.sensitivity_max_seqlen,
            "batch_size": args.sensitivity_batch_size,
            "noise_levels": args.sensitivity_noise_levels,
            "max_relative_norm": args.sensitivity_max_relative_norm,
            "seed": args.sensitivity_seed,
        },
        "quant_error": {
            "shape": list(map(int, args.quant_error_shape)),
            "trials": int(args.quant_error_trials),
            "seed": int(args.quant_error_seed),
            "fake_quantizer": "AMQ-compatible groupwise asymmetric weight quantization",
        },
    }
    return err_coeffs, quant_err, precompute



def solve_higgs_bit_allocation(
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Construct fresh or explicit cached HIGGS assets and solve the shared bit budget."""
    model_name = args.model_name or MODEL_KEY_TO_AMQ_NAME.get(args.model_key, args.model_key)
    config = _load_amq_config(args.amq_config, model_name)
    assets_dir = Path(args.assets_dir) if args.assets_dir else _default_assets_dir(args.model_key, args.dataset_key)
    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(args.model_key, args.dataset_key)

    err_coeffs, quant_err, precompute = _ensure_assets(args=args, config=config, assets_dir=assets_dir)
    ordered_keys = _validate_layer_coverage(err_coeffs, config)
    bits = sorted(set(int(bit) for bit in args.bits))
    candidate_keys = _candidate_keys(args.backend, bits)
    _validate_quant_err_coverage(quant_err, candidate_keys)

    owq_selection_offset = 0.1 if args.backend == "owq" else 0.0
    min_bits_usage, max_bits_usage, raw_lower, raw_upper = _validate_target_window(
        ordered_keys=ordered_keys,
        config=config,
        bits=bits,
        group_size=args.group_size,
        target_bits=args.target_bits,
        target_bits_offset=args.target_bits_offset,
        owq_selection_offset=owq_selection_offset,
    )
    selected_candidates, result = _solve_milp(
        ordered_keys=ordered_keys,
        err_coeffs=err_coeffs,
        quant_err=quant_err,
        candidate_keys=candidate_keys,
        config=config,
        group_size=args.group_size,
        raw_lower_bits_usage=raw_lower,
        raw_upper_bits_usage=raw_upper,
        time_limit=args.time_limit,
    )

    arch = _architecture_from_selection(selected_candidates, config)
    raw_selected_bits_usage = _bits_usage(arch, config, args.group_size)
    selected_bits_usage = raw_selected_bits_usage + owq_selection_offset
    if not (args.target_bits - args.target_bits_offset <= selected_bits_usage <= args.target_bits + args.target_bits_offset):
        raise RuntimeError(
            f"Selected bits_usage={selected_bits_usage} is outside target window "
            f"[{args.target_bits - args.target_bits_offset}, {args.target_bits + args.target_bits_offset}]."
        )

    target_tag = _format_target(args.target_bits)
    offset_tag = _format_target(args.target_bits_offset)
    output_root = output_dir / model_name / "mckp" / args.backend / f"target_{target_tag}_offset_{offset_tag}"
    output_root.mkdir(parents=True, exist_ok=True)
    arch_path = output_root / "amq_arch.json"
    search_stats_path = output_root / "iter_higgs.stats"
    summary_path = output_root / "allocation_summary.json"
    result_path = None if args.result_path is None else Path(args.result_path)

    bit_counts = Counter(_candidate_key_to_bit(candidate_key) for candidate_key in selected_candidates.values())
    candidate_counts = Counter(selected_candidates.values())
    per_layer = []
    for qdict_key in ordered_keys:
        block_idx, linear_key = _split_qdict_key(qdict_key)
        candidate_key = selected_candidates[qdict_key]
        bit = _candidate_key_to_bit(candidate_key)
        mean_error = float(quant_err[candidate_key]["mean_error"])
        err_coeff = float(err_coeffs[qdict_key])
        per_layer.append(
            {
                "qdict_key": qdict_key,
                "block_idx": block_idx,
                "linear_key": linear_key,
                "raw_layer_name": _raw_layer_name(config, qdict_key),
                "selected_candidate": candidate_key,
                "selected_bits": int(bit),
                "err_coefficient": err_coeff,
                "quant_error": mean_error,
                "selected_objective_value": err_coeff * mean_error,
                "amq_memory_units": _amq_memory_units(config, linear_key, bit, args.group_size),
            }
        )

    metadata = {
        "source": "priceptq.baselines.higgs",
        "result_owner": "higgs",
        "solver": "scipy.optimize.milp",
        "solver_message": str(result.message),
        "objective_form": "sum_lq P_lq * err_coeffs[l] * quant_err[q]['mean_error']",
        "model_key": args.model_key,
        "model_name": model_name,
        "model_path": args.model_path,
        "dataset_key": args.dataset_key,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "dataset_split": args.dataset_split,
        "backend": args.backend,
        "candidate_bits": bits,
        "candidate_keys": candidate_keys,
        "amq_config_path": str(Path(args.amq_config)),
        "err_coeffs_path": precompute["err_coeffs_path"],
        "quant_err_path": precompute["quant_err_path"],
        "precompute": precompute,
        "target_bits": float(args.target_bits),
        "target_bits_offset": float(args.target_bits_offset),
        "group_size": int(args.group_size),
    }
    summary = {
        "metadata": metadata,
        "amq_budget": {
            "target_bits": float(args.target_bits),
            "target_bits_offset": float(args.target_bits_offset),
            "lower_bits_usage": float(args.target_bits - args.target_bits_offset),
            "upper_bits_usage": float(args.target_bits + args.target_bits_offset),
            "raw_lower_bits_usage": float(raw_lower),
            "raw_upper_bits_usage": float(raw_upper),
            "reachable_min_raw_bits_usage": float(min_bits_usage),
            "reachable_max_raw_bits_usage": float(max_bits_usage),
            "group_size": int(args.group_size),
            "model_numel": int(config["model_numel"]),
            "raw_selected_bits_usage": float(raw_selected_bits_usage),
            "owq_selection_offset": float(owq_selection_offset),
            "selected_bits_usage": float(selected_bits_usage),
            "selected_memory_units": float(raw_selected_bits_usage * float(config["model_numel"])),
            "selected_objective_value": float(result.fun),
            "bit_counts": {str(bit): int(count) for bit, count in sorted(bit_counts.items())},
            "candidate_counts": {key: int(value) for key, value in sorted(candidate_counts.items())},
        },
        "outputs": {
            "arch_path": str(arch_path),
            "search_stats_path": str(search_stats_path),
            "summary_path": str(summary_path),
            "result_path": None if result_path is None else str(result_path),
        },
        "per_layer": per_layer,
    }
    search_stats = {
        "archive": [[arch, float(result.fun), float(raw_selected_bits_usage)]],
        "candidates": [],
        "metadata": {
            **metadata,
            "summary_path": str(summary_path),
            "objective_value": float(result.fun),
            "raw_bits_usage": float(raw_selected_bits_usage),
            "bits_usage": float(selected_bits_usage),
        },
    }

    _json_dump(arch, arch_path)
    _json_dump(search_stats, search_stats_path)
    _json_dump(summary, summary_path)
    if result_path is not None:
        _json_dump(summary, result_path)
    return summary

