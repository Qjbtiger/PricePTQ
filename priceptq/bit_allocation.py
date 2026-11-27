"""Weight-only bit-allocation objective and MILP solver."""

from __future__ import annotations

import json

import math

from collections import Counter

from pathlib import Path

from typing import Any

import numpy as np

from scipy.optimize import Bounds, LinearConstraint, milp

from scipy.sparse import lil_matrix

from priceptq.configuration import get_amq_model_config

from priceptq.statistics.records import LayerStatistics, load_layer_statistics


def _load_amq_summary(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)



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



def _validate_layer_coverage(stats_layers: dict[str, LayerStatistics], config: dict[str, Any]) -> list[str]:
    ordered_keys = _amq_layer_keys(config)
    expected = set(ordered_keys)
    actual = set(stats_layers)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing={missing}")
        if extra:
            parts.append(f"extra={extra}")
        raise ValueError("Statistics file does not match the expected AMQ layer set: " + ", ".join(parts))
    return ordered_keys



def _objective_coefficient(stats: LayerStatistics, mode: str) -> float:
    def effective_weight_alpha() -> float:
        """Convert the recorded weight range into an input-dimension-scaled perturbation factor."""
        if stats.weight_alpha is None:
            raise ValueError(
                "Objective requires per-tensor weight range statistics "
                f"to compute alpha. Re-collect full statistics for {stats.raw_layer_name}."
            )
        if stats.input_dim is None:
            raise ValueError(
                "Objective requires input_dim to convert per-entry weight range into "
                f"E[Delta W Delta W^T] scale for {stats.raw_layer_name}."
            )
        return float(stats.input_dim) * stats.weight_alpha

    if mode == "weight_only":
        weight_alpha_eff = effective_weight_alpha()
        product = 0.5 * stats.sigma_sq * weight_alpha_eff * stats.tr_hz
        if product <= 0.0:
            raise ValueError(
                "Weight-only objective requires 0.5 * sigma_sq * alpha * tr_hz > 0 for every layer. "
                f"Got sigma_sq={stats.sigma_sq}, alpha={weight_alpha_eff}, tr_hz={stats.tr_hz} "
                f"at {stats.raw_layer_name}."
            )
        return product
    raise ValueError(f"Unsupported mode: {mode}")



def _effective_weight_alpha(stats: LayerStatistics) -> float | None:
    if stats.weight_alpha is None or stats.input_dim is None:
        return None
    return float(stats.input_dim) * stats.weight_alpha



def _bit_decay(bits: int) -> float:
    return 2.0 ** (-float(bits))



def _amq_memory_units(config: dict[str, Any], linear_key: str, bit: int, group_size: int) -> float:
    out_dim, in_dim = config["linear_shape"][linear_key]
    c_group_size = in_dim if group_size == -1 else group_size
    effective_bit = float(bit) + (32.0 / float(c_group_size) if bit < 16 else 0.0)
    return float(out_dim) * float(in_dim) * effective_bit



def _format_target(value: float) -> str:
    formatted = f"{value:.6f}".rstrip("0").rstrip(".")
    return formatted.replace(".", "p")



def _solve_milp(
    *,
    ordered_keys: list[str],
    stats_layers: dict[str, LayerStatistics],
    config: dict[str, Any],
    bits: list[int],
    mode: str,
    group_size: int,
    target_bits: float,
    target_bits_offset: float,
    time_limit: float,
) -> tuple[dict[str, int], dict[str, float], Any]:
    bits = sorted(set(int(bit) for bit in bits))
    n_layers = len(ordered_keys)
    n_bits = len(bits)
    n_vars = n_layers * n_bits

    coefficients = {
        qdict_key: _objective_coefficient(stats_layers[qdict_key], mode)
        for qdict_key in ordered_keys
    }

    c = np.zeros(n_vars, dtype=np.float64)
    memory = np.zeros(n_vars, dtype=np.float64)
    var_index: dict[tuple[str, int], int] = {}

    for layer_idx, qdict_key in enumerate(ordered_keys):
        _, linear_key = _split_qdict_key(qdict_key)
        coeff = coefficients[qdict_key]
        for bit_idx, bit in enumerate(bits):
            idx = layer_idx * n_bits + bit_idx
            var_index[(qdict_key, bit)] = idx
            c[idx] = coeff * _bit_decay(bit)
            memory[idx] = _amq_memory_units(config, linear_key, bit, group_size)

    constraints = []

    one_hot = lil_matrix((n_layers, n_vars), dtype=np.float64)
    for layer_idx in range(n_layers):
        for bit_idx in range(n_bits):
            one_hot[layer_idx, layer_idx * n_bits + bit_idx] = 1.0
    constraints.append(LinearConstraint(one_hot.tocsr(), np.ones(n_layers), np.ones(n_layers)))

    budget = lil_matrix((1, n_vars), dtype=np.float64)
    budget[0, :] = memory
    strict_eps = 1e-10
    lower = (target_bits - target_bits_offset + strict_eps) * float(config["model_numel"])
    upper = (target_bits + target_bits_offset - strict_eps) * float(config["model_numel"])
    constraints.append(LinearConstraint(budget.tocsr(), np.array([lower]), np.array([upper])))

    result = milp(
        c=c,
        integrality=np.ones(n_vars, dtype=np.int8),
        bounds=Bounds(np.zeros(n_vars), np.ones(n_vars)),
        constraints=constraints,
        options={"time_limit": time_limit, "mip_rel_gap": 0.0},
    )
    if not result.success:
        raise RuntimeError(f"AMQ-aligned MILP failed: status={result.status}, message={result.message}")

    selected: dict[str, int] = {}
    for qdict_key in ordered_keys:
        chosen = [
            bit
            for bit in bits
            if result.x[var_index[(qdict_key, bit)]] > 0.5
        ]
        if len(chosen) != 1:
            raise RuntimeError(f"MILP returned invalid assignment for {qdict_key}: {chosen}")
        selected[qdict_key] = chosen[0]

    return selected, coefficients, result



def _architecture_from_selection(
    selected_bits: dict[str, int],
    config: dict[str, Any],
) -> dict[str, dict[str, list[int]]]:
    arch = {"linear": {linear_key: [] for linear_key in config["linear"]}}
    for block_idx in range(int(config["n_block"])):
        for linear_key in config["linear"]:
            arch["linear"][linear_key].append(int(selected_bits[f"{block_idx}_{linear_key}"]))
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
    min_bit = candidate_bits[0]
    max_bit = candidate_bits[-1]
    min_memory = 0.0
    max_memory = 0.0
    for qdict_key in ordered_keys:
        _, linear_key = _split_qdict_key(qdict_key)
        min_memory += _amq_memory_units(config, linear_key, min_bit, group_size)
        max_memory += _amq_memory_units(config, linear_key, max_bit, group_size)
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
) -> tuple[float, float]:
    min_usage, max_usage = _bits_usage_bounds(ordered_keys, config, bits, group_size)
    lower = target_bits - target_bits_offset
    upper = target_bits + target_bits_offset
    if lower >= max_usage or upper <= min_usage:
        raise ValueError(
            "Requested AMQ target window is outside the reachable bits_usage range. "
            f"target_window=({lower:.12f}, {upper:.12f}), "
            f"reachable_range=[{min_usage:.12f}, {max_usage:.12f}], "
            f"candidate_bits={sorted(set(int(bit) for bit in bits))}, group_size={group_size}. "
            "Choose a target within the reachable range, for example target-bits=3 for 2/3/4-bit Llama-3.2-1B."
        )
    return min_usage, max_usage



def solve_amq_bit_allocation(
    *,
    stats_path: str,
    amq_summary_path: str | None,
    amq_config_path: str,
    model_name: str,
    backend: str,
    mode: str,
    whitening_type: str,
    target_bits: float,
    target_bits_offset: float,
    group_size: int,
    bits: list[int],
    output_dir: str,
    result_path: str | None,
    time_limit: float,
) -> dict[str, Any]:
    """Solve the fixed candidate budget from layer statistics and write the selected architecture."""
    amq_summary = _load_amq_summary(amq_summary_path)
    if amq_summary:
        model_name = Path(amq_summary.get("model_path", model_name)).name or model_name
        target_bits = float(amq_summary.get("target_bits", target_bits))
        target_bits_offset = float(amq_summary.get("target_bits_offset", target_bits_offset))
        group_size = int(amq_summary.get("group_size", group_size))

    config = _load_amq_config(amq_config_path, model_name)
    loaded_stats = load_layer_statistics(stats_path, whitening_type)
    ordered_keys = _validate_layer_coverage(loaded_stats.layers, config)
    min_bits_usage, max_bits_usage = _validate_target_window(
        ordered_keys=ordered_keys,
        config=config,
        bits=bits,
        group_size=group_size,
        target_bits=target_bits,
        target_bits_offset=target_bits_offset,
    )

    selected_bits, coefficients, result = _solve_milp(
        ordered_keys=ordered_keys,
        stats_layers=loaded_stats.layers,
        config=config,
        bits=bits,
        mode=mode,
        group_size=group_size,
        target_bits=target_bits,
        target_bits_offset=target_bits_offset,
        time_limit=time_limit,
    )
    arch = _architecture_from_selection(selected_bits, config)
    selected_bits_usage = _bits_usage(arch, config, group_size)
    bit_counts = Counter(selected_bits.values())

    if not (target_bits - target_bits_offset < selected_bits_usage < target_bits + target_bits_offset):
        raise RuntimeError(
            f"Selected AMQ bits_usage={selected_bits_usage} is outside strict AMQ target window "
            f"({target_bits - target_bits_offset}, {target_bits + target_bits_offset})."
        )

    output_root = (
        Path(output_dir)
        / model_name
        / mode
        / f"whitening_{whitening_type}"
        / backend
        / f"target_{_format_target(target_bits)}_offset_{_format_target(target_bits_offset)}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    arch_path = output_root / "amq_arch.json"
    search_stats_path = output_root / "iter_ours.stats"
    summary_path = output_root / "allocation_summary.json"
    result_path_obj = None if result_path is None else Path(result_path)

    per_layer_summary = []
    for qdict_key in ordered_keys:
        block_idx, linear_key = _split_qdict_key(qdict_key)
        selected_bit = selected_bits[qdict_key]
        stats = loaded_stats.layers[qdict_key]
        per_layer_summary.append(
            {
                "qdict_key": qdict_key,
                "block_idx": block_idx,
                "linear_key": linear_key,
                "raw_layer_name": stats.raw_layer_name,
                "objective_coefficient": coefficients[qdict_key],
                "effective_weight_alpha": _effective_weight_alpha(stats),
                "selected_bits": selected_bit,
                "selected_objective_value": coefficients[qdict_key] * _bit_decay(selected_bit),
                "amq_memory_units": _amq_memory_units(config, linear_key, selected_bit, group_size),
                "stats": stats.to_summary_dict(),
            }
        )

    summary = {
        "metadata": {
            "solver": "scipy.optimize.milp",
            "solver_message": str(result.message),
            "model_name": model_name,
            "backend": backend,
            "stats_path": str(Path(stats_path)),
            "amq_summary_path": None if amq_summary_path is None else str(Path(amq_summary_path)),
            "amq_config_path": str(Path(amq_config_path)),
            "mode": mode,
            "whitening_type": whitening_type,
            "candidate_bits": sorted(set(int(bit) for bit in bits)),
            "objective_form": (
                "weight_only: sum_lb x_lb * (0.5 * sigma_sq_l * alpha_eff_l * Tr(Hbar_z_l)) * 2^{-b}; "
                "alpha_eff_l = input_dim_l * absmax(W_tilde_l)^2 approximates E[Delta W Delta W^T]"
            ),
        },
        "amq_budget": {
            "target_bits": float(target_bits),
            "target_bits_offset": float(target_bits_offset),
            "strict_lower_bits_usage": float(target_bits - target_bits_offset),
            "strict_upper_bits_usage": float(target_bits + target_bits_offset),
            "reachable_min_bits_usage": float(min_bits_usage),
            "reachable_max_bits_usage": float(max_bits_usage),
            "group_size": int(group_size),
            "model_numel": int(config["model_numel"]),
            "selected_bits_usage": float(selected_bits_usage),
            "selected_memory_units": float(selected_bits_usage * float(config["model_numel"])),
            "bit_counts": {str(bit): int(count) for bit, count in sorted(bit_counts.items())},
        },
        "reference_amq": amq_summary,
        "outputs": {
            "arch_path": str(arch_path),
            "search_stats_path": str(search_stats_path),
            "summary_path": str(summary_path),
            "result_path": None if result_path_obj is None else str(result_path_obj),
        },
        "per_layer": per_layer_summary,
    }

    search_stats = {
        "archive": [[arch, float(result.fun), float(selected_bits_usage)]],
        "candidates": [],
        "metadata": {
            "source": "priceptq.bit_allocation",
            "backend": backend,
            "summary_path": str(summary_path),
            "objective_value": float(result.fun),
            "bits_usage": float(selected_bits_usage),
        },
    }

    with arch_path.open("w", encoding="utf-8") as handle:
        json.dump(arch, handle, indent=2)
    with search_stats_path.open("w", encoding="utf-8") as handle:
        json.dump(search_stats, handle, indent=2)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    if result_path_obj is not None:
        result_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with result_path_obj.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

    return summary

