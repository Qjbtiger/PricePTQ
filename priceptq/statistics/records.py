"""Read per-layer statistics for configuration selection."""

from __future__ import annotations

import json

import re

from dataclasses import asdict, dataclass

from pathlib import Path

from typing import Any


LAYER_NAME_RE = re.compile(
    r"^model\.layers\.(\d+)\."
    r"(self_attn\.[qkvo]_proj|mlp\.(?:gate|up|down)_proj)$"
)



@dataclass(frozen=True)
class LayerStatistics:
    """Normalized layer statistics consumed by the budgeted allocation solver."""
    qdict_key: str
    raw_layer_name: str
    sigma_sq: float
    whitening_type: str
    tr_hz: float
    gamma: float
    weight_alpha: float | None
    activation_beta: float | None
    token_count: int | None
    input_dim: int | None
    output_dim: int | None
    model_name: str | None
    model_short_name: str | None

    def to_summary_dict(self) -> dict[str, Any]:
        """Return this layer record as a JSON-compatible mapping."""
        return asdict(self)



@dataclass(frozen=True)
class LoadedStatistics:
    """Loaded metadata and normalized statistics indexed by layer key."""
    stats_path: str
    metadata: dict[str, Any]
    layers: dict[str, LayerStatistics]



def _normalize_layer_name(raw_layer_name: str) -> str:
    match = LAYER_NAME_RE.match(raw_layer_name)
    if match is None:
        raise ValueError(f"Unsupported layer name in statistics file: {raw_layer_name}")
    return f"{match.group(1)}_{match.group(2)}"



def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)



def _sigma_sq(entry: dict[str, Any], whitening_type: str) -> float:
    if whitening_type == "none" and "constraints" not in entry:
        return float(entry["input_cov_sigma_sq_none"])
    return float(entry["constraints"][whitening_type]["sigma_sq"])



def _tr_hz(entry: dict[str, Any]) -> float:
    for key in ("hessian_trace", "hessian_trace_estimate", "hessian_trace_upper_bound"):
        if key in entry:
            return float(entry[key])
    raise KeyError("Statistics entry does not contain a Hessian trace estimate.")



def _gamma(entry: dict[str, Any]) -> float:
    # Default neutral placeholder for weight-only statistics records.
    return float(entry.get("gamma", 1.0))



def _per_tensor_absmax(
    entry: dict[str, Any],
    whitening_type: str,
    range_group: str,
) -> float | None:
    if "constraints" not in entry:
        return None
    try:
        return float(entry["constraints"][whitening_type][range_group]["per-tensor"]["absmax"])
    except KeyError:
        return None



def _layer_statistics(entry: dict[str, Any], whitening_type: str) -> LayerStatistics:
    raw_layer_name = entry["layer_name"]
    weight_absmax = _per_tensor_absmax(entry, whitening_type, "weight_ranges")
    activation_absmax = _per_tensor_absmax(entry, whitening_type, "activation_ranges")
    return LayerStatistics(
        qdict_key=_normalize_layer_name(raw_layer_name),
        raw_layer_name=raw_layer_name,
        sigma_sq=_sigma_sq(entry, whitening_type),
        whitening_type=whitening_type,
        tr_hz=_tr_hz(entry),
        gamma=_gamma(entry),
        weight_alpha=None if weight_absmax is None else weight_absmax**2,
        activation_beta=None if activation_absmax is None else activation_absmax**2,
        token_count=_optional_int(entry.get("token_count")),
        input_dim=_optional_int(entry.get("input_dim")),
        output_dim=_optional_int(entry.get("output_dim")),
        model_name=entry.get("model_name"),
        model_short_name=entry.get("model_short_name"),
    )



def load_layer_statistics(stats_path: str | Path, whitening_type: str = "none") -> LoadedStatistics:
    """Load and normalize per-layer JSON statistics in the requested whitening coordinate."""
    stats_path = Path(stats_path)
    with stats_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    layers = {
        stats.qdict_key: stats
        for stats in (_layer_statistics(entry, whitening_type) for entry in data["layers"])
    }

    return LoadedStatistics(
        stats_path=str(stats_path),
        metadata=data.get("metadata", {}),
        layers=layers,
    )

