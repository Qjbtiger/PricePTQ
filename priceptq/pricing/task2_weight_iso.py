"""Compute the main-paper task2 weight iso records."""

from __future__ import annotations

import torch

from priceptq.statistics.common import WhiteningTransform


def _weight_spec_key(weight_spec: dict[str, object]) -> str:
    return f"{weight_spec['quant_type']}_{weight_spec['granularity']}"



def build_rows(
    model_name: str,
    layer_name: str,
    hessian_trace: float,
    whitening: WhiteningTransform,
    weight_results: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Return JSON-compatible per-candidate prices and measured KL values for one layer."""
    sigma_sq = float(whitening.sigma_sq)
    rows: list[dict[str, object]] = []
    for record in weight_results:
        delta_w_tilde = record["delta_w_tilde"].to(torch.float64)
        d_out = delta_w_tilde.shape[0]
        alpha = float(delta_w_tilde.square().sum().item() / max(d_out, 1))
        theory_value = 0.5 * sigma_sq * alpha * float(hessian_trace)
        weight_spec = record["weight_spec"]
        rows.append(
            {
                "task_id": "task2_weight_iso",
                "model_name": model_name,
                "layer_name": layer_name,
                "constraint": whitening.constraint,
                "weight_quant_config": _weight_spec_key(weight_spec),
                "weight_quant_type": weight_spec["quant_type"],
                "weight_granularity": weight_spec["granularity"],
                "empirical_kl": float(record["empirical_kl"]),
                "theory_value": float(theory_value),
                "alpha": alpha,
                "sigma_sq": sigma_sq,
                "hessian_trace": float(hessian_trace),
            }
        )
    return rows

