"""Compute the main-paper task1 weight trace records."""

from __future__ import annotations

import torch

from priceptq.statistics.common import HessianSketch, WhiteningTransform, estimate_trace_hm


def _weight_spec_key(weight_spec: dict[str, object]) -> str:
    return f"{weight_spec['quant_type']}_{weight_spec['granularity']}"



def build_rows(
    model_name: str,
    layer_name: str,
    weight: torch.Tensor,
    hessian_sketch: HessianSketch,
    c_x: torch.Tensor,
    whitening: WhiteningTransform,
    weight_results: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Return JSON-compatible per-candidate prices and measured KL values for one layer."""
    c_tilde = whitening.p.to(dtype=torch.float64) @ c_x.to(dtype=torch.float64) @ whitening.p.to(dtype=torch.float64).T
    c_tilde = 0.5 * (c_tilde + c_tilde.T)

    rows: list[dict[str, object]] = []
    for record in weight_results:
        delta_w_tilde = record["delta_w_tilde"].to(torch.float64)
        trace_matrix = delta_w_tilde @ c_tilde @ delta_w_tilde.T
        theory_value = 0.5 * estimate_trace_hm(hessian_sketch, trace_matrix.float())

        weight_spec = record["weight_spec"]
        rows.append(
            {
                "task_id": "task1_weight_trace",
                "model_name": model_name,
                "layer_name": layer_name,
                "constraint": whitening.constraint,
                "weight_quant_config": _weight_spec_key(weight_spec),
                "weight_quant_type": weight_spec["quant_type"],
                "weight_granularity": weight_spec["granularity"],
                "empirical_kl": float(record["empirical_kl"]),
                "theory_value": float(theory_value),
                "sigma_sq": float(whitening.sigma_sq),
                "whitening_error": dict(whitening.verification),
                "delta_w_frobenius": float(delta_w_tilde.square().sum().sqrt().item()),
            }
        )
    return rows

