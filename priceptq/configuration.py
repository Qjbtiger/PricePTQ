"""Llama architecture metadata and AMQ-compatible storage-cost conventions."""

from __future__ import annotations

from typing import Any


def llama32_1b_amq_config() -> dict[str, Any]:
    """Return Llama-3.2-1B projection shapes and budget denominator."""
    return {
        "n_block": 16,
        "n_layer": 2,
        "layer": ["self_attn", "mlp"],
        "layer_numel": {
            "self_attn": 10485760,
            "mlp": 50331648,
        },
        "n_linear": 7,
        "linear": [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ],
        "linear_shape": {
            "self_attn.q_proj": [2048, 2048],
            "self_attn.k_proj": [512, 2048],
            "self_attn.v_proj": [512, 2048],
            "self_attn.o_proj": [2048, 2048],
            "mlp.gate_proj": [8192, 2048],
            "mlp.up_proj": [8192, 2048],
            "mlp.down_proj": [2048, 8192],
        },
        "hierarchy": {
            "self_attn.q_proj": "self_attn",
            "self_attn.k_proj": "self_attn",
            "self_attn.v_proj": "self_attn",
            "self_attn.o_proj": "self_attn",
            "mlp.gate_proj": "mlp",
            "mlp.up_proj": "mlp",
            "mlp.down_proj": "mlp",
        },
        # Use the AMQ-compatible total parameter count for effective-bit accounting.
        "model_numel": 1235814400,
        "model": "model",
        "layers": "model.layers",
        "pre_layer": ["model.embed_tokens"],
        "post_layer": ["model.norm", "lm_head", "model.rotary_emb"],
    }



def llama32_3b_amq_config() -> dict[str, Any]:
    """Return Llama-3.2-3B projection shapes and budget denominator."""
    return {
        "n_block": 28,
        "n_layer": 2,
        "layer": ["self_attn", "mlp"],
        "layer_numel": {
            "self_attn": 25165824,
            "mlp": 75497472,
        },
        "n_linear": 7,
        "linear": [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ],
        "linear_shape": {
            "self_attn.q_proj": [3072, 3072],
            "self_attn.k_proj": [1024, 3072],
            "self_attn.v_proj": [1024, 3072],
            "self_attn.o_proj": [3072, 3072],
            "mlp.gate_proj": [8192, 3072],
            "mlp.up_proj": [8192, 3072],
            "mlp.down_proj": [3072, 8192],
        },
        "hierarchy": {
            "self_attn.q_proj": "self_attn",
            "self_attn.k_proj": "self_attn",
            "self_attn.v_proj": "self_attn",
            "self_attn.o_proj": "self_attn",
            "mlp.gate_proj": "mlp",
            "mlp.up_proj": "mlp",
            "mlp.down_proj": "mlp",
        },
        "model_numel": 3212574720,
        "model": "model",
        "layers": "model.layers",
        "pre_layer": ["model.embed_tokens"],
        "post_layer": ["model.norm", "lm_head", "model.rotary_emb"],
    }



def get_amq_model_config(configs: dict[str, Any], model_name: str) -> dict[str, Any]:
    """Resolve a model name to its AMQ architecture and storage-cost metadata."""
    if model_name == "Llama-3.2-1B" and model_name not in configs:
        return llama32_1b_amq_config()
    if model_name == "Llama-3.2-3B" and model_name not in configs:
        return llama32_3b_amq_config()
    if model_name not in configs:
        raise KeyError(f"AMQ config does not contain model {model_name!r}.")
    return configs[model_name]

