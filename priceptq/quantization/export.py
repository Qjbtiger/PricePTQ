"""Apply a selected GPTQ plan and export dense Hugging Face weights."""

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from priceptq.runtime import read_config, write_json


def architecture_from_selection(selected: dict[str, int], config: dict) -> dict:
    """Convert complete layer-name assignments to AMQ's projection-by-block arrays."""
    expected = {f"{config['layers']}.{block}.{linear}" for block in range(config["n_block"]) for linear in config["linear"]}
    if set(selected) != expected:
        raise ValueError("Selected bitwidths must cover every configured linear layer exactly once.")
    if any(value not in (2, 3, 4) for value in selected.values()):
        raise ValueError("The paper candidate bitwidths are 2, 3, and 4.")
    return {"linear": {linear: [selected[f"{config['layers']}.{block}.{linear}"] for block in range(config["n_block"])]
                       for linear in config["linear"]}}


def export_gptq(source: str, revision: str | None, selected: dict[str, int], model_name: str,
                destination: Path, device: torch.device) -> None:
    """Run GPTQ quantization and export dense dequantized weights."""
    if device.type != "cuda":
        raise ValueError("GPTQ export requires CUDA.")
    from priceptq.vendor.amq.quantization.gptq import GPTQ
    config = read_config("llama.json")[model_name]
    arch = architecture_from_selection(selected, config)
    tokenizer = AutoTokenizer.from_pretrained(source, revision=revision, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(source, revision=revision, torch_dtype=torch.float16,
                                                attn_implementation="eager").eval()
    quantizer = GPTQ(model, tokenizer, "gptq", arch, 3.0, group_size=128, config=config, dev=str(device))
    quantizer.run(nsamples=128, seqlen=2048, true_sequential=True, act_order=True, mse=False,
                  calib_data="wikitext2", sym=False)
    model.cpu().save_pretrained(destination, safe_serialization=True)
    tokenizer.save_pretrained(destination)
    write_json(destination/"quantization_metadata.json", {
        "format": "dense-dequantized-gptq", "group_size": 128, "selected": selected,
        "true_sequential": True, "act_order": True, "mse_search": False,
        "calibration": {"dataset": "Salesforce/wikitext", "split": "train", "samples": 128, "seqlen": 2048}})
