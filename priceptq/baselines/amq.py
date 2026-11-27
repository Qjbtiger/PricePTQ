"""Wrapper around the AMQ proxy-based bit-allocation baseline."""

import gc
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from priceptq.runtime import read_config, seed_everything, write_json


def select_archive(archive: list, target: float = 3.0, offset: float = 0.005) -> tuple[dict, float, float]:
    """Apply AMQ's strict-budget filter and ASF selection rule."""
    from pymoo.decomposition.asf import ASF
    if not archive:
        raise ValueError("AMQ produced an empty search archive.")
    metrics = np.asarray([[float(row[1]), float(row[2])] for row in archive])
    if not np.isfinite(metrics).all():
        raise ValueError("AMQ archive contains nonfinite objectives or costs.")
    order = np.argsort(metrics[:, 0])
    eligible = order[(metrics[order, 1] > target-offset) & (metrics[order, 1] < target+offset)]
    if not len(eligible):
        raise ValueError("No AMQ candidate satisfies the requested strict budget window.")
    index = eligible[int(ASF().do(metrics[eligible], np.array([0.0, target])).argsort()[0])]
    return archive[index][0], float(metrics[index, 0]), float(metrics[index, 1])


def run_amq(model_id: str, model_name: str, output: Path, timer) -> dict:
    """Generate fresh HQQ proxies, profile sensitivity, and run the paper AMQ search."""
    from transformers import AutoModelForCausalLM
    from hqq.models.hf.base import AutoHQQHFModel
    from hqq.core.quantize import BaseQuantizeConfig
    from priceptq.vendor.amq.sensitivity import linear_sensitivity
    from priceptq.vendor.amq.search.optimizer import Search
    from priceptq.vendor.amq.utils.func import init_accelerator

    settings = read_config("selection.json")["amq"]
    config = read_config("llama.json")[model_name]
    config_path = output/"amq_config.json"
    write_json(config_path, {model_name: config})
    proxies = []
    with timer.stage("proxy_model_construction"):
        for bit in (2, 3, 4):
            seed_everything(0)
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
            quant_config = BaseQuantizeConfig(nbits=bit, group_size=128)
            AutoHQQHFModel.quantize_model(model, quant_config=quant_config, compute_dtype=torch.float16, device="cuda")
            path = output/"proxies"/f"{bit}bit"
            AutoHQQHFModel.save_quantized(model, save_dir=str(path))
            proxies.append(str(path))
            del model
            gc.collect()
            torch.cuda.empty_cache()
    sensitivity_path = output/"sensitivity.json"
    common = dict(gpu_id="0", model_id=model_id, model_name=model_name, config=str(config_path),
                  quantization_proxy_paths=proxies, bits_range=[2, 3, 4], dataset="wikitext2",
                  n_sample=32, seqlen=512, seed=0)
    with timer.stage("sensitivity_profile"):
        sensitivity = linear_sensitivity(SimpleNamespace(**common, output_file=str(sensitivity_path)))
    args = SimpleNamespace(**common, **settings, sensitivity_json=sensitivity, save_path=str(output/"search"),
                           result_file="search_args.txt", resume_path=None)
    (output/"search").mkdir()
    with timer.stage("search"):
        seed_everything(0)
        accelerator, device_map = init_accelerator("0", config)
        engine = Search(args, config, accelerator, device_map)
        engine.search(accelerator)
    with timer.stage("budgeted_archive_selection"):
        files = list((output/"search").glob("iter_*.stats"))
        if not files:
            raise RuntimeError("AMQ did not save a search checkpoint.")
        latest = max(files, key=lambda path: int(path.stem.split("_")[1]))
        checkpoint = json.loads(latest.read_text())
        arch, loss, cost = select_archive(checkpoint["archive"] + checkpoint["candidates"])
        selected = {f"{config['layers']}.{block}.{linear}": int(arch["linear"][linear][block])
                    for block in range(config["n_block"]) for linear in config["linear"]}
        write_json(output/"amq_arch.json", arch)
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return {"selected": selected, "search_loss": loss, "effective_bits": cost,
            "archive_path": str(latest), "cache_used": False}
