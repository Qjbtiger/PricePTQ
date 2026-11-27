"""Run the paper's bit-allocation baselines with explicit timing boundaries."""

from pathlib import Path
from types import SimpleNamespace

from priceptq.runtime import CONFIG_DIR, write_json


def run_baseline(args, model_info, source, revision, output, timer, device):
    """Return baseline assignments and metadata, optionally exporting a GPTQ model."""
    if device.type != "cuda" or device.index not in (None, 0):
        raise ValueError("AMQ/HIGGS launchers use one visible CUDA GPU; set CUDA_VISIBLE_DEVICES and use cuda:0.")
    if args.dataset != "Salesforce/wikitext":
        raise ValueError("Baseline loaders require Salesforce/wikitext; use the HF cache for offline runs.")
    with timer.stage("model_snapshot_resolution", selection=False):
        if Path(source).is_dir():
            resolved = str(Path(source).resolve())
        else:
            from huggingface_hub import snapshot_download
            resolved = snapshot_download(source, revision=revision, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"])
    if args.method == "higgs":
        from priceptq.baselines.higgs import solve_higgs_bit_allocation
        config = SimpleNamespace(
            model_key=args.model, model_name=model_info["name"], model_path=resolved,
            amq_config=str(CONFIG_DIR/"llama.json"), dataset_key="wikitext",
            dataset_name="Salesforce/wikitext", dataset_config="wikitext-2-raw-v1", dataset_split="validation",
            backend="gptq", bits=[2, 3, 4], group_size=128, target_bits=3.0, target_bits_offset=0.005,
            assets_dir=str(output/"assets"), output_dir=str(output/"allocation"), result_path=str(output/"solver.json"),
            time_limit=60.0, force_recompute_assets=True, force_recompute_err_coeffs=False,
            force_recompute_quant_err=False, sensitivity_num_samples=32, sensitivity_max_seqlen=512,
            sensitivity_batch_size=1, sensitivity_seed=0, sensitivity_noise_levels=16,
            sensitivity_max_relative_norm=0.01, sensitivity_metric="loss_increase", sensitivity_device=str(device),
            local_files_only=True, trust_remote_code=False, quant_error_shape=[4096, 4096],
            quant_error_trials=1, quant_error_seed=0, quant_error_device=str(device))
        with timer.stage("fresh_sensitivity_quantization_error_and_solve"):
            solution = solve_higgs_bit_allocation(args=config)
        selected = {row["raw_layer_name"]: row["selected_bits"] for row in solution["per_layer"]}
        extra = {"amq_budget": solution["amq_budget"]}
    else:
        from priceptq.baselines.amq import run_amq
        result = run_amq(resolved, model_info["name"], output, timer)
        selected = result.pop("selected")
        extra = result
    timing = timer.report()
    write_json(output/"selected.json", selected)
    if args.export_model:
        from priceptq.quantization.export import export_gptq
        with timer.stage("quantization_and_export", selection=False):
            export_gptq(resolved, None, selected, model_info["name"], output/"model", device)
        timing["quantization_and_export_seconds"] = timer.seconds["quantization_and_export"]
    return selected, {**extra, "cache_used": False, "timing": timing,
        "timing_scope": "Fresh baseline pipeline including its internal model/data loads, proxy construction, sensitivity and candidate loss evaluations; excludes initial snapshot download and final GPTQ/export."}
