"""Run one independently timed main-paper configuration selector."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from priceptq import bit_allocation
from priceptq.candidates import granularity, transformation
from priceptq.runtime import CONFIG_DIR, Timer, environment, prepare_output, read_config, seed_everything, token_fingerprint, write_json
from priceptq.statistics.collect import build_layer_record, collect_weight_only_statistics
from priceptq.statistics.common import build_calibration_batches


METHODS = {"bitwidth": ("ours", "higgs", "amq"),
           "transformation": ("ours", "calm-cka", "random"),
           "granularity": ("ours", "fixed-group128", "random")}


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI for the fixed main-paper selection protocol."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=METHODS, required=True)
    parser.add_argument("--method", choices=sorted({m for methods in METHODS.values() for m in methods}), default="ours")
    parser.add_argument("--model", choices=("llama-1b", "llama-3b", "llama-8b"), required=True)
    parser.add_argument("--model-path", help="Optional local HF checkpoint or alternative model ID for this architecture.")
    parser.add_argument("--revision", help="Override the default model revision when using an HF ID.")
    parser.add_argument("--dataset", default="Salesforce/wikitext", help="HF dataset ID or local validation .arrow file (Ours/transform/granularity).")
    parser.add_argument("--statistics", type=Path, help="Reuse explicit statistics; reported as a cached run.")
    parser.add_argument("--statistics-batch-size", type=int, help="Batch size for statistics collection (defaults to the per-model config value).")
    parser.add_argument("--seed", type=int, default=0, help="Random-selection seed; shared calibration stays at seed 0.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--output", required=True)
    parser.add_argument("--export-model", action="store_true", help="Export dense quantized model weights after selection.")
    return parser


def build_transform_cache(layers, inputs, *, device, seed=0):
    """Prepare the same transformation candidates for independently timed baselines."""
    cache = {}
    for name, layer in layers.items():
        cache[name] = {}
        for candidate in transformation.transformations_for_input_dim(layer.in_features):
            state = transformation.build_transform(candidate, layer_name=name, x_sample=inputs[name],
                                                    weight=layer.weight.detach().float().cpu(), seed=seed)
            cache[name][candidate] = transformation.quantize_layer(layer, state, bitwidth=4, group_size=128, device=device)
    return cache


def validate_model_layers(layers, model_name: str) -> None:
    """Reject checkpoints whose projection names or dimensions differ from the budget configuration."""
    config = read_config("llama.json")[model_name]
    expected = {f"{config['layers']}.{block}.{linear}": tuple(config["linear_shape"][linear])
                for block in range(config["n_block"]) for linear in config["linear"]}
    if set(layers) != set(expected):
        raise ValueError("Checkpoint layers differ from the selected paper model architecture.")
    for name, layer in layers.items():
        if tuple(layer.weight.shape) != expected[name]:
            raise ValueError(f"Checkpoint projection shape differs from the cost configuration at {name}.")


def _granularity_baseline(layers, method, seed, device):
    # Sample using the same candidate order and RNG sequence as the price-guided selector.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    selected, quantized = {}, {}
    for name, layer in layers.items():
        candidate = "group-128" if method == "fixed-group128" else granularity.GRANULARITIES[
            int(torch.randint(0, len(granularity.GRANULARITIES), (1,), generator=generator).item())]
        selected[name] = candidate
        cost = granularity.candidate_cost(candidate, input_dim=layer.in_features, output_dim=layer.out_features,
                                          bitwidth=4, latency_proxy=0.0, latency_source="none")
        quantized[name], _ = granularity.quantize_layer(layer, granularity=candidate, bitwidth=4, cost=cost, device=device)
    return selected, quantized


def _run_ours_or_local_baseline(args, protocol, model_info, source, revision, output, timer, device):
    batch_size = args.statistics_batch_size or model_info["statistics_batch_size"]
    need_statistics = args.method == "ours"
    dtype = torch.float32 if need_statistics and args.statistics is None else torch.float16
    if device.type == "cpu":
        dtype = torch.float32
    with timer.stage("model_and_tokenizer_load", selection=False):
        tokenizer = AutoTokenizer.from_pretrained(source, revision=revision, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(source, revision=revision, torch_dtype=dtype,
                                                    attn_implementation="sdpa").to(device).eval()
        model.config.use_cache = False
    with timer.stage("calibration_data_build", selection=False):
        batches = build_calibration_batches(tokenizer, dataset_name=args.dataset,
                                           num_samples=32, max_seqlen=512, batch_size=batch_size, seed=0)
        layers = transformation.get_quantizable_linear_layers(model, layer_pattern=None, max_layers=None)
        if len(layers) != model_info["layers"]:
            raise ValueError(f"Expected {model_info['layers']} quantized layers, found {len(layers)}")
        validate_model_layers(layers, model_info["name"])
        fingerprint = token_fingerprint(batches)
    statistics = None
    stats_path = output / "statistics.json"
    if need_statistics:
        with timer.stage("statistics_cache_read" if args.statistics else "statistics_collection"):
            if args.statistics:
                payload = json.loads(args.statistics.read_text())
                recorded_hash = payload.get("metadata", {}).get("calibration_token_sha256")
                if recorded_hash != fingerprint:
                    raise ValueError("Cached statistics must record the same ordered calibration token fingerprint.")
                if (payload["metadata"].get("model_source"), payload["metadata"].get("revision")) != (source, revision):
                    raise ValueError("Cached statistics must use the same model source and revision.")
                records = payload["layers"]
                if {r["layer_name"] for r in records} != set(layers):
                    raise ValueError("Cached statistics cover a different layer set.")
            else:
                statistics = collect_weight_only_statistics(model=model, layers=layers, batches=batches, device=device)
                records = [build_layer_record(model_name=model_info["name"], layer_name=name,
                            layer_module=module, statistics=statistics[name]) for name, module in layers.items()]
            write_json(stats_path, {"metadata": {"calibration_token_sha256": fingerprint, "num_samples": 32,
                        "max_seqlen": 512, "seed": 0, "batch_size": batch_size, "dtype": "float32",
                        "model_source": source, "revision": revision}, "layers": records})
        statistics = {r["layer_name"]: r for r in records}
    prices, quantized, latency = None, None, None
    if args.family == "bitwidth":
        with timer.stage("candidate_scoring_and_solve"):
            solution = bit_allocation.solve_amq_bit_allocation(
                stats_path=str(stats_path), amq_summary_path=None, amq_config_path=str(CONFIG_DIR/"llama.json"),
                model_name=model_info["name"], backend="gptq", mode="weight_only", whitening_type="none",
                target_bits=3.0, target_bits_offset=0.005, group_size=128, bits=[2, 3, 4],
                output_dir=str(output/"allocation"), result_path=str(output/"solver.json"), time_limit=60.0)
            selected = {r["raw_layer_name"]: r["selected_bits"] for r in solution["per_layer"]}
        extra = {"candidate_count": 3*len(layers), "amq_budget": solution["amq_budget"]}
    else:
        with timer.stage("candidate_model_dtype_conversion"):
            model.to(dtype=torch.float16 if device.type == "cuda" else torch.float32)
        if args.family == "transformation" or args.method == "ours":
            with timer.stage("price_input_collection"):
                ids = torch.cat([b["input_ids"] for b in batches])
                labels = torch.cat([b["labels"] for b in batches])
                masks = torch.cat([b["loss_mask"] for b in batches])
                price_batches = [{"input_ids": ids[i:i+4], "labels": labels[i:i+4], "loss_mask": masks[i:i+4]}
                                 for i in range(0, len(ids), 4)]
                inputs = transformation.collect_layer_inputs(model, layers, price_batches, device=device, max_price_tokens=512)
        if args.family == "transformation":
            if args.method == "ours":
                with timer.stage("candidate_scoring_and_solve"):
                    selected, prices, quantized, cache = transformation.price_transformations(
                        layers=layers, layer_inputs=inputs, stats_by_layer=statistics,
                        bitwidth=4, group_size=128, seed=0, device=device)
            else:
                with timer.stage("candidate_construction"):
                    cache = build_transform_cache(layers, inputs, device=device)
                with timer.stage("candidate_scoring_and_solve"):
                    if args.method == "calm-cka":
                        selected, prices, quantized = transformation.calm_cka_selection(
                            layers=layers, layer_inputs=inputs, quantized_cache=cache, device=device)
                    else:
                        selected, quantized = transformation.random_quantized_selection(
                            layer_names=list(layers), quantized_cache=cache, seed=args.seed)
            count = sum(len(c) for c in cache.values())
            del cache
        elif args.method == "ours":
            with timer.stage("latency_table_build"):
                latency = granularity.build_latency_table(layers, bitwidth=4, device=device,
                        dtype=torch.float16 if device.type == "cuda" else torch.float32,
                        skip_benchmark=False, latency_tokens=128, warmup=10, iters=50, seed=0)
            with timer.stage("candidate_scoring_and_solve"):
                selected, prices, quantized, cache = granularity.price_granularities(
                    layers=layers, layer_inputs=inputs, stats_by_layer=statistics, bitwidth=4,
                    latency_table=latency, device=device)
            count = sum(len(c) for c in cache.values())
            del cache
        else:
            with timer.stage("candidate_construction_and_selection"):
                selected, quantized = _granularity_baseline(layers, args.method, args.seed, device)
            count = len(granularity.GRANULARITIES) * len(layers)
        extra = {"candidate_count": count}
    timing = timer.report()
    write_json(output/"selected.json", selected)
    if prices is not None:
        write_json(output/"prices.json", prices)
    if latency is not None:
        write_json(output/"latency_table.json", latency)
    if args.export_model:
        with timer.stage("quantization_and_export", selection=False):
            if args.family == "bitwidth":
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                from priceptq.quantization.export import export_gptq
                export_gptq(source, revision, selected, model_info["name"], output/"model", device)
            else:
                if args.family == "transformation":
                    transformation.write_folded_fake_quant_weights_to_model(model, quantized)
                else:
                    granularity.write_fake_quant_weights_to_model(model, quantized)
                model.save_pretrained(output/"model", safe_serialization=True)
                tokenizer.save_pretrained(output/"model")
        timing["quantization_and_export_seconds"] = timer.seconds["quantization_and_export"]
    return selected, {**extra, "calibration_token_sha256": fingerprint, "statistics_batch_size": batch_size,
                      "statistics_cache_used": args.statistics is not None, "timing": timing}


def run(args) -> dict:
    """Execute one selector and save its choices, timing, and reproduction metadata."""
    if args.method not in METHODS[args.family]:
        raise ValueError(f"{args.method} is not a baseline for {args.family}")
    if args.statistics and args.method != "ours":
        raise ValueError("Only Ours consumes a statistics cache.")
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the paper timing protocol.")
        torch.cuda.set_device(device)
    torch.set_num_threads(args.cpu_threads)
    seed_everything(0)
    protocol = read_config("selection.json")
    model_info = protocol["models"][args.model]
    source = args.model_path or model_info["hf_id"]
    revision = args.revision or (None if args.model_path else model_info["revision"])
    output = prepare_output(args.output)
    timer = Timer(device)
    if args.family == "bitwidth" and args.method != "ours":
        from priceptq.baselines.run import run_baseline
        selected, extra = run_baseline(args, model_info, source, revision, output, timer, device)
    else:
        selected, extra = _run_ours_or_local_baseline(args, protocol, model_info, source, revision, output, timer, device)
    result = {"family": args.family, "method": args.method, "model": model_info["name"],
              "model_source": source, "revision": revision, "random_seed": args.seed, "layer_count": len(selected),
              "protocol": protocol["protocol"], "environment": environment(device), **extra}
    write_json(output/"result.json", result)
    print(json.dumps({"output": str(output), "selection_seconds": result["timing"]["selection_seconds"]}, indent=2))
    return result


def main() -> None:
    """Parse arguments and execute one paper experiment."""
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
