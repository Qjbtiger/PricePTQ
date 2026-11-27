"""Main-paper realized-price and trace-price comparisons with forward KL."""

import argparse
from contextlib import contextmanager
import csv
from itertools import combinations
import random

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

from priceptq.pricing import task1_weight_trace, task2_weight_iso
from priceptq.quantization.fake_quant import fake_quantize
from priceptq.runtime import Timer, environment, prepare_output, read_config, seed_everything, token_fingerprint, write_json
from priceptq.statistics.common import (QuantSpec, build_calibration_batches, collect_layer_statistics,
    compute_masked_forward_kl, compute_whitening_transform, estimate_hessian_trace,
    get_quantizable_linear_layers, stable_seed)


@contextmanager
def replace_weights(replacements: list[tuple[torch.nn.Linear, torch.Tensor]]):
    """Temporarily replace weights and restore the reference even on a failed forward."""
    originals = []
    try:
        with torch.no_grad():
            for module, weight in replacements:
                originals.append((module, module.weight.detach().clone()))
                module.weight.copy_(weight.to(module.weight))
        yield
    finally:
        with torch.no_grad():
            for module, original in originals:
                module.weight.copy_(original)


def measure_kl(model, batches, plans: dict, device: torch.device) -> dict[str, float]:
    """Measure token-averaged reference-to-perturbed KL for weight-only plans."""
    sums = dict.fromkeys(plans, 0.0)
    counts = dict.fromkeys(plans, 0)
    with torch.no_grad():
        for batch in batches:
            inputs, mask = batch["input_ids"].to(device), batch["loss_mask"].to(device)
            reference = model(input_ids=inputs, use_cache=False).logits
            for key, replacements in plans.items():
                with replace_weights(replacements):
                    perturbed = model(input_ids=inputs, use_cache=False).logits
                value, count = compute_masked_forward_kl(reference, perturbed, mask)
                sums[key] += value
                counts[key] += count
    if any(count == 0 for count in counts.values()):
        raise ValueError("KL measurement requires supervised tokens.")
    return {key: max(sums[key]/counts[key], 0.0) for key in plans}


def alignment(rows: list[dict]) -> dict:
    """Compute log-Pearson and Spearman alignment metrics over positive finite price/KL pairs."""
    x = np.asarray([r["theory_value"] for r in rows], dtype=float)
    y = np.asarray([r["empirical_kl"] for r in rows], dtype=float)
    keep = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    result = {"n": int(keep.sum()), "excluded": int((~keep).sum()), "pearson_log10": None, "spearman": None}
    x, y = x[keep], y[keep]
    if len(x) >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0:
        result.update(pearson_log10=float(pearsonr(np.log10(x), np.log10(y)).statistic),
                      spearman=float(spearmanr(x, y).statistic))
    return result


def run_model(model, batches, layers, settings: dict, device: torch.device, timer: Timer) -> dict:
    """Compute main-paper single-layer prices and sampled two-layer perturbations."""
    constraints = settings["constraints"]
    specs = [QuantSpec(quant_type=q, granularity=g) for q in settings["weight_quant_types"]
             for g in settings["weight_granularities"]]
    results = {"task1_weight_trace": [], "task2_weight_iso": [], "layer_summaries": [], "layer_pairs": []}
    cache, lookup = {}, {}
    for name, module in layers.items():
        with timer.stage("statistics_and_prices"):
            sketch, covariance, summary = collect_layer_statistics(model, name, module, batches, device,
                                                                   settings["num_hessian_probes"])
            trace = estimate_hessian_trace(sketch)
            weight = module.weight.detach().float()
            whitenings = {c: compute_whitening_transform(covariance, c) for c in constraints}
            records, plans = {}, {}
            for constraint, whitening in whitenings.items():
                records[constraint] = []
                transformed = weight @ whitening.p_inv
                for spec in specs:
                    key = f"{constraint}:{spec.key}"
                    quantized = fake_quantize(transformed, spec.quant_type, spec.granularity).float()
                    records[constraint].append({"plan_key": key, "weight_spec": spec.__dict__,
                                                "delta_w_tilde": quantized-transformed})
                    plans[key] = [(module, quantized @ whitening.p)]
        with timer.stage("single_layer_kl"):
            measured = measure_kl(model, batches, plans, device)
        with timer.stage("price_records"):
            for constraint, whitening in whitenings.items():
                local = [{**r, "empirical_kl": measured[r["plan_key"]]} for r in records[constraint]]
                real = task1_weight_trace.build_rows(settings["model_id"], name, weight, sketch, covariance, whitening, local)
                scalar = task2_weight_iso.build_rows(settings["model_id"], name, trace, whitening, local)
                results["task1_weight_trace"].extend(real)
                results["task2_weight_iso"].extend(scalar)
                for a, b in zip(real, scalar, strict=True):
                    lookup[(name, constraint, a["weight_quant_config"])] = (a["theory_value"], b["theory_value"])
            results["layer_summaries"].append({"layer_name": name, **summary, "hessian_trace": float(trace)})
            if settings["num_layer_pairs"]:
                cache[name] = {c: (w.p.cpu(), w.p_inv.cpu()) for c, w in whitenings.items()}
        del sketch, covariance, whitenings, records, plans, weight

    if settings["num_layer_pairs"] and len(layers) >= 2:
        seed_name = settings.get("pair_seed_name") or settings["model_id"]
        rng = random.Random(stable_seed(f"{settings['seed']}:{seed_name}:layer_pairs"))
        pairs = list(combinations(layers, 2))
        rng.shuffle(pairs)
        with timer.stage("two_layer_kl"):
            for left, right in pairs[:settings["num_layer_pairs"]]:
                constraint = rng.choice(constraints)
                chosen = [rng.choice(specs), rng.choice(specs)]
                plans, real_price, scalar_price = [], 0.0, 0.0
                for name, spec in zip((left, right), chosen, strict=True):
                    module = layers[name]
                    p, inverse = (x.to(device) for x in cache[name][constraint])
                    quantized = fake_quantize(module.weight.detach().float() @ inverse, spec.quant_type, spec.granularity)
                    plans.append((module, quantized.float() @ p))
                    real, scalar = lookup[(name, constraint, spec.key)]
                    real_price += real
                    scalar_price += scalar
                kl = measure_kl(model, batches, {"pair": plans}, device)["pair"]
                results["layer_pairs"].append({"layers": [left, right], "constraint": constraint,
                    "weight_specs": [s.__dict__ for s in chosen], "empirical_kl": kl,
                    "realized_price": real_price, "trace_price": scalar_price})
                # Advance RNG state to match the paper's deterministic two-layer pair sampling sequence.
                if settings.get("pair_rng_stride", 0):
                    rng.choice(specs)
                    rng.choice(specs)
                    rng.choice(range(settings["pair_rng_stride"]))
                    rng.choice(range(settings["pair_rng_stride"]))
    results["alignment"] = {}
    for task, price_key in (("task1_weight_trace", "realized_price"), ("task2_weight_iso", "trace_price")):
        pairs = [{"theory_value": row[price_key], "empirical_kl": row["empirical_kl"]}
                 for row in results["layer_pairs"]]
        results["alignment"][task] = {
            "single_layer": alignment(results[task]),
            "two_layer": alignment(pairs),
            "pooled_main_figure": alignment(results[task] + pairs),
        }
    return results


def build_parser() -> argparse.ArgumentParser:
    """Return the main-paper Price–KL CLI with explicit small-run overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("opt-125m", "qwen3-0.6b"), required=True)
    parser.add_argument("--model-path", help="Local checkpoint or alternate HF ID.")
    parser.add_argument("--revision")
    parser.add_argument("--pair-seed-name", help="Optional model-name string for deterministic pair-sampling seeding.")
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--max-seqlen", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-layer-pairs", type=int)
    parser.add_argument("--num-hessian-probes", type=int)
    parser.add_argument("--max-layers", type=int)
    parser.add_argument("--layer-pattern")
    parser.add_argument("--weight-quant-types", nargs="+")
    parser.add_argument("--weight-granularities", nargs="+")
    return parser


def main() -> None:
    """Run Price–KL measurements and write raw records, correlations, and timing."""
    args = build_parser().parse_args()
    config = read_config("price_kl.json")
    settings = dict(config["protocol"])
    source = args.model_path or config["models"][args.model]
    # Use the public model ID for deterministic pair sampling, independent of cache locations.
    settings["model_id"] = config["models"][args.model]
    settings["pair_seed_name"] = args.pair_seed_name or settings["model_id"]
    for key in ("num_samples", "max_seqlen", "batch_size", "num_layer_pairs", "num_hessian_probes",
                "weight_quant_types", "weight_granularities"):
        value = getattr(args, key)
        if value is not None:
            settings[key] = value
    if min(settings[k] for k in ("num_samples", "max_seqlen", "batch_size", "num_hessian_probes")) < 1:
        raise ValueError("Sample, sequence, batch, and probe counts must be positive.")
    if settings["num_layer_pairs"] < 0:
        raise ValueError("The number of layer pairs cannot be negative.")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    seed_everything(settings["seed"])
    output = prepare_output(args.output)
    timer = Timer(device)
    with timer.stage("model_and_tokenizer_load", selection=False):
        tokenizer = AutoTokenizer.from_pretrained(source, revision=args.revision)
        model = AutoModelForCausalLM.from_pretrained(source, revision=args.revision, torch_dtype=torch.float32,
                                                    attn_implementation="eager").to(device).eval()
        model.config.use_cache = False
    with timer.stage("calibration_data_build", selection=False):
        batches = build_calibration_batches(tokenizer, dataset_name=args.dataset, num_samples=settings["num_samples"],
                max_seqlen=settings["max_seqlen"], batch_size=settings["batch_size"], seed=settings["seed"])
        layers = get_quantizable_linear_layers(model, layer_pattern=args.layer_pattern, max_layers=args.max_layers)
        if not layers:
            raise ValueError("The requested layer filter selected no linear layers.")
    results = run_model(model, batches, layers, settings, device, timer)
    timing = timer.report()
    timing["measurement_seconds"] = timing.pop("selection_seconds")
    timing["measurement_stages"] = timing.pop("selection_stages")
    results["metadata"] = {"protocol": settings, "model_source": source, "revision": args.revision,
            "resolved_revision": getattr(model.config, "_commit_hash", None),
            "calibration_token_sha256": token_fingerprint(batches), "layers": list(layers),
            "environment": environment(device), "timing": timing}
    write_json(output/"price_kl_results.json", results)
    fields = ("task_id", "model_name", "layer_name", "constraint", "weight_quant_type", "weight_granularity", "theory_value", "empirical_kl")
    with (output/"price_kl.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results["task1_weight_trace"] + results["task2_weight_iso"])
    print(f"Saved Price–KL records and timing to {output}")


if __name__ == "__main__":
    main()
