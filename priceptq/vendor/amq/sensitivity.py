# Adapted for the PricePTQ paper release.
"""AMQ search-space sensitivity analysis (Apache-2.0)."""

import os

import time

import json

from priceptq.vendor.amq.evaluation.evaluator import Evaluator

from priceptq.vendor.amq.utils.func import init_accelerator, set_seed

from tqdm import tqdm


def linear_sensitivity(args):
    """Run linear sensitivity analysis for space pruning."""
    with open(args.config, 'r') as f:
        config = json.load(f)[args.model_name]
    
    accelerator, device_map = init_accelerator(args.gpu_id, config)
    accelerator.print("Running Linear Sensitivity Analysis")
    accelerator.print(args)

    set_seed(args.seed)
    
    bits_range = list(getattr(args, 'bits_range', [2, 3, 4]))
    if not bits_range:
        raise ValueError('bits_range must contain at least one candidate bit-width')
    if len(bits_range) != len(args.quantization_proxy_paths):
        raise ValueError(
            'bits_range and quantization_proxy_paths must have the same length: '
            f'{len(bits_range)} != {len(args.quantization_proxy_paths)}'
        )
    min_bit = min(bits_range)
    max_bit = max(bits_range)

    evaluator = Evaluator(
        config=config,
        accelerator=accelerator,
        device_map=device_map,
        model_id=args.model_id,
        bits_range=bits_range,
        quantization_proxy_paths=args.quantization_proxy_paths,
        seqlen=args.seqlen,
        n_sample=args.n_sample,
        datasets=[args.dataset]
    )
    
    n_block = config['n_block']
    linears = config['linear']
    loss_list = {}
    
    # Initialize architecture with maximum bit-width for all layers
    architecture = {'linear': {linear_group: [max_bit] * n_block for linear_group in linears}}

    start_time = time.time()
    
    # Measure sensitivity for each linear group in each block
    for block_idx in tqdm(range(n_block), desc="Measuring sensitivity"):
        for linear_group in linears:
            iter_start_time = time.time()
            
            key = f'{block_idx}.{linear_group}'

            architecture['linear'][linear_group][block_idx] = min_bit
            
            loss, _ = evaluator.eval(accelerator=accelerator, architecture=architecture)
            loss_list[key] = loss[args.dataset]
            
            iter_time = time.time() - iter_start_time
            accelerator.print(f"[{key}] Loss={loss_list[key]:.4f}, time: {iter_time:.2f}")
            
            architecture['linear'][linear_group][block_idx] = max_bit

    results = {
        "model_name": args.model_name,
        "dataset": args.dataset,
        "bits_range": bits_range,
        "baseline_bit": max_bit,
        "perturbation_bit": min_bit,
        "loss": loss_list,
        "time_elapsed": time.time() - start_time
    }
    
    with accelerator.main_process_first():
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        accelerator.print(f"Results saved to {args.output_file}")

    accelerator.print(f"Time Elapsed: {time.time() - start_time:.2f} seconds")
    accelerator.print("Linear sensitivity analysis completed")
    
    return results

