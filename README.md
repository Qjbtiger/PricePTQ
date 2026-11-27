<div align="center">

<h1>PricePTQ</h1>

<h3>Predicting Quantization Price for Selecting PTQ Configurations Before Deployment</h3>


<p>
  <a href="https://neurips.cc/virtual/2026/loc/sydney/poster/153413"><img src="https://img.shields.io/badge/Paper-NeurIPS%202026-blue" alt="NeurIPS 2026"></a>
</p>

</div>

We formulate post-training quantization as configuration selection before deployment.
PricePTQ uses downstream curvature from the full-precision model to price the layer-output errors induced by candidate quantization configurations.
These calibration-time prices provide a common criterion for bit allocation, transformation selection, and granularity allocation under deployment constraints.

## Setup

Use Python 3.12 and a compatible GPU.

```bash
uv sync
source .venv/bin/activate
```

For the AMQ baseline, install the additional dependencies with `uv sync --extra amq`.

## Experiments

### Price–KL Alignment

Compare predicted quantization prices with measured forward KL on OPT-125M and Qwen3-0.6B.

```bash
bash scripts/price_kl.sh --model opt-125m --output outputs/opt-price-kl
bash scripts/price_kl.sh --model qwen3-0.6b --output outputs/qwen-price-kl
```

### Bit Allocation

Compare Ours, HIGGS, and AMQ for layer-wise bit allocation.

```bash
bash scripts/bit_allocation.sh --model llama-1b --method ours --output outputs/1b-bits-ours
bash scripts/bit_allocation.sh --model llama-1b --method higgs --output outputs/1b-bits-higgs
bash scripts/bit_allocation.sh --model llama-1b --method amq --output outputs/1b-bits-amq
```

AMQ uses HQQ proxies during search and GPTQ for the optional model export.

### Transformation Selection

Compare Ours, CALM-CKA, and random selection across five transformations.

```bash
bash scripts/transformation_selection.sh --model llama-1b --method ours --output outputs/1b-transform-ours
bash scripts/transformation_selection.sh --model llama-1b --method calm-cka --output outputs/1b-transform-cka
bash scripts/transformation_selection.sh --model llama-1b --method random --output outputs/1b-transform-random0
```

### Granularity Allocation

Compare Ours, fixed group-128, and random selection across six granularities.

```bash
bash scripts/granularity_allocation.sh --model llama-1b --method ours --output outputs/1b-granularity-ours
bash scripts/granularity_allocation.sh --model llama-1b --method fixed-group128 --output outputs/1b-granularity-fixed
bash scripts/granularity_allocation.sh --model llama-1b --method random --output outputs/1b-granularity-random0
```

## Citation

```bibtex
@inproceedings{qiu2026priceptq,
  title = {Predicting Quantization Price for Selecting PTQ Configurations Before Deployment},
  author = {Qiu, Junbin and Mu, Jian and Zhang, Weitong and Shu, Yao},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2026}
}
```
