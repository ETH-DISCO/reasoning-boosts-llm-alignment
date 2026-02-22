# Reasoning Boosts Opinion Alignment in LLMs

**Frédéric Berdoz, Yann Billeter, Yann Vonlanthen, Roger Wattenhofer**
*(Authors in alphabetical order)*

[![Paper](https://img.shields.io/badge/Paper-OpenReview-blue)](https://openreview.net/forum?id=mdunHhVDPz)

> **ICLR 2026 Poster** | January 2026

Official code repository for the paper *"Reasoning Boosts Opinion Alignment in LLMs"*.

## Abstract

Using reasoning and reinforcement learning, we improve LLM opinion modeling, highlight remaining biases, and provide a baseline for future research. This research investigates whether structured reasoning enhances LLM opinion modeling through reinforcement learning. We evaluate our approach using datasets covering U.S., European, and Swiss politics. While reasoning improves opinion alignment competitively, residual biases persist, suggesting the necessity of additional mechanisms for building faithful political digital twins using LLMs.

## Project Structure

- `src/reasoning/main.py` - Main training script for GRPO-based reasoning methods
- `src/reasoning/baseline.py` - Baseline comparison methods (SFT, DPO, In-Context Learning)
- `src/reasoning/rewards/` - Reward function implementations
- `config/` - Configuration files

## Installation

This project requires Python 3.12.

```bash
# Install the package and dependencies
pip install -e .
```

## Usage

### Main Training (GRPO)

The main training script supports supervised fine-tuning (SFT) followed by Group Relative Policy Optimization (GRPO).

```bash
# Run training - expects params.yaml in current directory
reasoning

# Or use Python module directly
python -m reasoning.main
```

The script expects a `params.yaml` file with configuration sections for model, dataset, trainer, and optional SFT settings.

### Baseline Methods

Run baseline comparison methods:

```bash
# Available methods: sft, dpo, sft_dpo, in_context_learning
reasoning-baseline --method sft --config params.yaml
reasoning-baseline --method dpo --config params.yaml
reasoning-baseline --method in_context_learning --config params.yaml

# Optionally override dataset splits
reasoning-baseline --method sft --train-split train --test-split test

# Or use Python module directly
python -m reasoning.baseline --method sft --config params.yaml
```

## Configuration

Configuration is managed through YAML files using OmegaConf. See `config/` for examples.

### Evaluation Settings

The test evaluation behavior can be configured in `params.yaml` under the `reasoning.evaluation` section:

```yaml
reasoning:
  evaluation:
    num_repeats: 8          # Number of generations per test sample (default: 8)
    temperature: 0.9        # Sampling temperature (default: 0.9)
    max_new_tokens: 512     # Maximum tokens to generate (default: 512)
```

### Baseline Settings

For in-context learning baselines, configure in `params.yaml` under `reasoning.baseline`:

```yaml
reasoning:
  baseline:
    num_repeats: 8             # Generations per sample (default: 8)
```

## Outputs

Training creates:
- `sft_checkpoint/` - Model checkpoint after SFT (if enabled)
- `grpo_saved_lora/` - Final model checkpoint after GRPO
- `test_results.json` / `test_results.csv` - Test evaluation results
- `test_summary.json` - Aggregate test metrics

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{
    berdoz2026reasoning,
    title={Reasoning Boosts Opinion Alignment in {LLM}s},
    author={Fr{\'e}d{\'e}ric Berdoz and Yann Billeter and Yann Vonlanthen and Roger Wattenhofer},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=mdunHhVDPz}
}
```
