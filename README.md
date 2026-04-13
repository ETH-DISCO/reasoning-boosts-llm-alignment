<div align="center">

# Reasoning Boosts Opinion Alignment in LLMs

**Frédéric Berdoz · Yann Billeter · Yann Vonlanthen · Roger Wattenhofer**
<p><sub>(Authors in alphabetical order, see manuscript for contribution details)</sub></p>

[![arXiv](https://img.shields.io/badge/arXiv-2603.01214-b31b1b.svg)](https://arxiv.org/abs/2603.01214)
[![Paper](https://img.shields.io/badge/Paper-OpenReview-blue)](https://openreview.net/forum?id=mdunHhVDPz)
[![Data](https://img.shields.io/badge/Data-HuggingFace-yellow)](https://huggingface.co/datasets/disco-eth/Reasoning-Boosts-Opinion-Alignment-in-LLMs)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://fberdoz.github.io/projects/reasoning-boosts-alignment/)

Accepted at ICLR 2026 🎉

Opinion modeling aims to capture individual or group political preferences, enabling applications such as digital democracies, where models could help shape fairer and more popular policies. Given their versatility, strong generalization capabilities, and demonstrated success across diverse text-to-text applications, large language models (LLMs) are natural candidates for this task. However, due to their statistical nature and limited causal understanding, they tend to produce biased opinions when prompted naively. In this work, we study whether reasoning can improve opinion alignment. Motivated by the recent advancement in mathematical reasoning enabled by reinforcement learning (RL), we train models to produce profile-consistent answers through structured reasoning. We evaluate our approach on three datasets covering U.S., European, and Swiss politics. Results indicate that reasoning enhances opinion modeling and is competitive with strong baselines, but does not fully remove bias, highlighting the need for additional mechanisms to build faithful political digital twins using LLMs. By releasing both our method and datasets, we establish a solid baseline to support future research on LLM opinion alignment.

</div>

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
    booktitle={The Thirteenth International Conference on Learning Representations (ICLR)},
    year={2026}
}
```
