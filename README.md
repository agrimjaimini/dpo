# DPO training on Anthropic HH

This repo trains a policy with Direct Preference Optimization (DPO) on the Anthropic Helpful–Harmless dataset. It includes a supervised fine-tune (SFT) stage, DPO training, and simple evaluation.

## What’s here
- Configs for SFT/DPO (defaults to `mistralai/Mistral-7B-Instruct-v0.3`; swap to a smaller model if you need speed)
- Training scripts (`scripts/train_sft.py`, `scripts/train_dpo.py`)
- Evaluation script (`scripts/evaluate.py`) for preference accuracy/reward margin/perplexity and sample generations
- Minimal sanity checks (`tests/test_sanity.py`)

## Project structure
```
configs/        YAML configs (sft, dpo, debug)
scripts/        train_sft.py, train_dpo.py, evaluate.py
src/            data prep, models, losses, trainers, evaluation utils
tests/          sanity test
```

## Sanity check
```bash
python3 tests/test_sanity.py
```
