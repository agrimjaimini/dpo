# Direct Preference Optimization (DPO) Training

A complete, end-to-end implementation of Direct Preference Optimization (DPO) for training language models on the Anthropic Helpful-Harmless (HH) dataset.

This implementation is based on the paper: **"Direct Preference Optimization: Your Language Model is Secretly a Reward Model"** by Rafailov et al.

## Overview

Direct Preference Optimization (DPO) is a method for training language models directly from human preference data without explicit reward modeling or reinforcement learning. Instead of the standard RLHF pipeline (SFT → Reward Modeling → PPO), DPO:

1. Trains a supervised fine-tuned (SFT) model on preferred responses
2. Directly optimizes the policy using a simple classification loss on preference pairs
3. Implicitly learns a reward function through the log probability ratio between policy and reference models

**Key advantages:**
- Simpler than PPO-based RLHF (no reward model, no on-policy sampling)
- More stable training (no actor-critic dynamics)
- Same theoretical guarantees as KL-regularized RLHF

## Features

- **From-scratch implementation** of DPO loss and training loop
- **GPT-2 small** (124M parameters) as base model
- **Anthropic HH** preference dataset from HuggingFace
- **SFT baseline** for comparison
- **Comprehensive evaluation** (preference accuracy, reward margin, perplexity)
- **Modular design** with clear separation of concerns
- **Extensive configuration** via YAML files and CLI arguments
- **Debug mode** for fast iteration
- **Sanity checks** to verify correctness

## Installation

```bash
# Clone the repository
cd dpo

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.35+
- CUDA-capable GPU (recommended, but CPU works too)

## Training Options

### Option 1: Local Training (Mac/Linux/Windows)

Train a DPO model in 3 commands:

```bash
# 1. Run sanity checks
python tests/test_sanity.py

# 2. Train SFT baseline (creates initial policy)
python scripts/train_sft.py --config configs/sft.yaml

# 3. Train DPO (optimizes for preferences)
python scripts/train_dpo.py --config configs/dpo.yaml --sft_model_path outputs/sft/final

# 4. Evaluate all models
python scripts/evaluate.py \
    --reference_model gpt2 \
    --sft_model outputs/sft/final \
    --dpo_model outputs/dpo/final \
    --output_file results.json
```

### Option 2: Google Colab (Free GPU)

Train on Google Colab's free GPUs (no local GPU needed):

**Method A: Clone from GitHub (Recommended)**

1. **Push your code to GitHub:**
   ```bash
   cd /Users/agrimjaimini/Desktop/CS/Projects/dpo
   git add .
   git commit -m "Add DPO training code"
   git push
   ```

2. **Open Colab:**
   - Go to https://colab.research.google.com/
   - Upload `DPO_Training_Colab.ipynb`
   - Runtime → Change runtime type → GPU → Save

3. **Update the clone command:**
   - In the notebook, replace `YOUR_USERNAME` with your GitHub username
   - Run the GitHub clone cell (Option B)
   - Skip the upload cell (Option A)

4. **Run all cells:**
   - Training will use Colab's free T4 GPU

**Method B: Upload ZIP**

1. **Package your code:**
   ```bash
   cd /Users/agrimjaimini/Desktop/CS/Projects/dpo
   ./scripts/package_for_colab.sh
   ```

2. **Open Colab and upload:**
   - Upload `DPO_Training_Colab.ipynb`
   - Set Runtime → GPU
   - Run Option A cell and upload `dpo.zip`

**Benefits:**
- ✅ Free GPU access (T4 GPU)
- ✅ No local setup required
- ✅ ~12 hour sessions (free tier)
- ✅ Download trained models when done

**Expected times on Colab T4:**
- Debug mode: ~10 min (SFT) + ~10 min (DPO)
- Small training (1k samples): ~1 hour total
- Full training: ~8-12 hours total

**Tips:**
- Save checkpoints to Google Drive to survive disconnects
- Use debug mode first to verify everything works
- Colab Pro gives better GPUs and longer sessions

## Project Structure

```
dpo/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── configs/                       # Configuration files
│   ├── sft.yaml                  # SFT training config
│   ├── dpo.yaml                  # DPO training config
│   └── debug.yaml                # Debug/sanity check config
├── src/                          # Source code
│   ├── data/                     # Data loading and preprocessing
│   ├── models/                   # Model wrappers
│   ├── trainers/                 # Training loops
│   ├── losses/                   # DPO loss implementation
│   ├── evaluation/               # Evaluation metrics
│   └── utils/                    # Configuration and logging
├── scripts/                      # Executable scripts
│   ├── train_sft.py             # Train SFT baseline
│   ├── train_dpo.py             # Train DPO
│   └── evaluate.py              # Evaluate models
└── tests/                        # Tests
    └── test_sanity.py           # Sanity checks
```

## Usage

### 1. Training SFT Baseline

The SFT model is trained on chosen responses only using standard language modeling loss. This creates the initial policy before DPO training.

```bash
# Standard training
python scripts/train_sft.py --config configs/sft.yaml

# Debug mode (small dataset, fast iteration)
python scripts/train_sft.py --config configs/debug.yaml --debug

# Override hyperparameters
python scripts/train_sft.py \
    --config configs/sft.yaml \
    --learning_rate 1e-4 \
    --num_epochs 2 \
    --output_dir outputs/sft_custom
```

**Outputs:**
- Model checkpoints in `outputs/sft/checkpoint-{step}/`
- Final model in `outputs/sft/final/`
- Training logs in `outputs/sft/train.log`
- Config saved to `outputs/sft/config.yaml`

### 2. Training DPO

DPO training requires a trained SFT model as the starting point. The reference model is a frozen copy of the SFT model.

```bash
# Standard DPO training
python scripts/train_dpo.py \
    --config configs/dpo.yaml \
    --sft_model_path outputs/sft/final

# Tune beta (controls KL penalty strength)
python scripts/train_dpo.py \
    --config configs/dpo.yaml \
    --sft_model_path outputs/sft/final \
    --beta 0.05

# Debug mode
python scripts/train_dpo.py \
    --config configs/debug.yaml \
    --sft_model_path outputs/debug/final \
    --debug
```

**Key hyperparameters:**
- `beta`: Temperature controlling deviation from reference (default: 0.1)
  - Higher beta = stay closer to reference
  - Lower beta = optimize preference signal more aggressively
- `loss_type`: DPO loss variant (sigmoid, ipo, hinge)
- `learning_rate`: Typically 10-100x smaller than SFT (default: 5e-6)

**Outputs:**
- Model checkpoints in `outputs/dpo/checkpoint-{step}/`
- Final model in `outputs/dpo/final/`
- Training logs with DPO-specific metrics (accuracy, reward margin)

### 3. Evaluation

Compare reference, SFT, and DPO models on held-out test set.

```bash
python scripts/evaluate.py \
    --reference_model gpt2 \
    --sft_model outputs/sft/final \
    --dpo_model outputs/dpo/final \
    --output_file results.json
```

**Metrics computed:**
- **Preference Accuracy**: Fraction of test examples where model assigns higher reward to chosen response
- **Reward Margin**: Average difference in implicit reward between chosen and rejected
- **Perplexity**: Language modeling quality on chosen responses
- **KL Divergence**: Deviation from reference model

**Outputs:**
- Quantitative results in `results.json`
- Qualitative generation samples in `generation_samples.json`
- Markdown comparison table printed to console

### 4. Sanity Checks

Run comprehensive tests to verify the implementation:

```bash
python tests/test_sanity.py
```

**Tests:**
1. Data loading and preprocessing
2. Tokenization and batching
3. Model forward pass
4. DPO loss computation
5. Full training step with gradients
6. Model checkpointing

## Configuration

### YAML Configuration Files

All training parameters can be set via YAML config files:

```yaml
# configs/dpo.yaml
model:
  model_name_or_path: "gpt2"
  use_gradient_checkpointing: true
  torch_dtype: "float16"

data:
  dataset_name: "Anthropic/hh-rlhf"
  max_length: 512
  max_prompt_length: 256

training:
  learning_rate: 5.0e-6
  num_epochs: 1
  per_device_batch_size: 4
  gradient_accumulation_steps: 8

beta: 0.1
loss_type: "sigmoid"
```

### CLI Overrides

Override config values via command line:

```bash
python scripts/train_dpo.py \
    --config configs/dpo.yaml \
    --sft_model_path outputs/sft/final \
    --beta 0.2 \
    --learning_rate 1e-5 \
    --num_epochs 2
```

## DPO Algorithm Details

### The DPO Loss

The DPO loss is:

```
L_DPO = -E[log σ(β * (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]
```

Where:
- `π_θ` = policy model (being trained)
- `π_ref` = reference model (frozen)
- `y_w` = chosen (winning) response
- `y_l` = rejected (losing) response
- `β` = temperature parameter
- `σ` = sigmoid function

### Implicit Reward

The DPO objective implicitly defines a reward function:

```
r(x, y) = β * log(π_θ(y|x) / π_ref(y|x))
```

This reward is optimized such that chosen responses get higher rewards than rejected ones.

### Relationship to RLHF

DPO is theoretically equivalent to KL-regularized RLHF:

```
max E[r(x, y)] - β * KL(π_θ || π_ref)
```

But instead of:
1. Learning explicit reward model r_φ
2. Running PPO with KL penalty

DPO directly optimizes the policy via a simple classification loss on preference pairs.

## Expected Results

On the Anthropic HH test set, you should see:

| Model | Preference Accuracy | Reward Margin | Perplexity |
|-------|-------------------|---------------|------------|
| Reference (GPT-2) | ~50% | ~0.0 | ~30-40 |
| SFT | ~60-70% | ~0.5-1.0 | ~20-25 |
| DPO | ~70-80% | ~1.0-2.0 | ~25-30 |

**Notes:**
- DPO should outperform SFT on preference accuracy
- DPO may have slightly higher perplexity than SFT (trades off LM quality for alignment)
- Reward margin should increase significantly with DPO training
- Results vary based on hyperparameters and training duration

## Troubleshooting

### Out of Memory (OOM)

- Reduce `per_device_batch_size` to 2 or 1
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Use `fp16: true` for mixed precision
- Enable `use_gradient_checkpointing: true`

### Loss Not Decreasing

- Check learning rate (DPO uses much smaller LR than SFT: ~5e-6 vs 2e-4)
- Verify SFT model path is correct
- Try increasing beta (encourages staying closer to reference)
- Check that reference model is actually frozen

### Preference Accuracy Stuck at 50%

- This means the model isn't learning to distinguish chosen from rejected
- Try smaller learning rate
- Increase training steps
- Check that data preprocessing is correct (run sanity tests)

### Import Errors

Make sure you're running scripts from the project root:
```bash
# Correct
python scripts/train_sft.py --config configs/sft.yaml

# Wrong (will cause import errors)
cd scripts
python train_sft.py --config ../configs/sft.yaml
```

## Mac / Apple Silicon Optimization

This implementation automatically detects and uses **Metal Performance Shaders (MPS)** for GPU acceleration on Apple Silicon Macs (M1, M2, M3, etc.).

### Recommended Settings for Mac

For optimal performance on Mac:

```yaml
# configs/sft.yaml or configs/dpo.yaml
model:
  torch_dtype: "float16"  # Much faster on MPS

training:
  fp16: true  # Enable mixed precision
  per_device_batch_size: 4  # Adjust based on your Mac's memory
```

### MPS Notes

- **Automatic Detection**: The code automatically uses MPS if available
- **Performance**: MPS provides significant speedup vs CPU (3-5x typical)
- **Memory**: Monitor memory usage with Activity Monitor
- **Fallback**: Some operations may fall back to CPU if not MPS-compatible

### Troubleshooting MPS

If you encounter MPS-related errors:

```yaml
# Fallback to CPU if needed
training:
  fp16: false

model:
  torch_dtype: "float32"
```

Or set the environment variable:
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## Advanced Usage

### Using Different Base Models

Replace GPT-2 with any HuggingFace causal LM:

```yaml
# configs/sft.yaml
model:
  model_name_or_path: "EleutherAI/pythia-160m"  # or "facebook/opt-125m", etc.
```

### Logging to Weights & Biases

```yaml
# configs/sft.yaml
logging:
  use_wandb: true
  wandb_project: "my-dpo-project"
  wandb_run_name: "sft-baseline"
```

### Training on Custom Preference Data

Modify `src/data/preprocessing.py` to load your custom dataset. The format should be:

```json
{
  "prompt": "Human: Your question here",
  "chosen": "Good response",
  "rejected": "Bad response"
}
```

## References

### Paper
```bibtex
@article{rafailov2023direct,
  title={Direct Preference Optimization: Your Language Model is Secretly a Reward Model},
  author={Rafailov, Rafael and Sharma, Archit and Mitchell, Eric and Ermon, Stefano and Manning, Christopher D and Finn, Chelsea},
  journal={arXiv preprint arXiv:2305.18290},
  year={2023}
}
```

### Dataset
```bibtex
@article{bai2022training,
  title={Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback},
  author={Bai, Yuntao and others},
  journal={arXiv preprint arXiv:2204.05862},
  year={2022}
}
```

## License

MIT License - feel free to use this for research or production.

## Contributing

This is an educational implementation. Contributions welcome:
- Bug fixes
- Additional evaluation metrics
- Support for other datasets
- Documentation improvements

## Contact

For questions or issues, please open a GitHub issue.
