"""Configuration management using Pydantic."""

import yaml
from pathlib import Path
from typing import Optional, Literal, Dict, Any
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Model configuration."""
    model_name_or_path: str = "gpt2"
    use_gradient_checkpointing: bool = True
    torch_dtype: str = "float16"


class DataConfig(BaseModel):
    """Data configuration."""
    dataset_name: str = "Anthropic/hh-rlhf"
    max_length: int = 512
    max_prompt_length: int = 256
    train_split: str = "train"
    val_ratio: float = 0.05
    test_ratio: float = 0.05
    num_proc: int = 4  # Number of processes for data preprocessing


class TrainingConfig(BaseModel):
    """Base training configuration."""
    learning_rate: float = 2e-4
    num_epochs: int = 1
    per_device_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    fp16: bool = True
    logging_steps: int = 10
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 3
    seed: int = 42


class SFTConfig(BaseModel):
    """SFT-specific configuration."""
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    output_dir: str = "outputs/sft"
    logging: Dict[str, Any] = Field(default_factory=lambda: {
        "log_level": "info",
        "use_wandb": False,
        "wandb_project": "dpo-hh",
        "wandb_run_name": None,
    })


class DPOTrainingConfig(TrainingConfig):
    """DPO-specific training configuration."""
    learning_rate: float = 5e-6  # Much smaller than SFT
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 8


class DPOConfig(BaseModel):
    """DPO configuration."""
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    training: DPOTrainingConfig = Field(default_factory=DPOTrainingConfig)
    beta: float = 0.1
    loss_type: Literal["sigmoid", "ipo", "hinge"] = "sigmoid"
    label_smoothing: float = 0.0
    output_dir: str = "outputs/dpo"
    logging: Dict[str, Any] = Field(default_factory=lambda: {
        "log_level": "info",
        "use_wandb": False,
        "wandb_project": "dpo-hh",
        "wandb_run_name": None,
    })


class DebugConfig(BaseModel):
    """Debug mode configuration."""
    num_train_samples: int = 100
    num_eval_samples: int = 20
    num_epochs: int = 2
    eval_steps: int = 10
    logging_steps: int = 1


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""
    batch_size: int = 32
    num_samples: Optional[int] = None  # None = evaluate on full test set
    num_comparison_samples: int = 10  # Number of qualitative samples to generate


def load_config(config_path: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file with optional CLI overrides.

    Args:
        config_path: Path to YAML config file
        overrides: Dictionary of override values (typically from argparse)

    Returns:
        Dictionary with loaded and validated configuration
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load YAML
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Apply overrides if provided
    if overrides:
        # Remove None values from overrides
        overrides = {k: v for k, v in overrides.items() if v is not None}

        # Recursively update nested dicts
        config_dict = _deep_update(config_dict, overrides)

    return config_dict


def _deep_update(base_dict: Dict, update_dict: Dict) -> Dict:
    """
    Recursively update nested dictionaries.

    Args:
        base_dict: Base dictionary
        update_dict: Updates to apply

    Returns:
        Updated dictionary
    """
    result = base_dict.copy()

    for key, value in update_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value

    return result


def save_config(config: BaseModel, path: str):
    """
    Save configuration to YAML file.

    Args:
        config: Pydantic configuration model
        path: Path to save config
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False, sort_keys=False)
