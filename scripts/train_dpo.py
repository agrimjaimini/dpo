"""Train DPO on Anthropic HH preference data."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
from transformers import AutoTokenizer
from datasets import load_dataset

from src.utils.config import DPOConfig, load_config, save_config
from src.utils.logging_utils import setup_logging
from src.data.preprocessing import preprocess_hh_dataset, create_splits, debug_sample
from src.data.dataset import PreferenceDataset
from src.data.collator import PreferenceDataCollator
from src.models.policy import PolicyModel
from src.models.reference import ReferenceModel
from src.trainers.dpo_trainer import DPOTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train DPO")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--sft_model_path", type=str, required=True, help="Path to SFT model checkpoint")
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    parser.add_argument("--beta", type=float, help="Override DPO beta parameter")
    parser.add_argument("--learning_rate", type=float, help="Override learning rate")
    parser.add_argument("--num_epochs", type=int, help="Override number of epochs")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with small dataset")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    config_dict = load_config(args.config)

    # Apply CLI overrides
    if args.output_dir:
        config_dict["output_dir"] = args.output_dir
    if args.beta:
        config_dict["beta"] = args.beta
    if args.learning_rate:
        config_dict["training"]["learning_rate"] = args.learning_rate
    if args.num_epochs:
        config_dict["training"]["num_epochs"] = args.num_epochs

    # Validate config
    config = DPOConfig(**config_dict)

    # Setup logging
    logger = setup_logging(
        log_level=config.logging.get("log_level", "info"),
        log_file=f"{config.output_dir}/train.log",
        name="train_dpo",
    )

    logger.info("="*80)
    logger.info("DPO Training")
    logger.info("="*80)
    logger.info(f"SFT model: {args.sft_model_path}")
    logger.info(f"Beta: {config.beta}")
    logger.info(f"Loss type: {config.loss_type}")

    # Save config to output dir
    save_config(config, f"{config.output_dir}/config.yaml")

    # Load tokenizer (from SFT model)
    logger.info(f"Loading tokenizer from: {args.sft_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and preprocess data
    logger.info(f"Loading dataset: {config.data.dataset_name}")
    raw_dataset = load_dataset(config.data.dataset_name)

    # Preprocess to extract preference triples
    logger.info("Preprocessing dataset...")
    processed_dataset = preprocess_hh_dataset(
        raw_dataset[config.data.train_split],
        num_proc=config.data.num_proc,
    )

    # Debug mode: use small subset
    if args.debug:
        logger.info("DEBUG MODE: Using small dataset")
        processed_dataset = processed_dataset.select(range(min(100, len(processed_dataset))))

    # Create splits
    logger.info("Creating train/val/test splits...")
    train_dataset, val_dataset, test_dataset = create_splits(
        processed_dataset,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
    )

    logger.info(f"Train size: {len(train_dataset)}")
    logger.info(f"Val size: {len(val_dataset)}")
    logger.info(f"Test size: {len(test_dataset)}")

    # Show sample
    debug_sample(train_dataset, num_samples=2)

    # Create PyTorch datasets
    logger.info("Creating PyTorch datasets...")
    train_dataset = PreferenceDataset(
        train_dataset,
        tokenizer,
        max_length=config.data.max_length,
        max_prompt_length=config.data.max_prompt_length,
    )

    val_dataset = PreferenceDataset(
        val_dataset,
        tokenizer,
        max_length=config.data.max_length,
        max_prompt_length=config.data.max_prompt_length,
    )

    # Create data collator
    collator = PreferenceDataCollator(pad_token_id=tokenizer.pad_token_id)

    # Load policy model (from SFT checkpoint)
    logger.info(f"Loading policy model from: {args.sft_model_path}")
    policy_model = PolicyModel(
        model_name_or_path=args.sft_model_path,
        use_gradient_checkpointing=config.model.use_gradient_checkpointing,
        torch_dtype=config.model.torch_dtype,
    )

    # Create reference model (frozen copy of SFT model)
    logger.info("Creating reference model (frozen copy)...")
    reference_model = ReferenceModel(policy_model)

    # Create trainer
    logger.info("Creating DPO trainer...")
    trainer = DPOTrainer(
        policy_model=policy_model,
        reference_model=reference_model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        collate_fn=collator,
        config=config.model_dump(),
        output_dir=config.output_dir,
    )

    # Train
    logger.info("Starting DPO training...")
    trainer.train()

    # Save tokenizer
    tokenizer.save_pretrained(f"{config.output_dir}/final")

    logger.info("Training completed!")
    logger.info(f"Model saved to {config.output_dir}/final")


if __name__ == "__main__":
    main()
