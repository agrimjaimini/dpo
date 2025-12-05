"""Evaluate and compare models."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset

from src.utils.logging_utils import setup_logging
from src.utils.device import get_device_name
from src.data.preprocessing import preprocess_hh_dataset, create_splits
from src.data.dataset import PreferenceDataset
from src.data.collator import PreferenceDataCollator
from src.models.policy import PolicyModel
from src.models.reference import ReferenceModel
from src.evaluation.evaluator import Evaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate models")
    parser.add_argument("--reference_model", type=str, required=True,
                       help="HuggingFace model name or path for base reference model")
    parser.add_argument("--sft_model", type=str, help="Path to SFT model")
    parser.add_argument("--dpo_model", type=str, help="Path to DPO model")
    parser.add_argument("--output_file", type=str, default="results.json",
                       help="Path to save results JSON")
    parser.add_argument("--num_samples", type=int, help="Number of test samples to use (None = all)")
    parser.add_argument("--num_generation_samples", type=int, default=5,
                       help="Number of qualitative generation samples")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logging
    logger = setup_logging(log_level="info", name="evaluate")

    logger.info("="*80)
    logger.info("Model Evaluation")
    logger.info("="*80)

    # Setup device (prioritize MPS for Mac, then CUDA, then CPU)
    device = get_device_name()
    logger.info(f"Using device: {device}")

    # Load models
    models = {}

    logger.info(f"\nLoading reference model: {args.reference_model}")
    ref_model_for_eval = PolicyModel(args.reference_model, torch_dtype="float16")
    models["reference"] = ref_model_for_eval

    # Use reference model as the reference for DPO metrics
    reference_model = ReferenceModel(ref_model_for_eval)

    if args.sft_model:
        logger.info(f"Loading SFT model: {args.sft_model}")
        models["sft"] = PolicyModel(args.sft_model, torch_dtype="float16")

    if args.dpo_model:
        logger.info(f"Loading DPO model: {args.dpo_model}")
        models["dpo"] = PolicyModel(args.dpo_model, torch_dtype="float16")

    # Choose tokenizer source: prefer SFT, then DPO, then reference.
    tokenizer_path = args.sft_model or args.dpo_model or args.reference_model
    logger.info(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Sanity-check vocab alignment between tokenizer and models to avoid CUDA asserts
    tokenizer_vocab_size = len(tokenizer)
    for name, model in models.items():
        model_vocab_size = getattr(model.model.config, "vocab_size", None)
        if model_vocab_size and model_vocab_size != tokenizer_vocab_size:
            raise ValueError(
                f"Tokenizer vocab size ({tokenizer_vocab_size}) does not match model '{name}' "
                f"vocab size ({model_vocab_size}). Use matching model/tokenizer family "
                f"(e.g., keep reference/SFT/DPO all on Mistral or all on GPT2)."
            )

    # Load test data
    logger.info("\nLoading test dataset...")
    raw_dataset = load_dataset("Anthropic/hh-rlhf")
    processed_dataset = preprocess_hh_dataset(raw_dataset["train"], num_proc=4)

    # Create splits to get test set
    _, _, test_dataset = create_splits(processed_dataset, val_ratio=0.05, test_ratio=0.05)

    if args.num_samples:
        logger.info(f"Using {args.num_samples} samples from test set")
        test_dataset = test_dataset.select(range(min(args.num_samples, len(test_dataset))))

    logger.info(f"Test set size: {len(test_dataset)}")

    # Create PyTorch dataset
    test_dataset_torch = PreferenceDataset(
        test_dataset,
        tokenizer,
        max_length=args.max_length,
        max_prompt_length=256,
    )

    # Create dataloader
    collator = PreferenceDataCollator(pad_token_id=tokenizer.pad_token_id)
    test_loader = DataLoader(
        test_dataset_torch,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    # Create evaluator
    logger.info("\nCreating evaluator...")
    evaluator = Evaluator(
        models=models,
        reference_model=reference_model,
        test_dataloader=test_loader,
        tokenizer=tokenizer,
        beta=args.beta,
        device=device,
    )

    # Run evaluation
    logger.info("\nRunning evaluation...")
    results = evaluator.evaluate_all()

    # Print comparison table
    logger.info("\n" + "="*80)
    logger.info("Results Summary")
    logger.info("="*80 + "\n")
    table = evaluator.create_comparison_table(results)
    print(table)

    # Save results
    evaluator.save_results(results, args.output_file)

    # Generate comparison samples
    if args.num_generation_samples > 0:
        logger.info(f"\nGenerating {args.num_generation_samples} comparison samples...")

        # Get sample prompts from test set
        sample_indices = list(range(min(args.num_generation_samples, len(test_dataset))))
        sample_prompts = [test_dataset[i]["prompt"] + "\n\nAssistant:" for i in sample_indices]

        generations = evaluator.generate_comparison_samples(
            prompts=sample_prompts,
            max_new_tokens=100,
            temperature=0.7,
        )

        # Save generation samples
        output_dir = Path(args.output_file).parent
        generation_file = output_dir / "generation_samples.json"
        evaluator.save_generation_samples(
            sample_prompts,
            generations,
            str(generation_file),
        )

    logger.info("\nEvaluation completed!")


if __name__ == "__main__":
    main()
