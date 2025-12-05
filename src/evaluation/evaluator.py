"""Comprehensive evaluation pipeline."""

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from typing import Dict, List, Optional
import json
from pathlib import Path
from tqdm import tqdm

from .metrics import (
    compute_preference_accuracy,
    compute_reward_margin,
    compute_perplexity,
    compute_kl_divergence,
)


class Evaluator:
    """
    Comprehensive model evaluator.

    Evaluates multiple models on a test set and generates comparison reports.
    """

    def __init__(
        self,
        models: Dict[str, any],  # {"name": model}
        reference_model: Optional[any] = None,
        test_dataloader: Optional[DataLoader] = None,
        tokenizer: Optional[any] = None,
        beta: float = 0.1,
        device: str = "cuda",
    ):
        """
        Initialize evaluator.

        Args:
            models: Dictionary mapping model names to model instances
            reference_model: Reference model for DPO metrics (optional)
            test_dataloader: DataLoader for test set
            tokenizer: Tokenizer for text generation
            beta: DPO beta parameter
            device: Device to run evaluation on
        """
        self.models = models
        self.reference_model = reference_model
        self.test_dataloader = test_dataloader
        self.tokenizer = tokenizer
        self.beta = beta
        self.device = device

    def evaluate_all(self) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all models on all metrics.

        Returns:
            Dictionary mapping model names to their metrics
        """
        results = {}

        for name, model in self.models.items():
            print(f"\n{'='*80}")
            print(f"Evaluating {name}")
            print(f"{'='*80}\n")

            model.to(self.device)
            model.eval()

            metrics = {}

            # Perplexity (for all models)
            if self.test_dataloader:
                try:
                    metrics["perplexity"] = compute_perplexity(
                        model,
                        self.test_dataloader,
                        device=self.device,
                    )
                except Exception as e:
                    print(f"Warning: Could not compute perplexity for {name}: {e}")
                    metrics["perplexity"] = None

            # DPO metrics (requires reference model)
            if self.reference_model and self.test_dataloader:
                try:
                    metrics["preference_accuracy"] = compute_preference_accuracy(
                        model,
                        self.reference_model,
                        self.test_dataloader,
                        beta=self.beta,
                        device=self.device,
                    )

                    metrics["reward_margin"] = compute_reward_margin(
                        model,
                        self.reference_model,
                        self.test_dataloader,
                        beta=self.beta,
                        device=self.device,
                    )

                    metrics["kl_divergence"] = compute_kl_divergence(
                        model,
                        self.reference_model,
                        self.test_dataloader,
                        device=self.device,
                    )
                except Exception as e:
                    print(f"Warning: Could not compute DPO metrics for {name}: {e}")

            results[name] = metrics

            # Print metrics
            print(f"\n{name} Results:")
            for metric_name, value in metrics.items():
                if value is not None:
                    print(f"  {metric_name}: {value:.4f}")

        return results

    def generate_comparison_samples(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
        temperature: float = 0.7,
    ) -> Dict[str, List[str]]:
        """
        Generate responses from all models for qualitative comparison.

        Args:
            prompts: List of prompt texts
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Dictionary mapping model names to lists of generated responses
        """
        if not self.tokenizer:
            print("Warning: No tokenizer provided, skipping generation")
            return {}

        results = {}

        for name, model in self.models.items():
            print(f"\nGenerating with {name}...")
            model.to(self.device)
            model.eval()

            generations = []

            for prompt in tqdm(prompts, desc=f"Generating ({name})"):
                # Tokenize prompt
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self.device)

                # Generate
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=True,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )

                # Decode (remove prompt)
                generated_text = self.tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )

                generations.append(generated_text)

            results[name] = generations

        return results

    def create_comparison_table(
        self,
        results: Dict[str, Dict[str, float]],
    ) -> str:
        """
        Create a markdown comparison table.

        Args:
            results: Results from evaluate_all()

        Returns:
            Markdown table string
        """
        if not results:
            return "No results to display."

        # Get all metrics
        all_metrics = set()
        for model_metrics in results.values():
            all_metrics.update(model_metrics.keys())
        all_metrics = sorted(list(all_metrics))

        # Create header
        table = "| Model | " + " | ".join(all_metrics) + " |\n"
        table += "|" + "---|" * (len(all_metrics) + 1) + "\n"

        # Add rows
        for model_name, metrics in results.items():
            row = f"| {model_name} |"
            for metric in all_metrics:
                value = metrics.get(metric)
                if value is not None:
                    row += f" {value:.4f} |"
                else:
                    row += " N/A |"
            table += row + "\n"

        return table

    def save_results(
        self,
        results: Dict[str, Dict[str, float]],
        output_path: str,
    ):
        """
        Save results to JSON file.

        Args:
            results: Results from evaluate_all()
            output_path: Path to save JSON
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {output_path}")

    def save_generation_samples(
        self,
        prompts: List[str],
        generations: Dict[str, List[str]],
        output_path: str,
    ):
        """
        Save generation samples to JSON file.

        Args:
            prompts: Original prompts
            generations: Generated responses from generate_comparison_samples()
            output_path: Path to save JSON
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Format for readability
        samples = []
        for i, prompt in enumerate(prompts):
            sample = {"prompt": prompt}
            for model_name, model_generations in generations.items():
                sample[model_name] = model_generations[i]
            samples.append(sample)

        with open(output_path, "w") as f:
            json.dump(samples, f, indent=2)

        print(f"\nGeneration samples saved to {output_path}")
