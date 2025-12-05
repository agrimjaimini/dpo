"""Evaluation metrics for preference models."""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict
from tqdm import tqdm


def compute_preference_accuracy(
    model,
    ref_model,
    dataloader: DataLoader,
    beta: float,
    device: str = "cuda",
) -> float:
    """
    Compute preference accuracy: fraction of examples where model prefers chosen over rejected.

    Args:
        model: Policy model
        ref_model: Reference model
        dataloader: DataLoader with preference pairs
        beta: DPO beta parameter
        device: Device to run on

    Returns:
        Accuracy (fraction of correctly ordered preferences)
    """
    model.eval()
    model.to(device)
    ref_model.to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing preference accuracy"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Get log probs
            policy_chosen_logps = model.get_logprobs(
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
                batch["chosen_labels"],
            )
            policy_rejected_logps = model.get_logprobs(
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
                batch["rejected_labels"],
            )
            ref_chosen_logps = ref_model.get_logprobs(
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
                batch["chosen_labels"],
            )
            ref_rejected_logps = ref_model.get_logprobs(
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
                batch["rejected_labels"],
            )

            # Compute implicit rewards
            chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
            rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)

            # Count correct preferences
            correct += (chosen_rewards > rejected_rewards).sum().item()
            total += len(chosen_rewards)

    return correct / total if total > 0 else 0.0


def compute_reward_margin(
    model,
    ref_model,
    dataloader: DataLoader,
    beta: float,
    device: str = "cuda",
) -> float:
    """
    Compute average reward margin: reward(chosen) - reward(rejected).

    Args:
        model: Policy model
        ref_model: Reference model
        dataloader: DataLoader with preference pairs
        beta: DPO beta parameter
        device: Device to run on

    Returns:
        Average reward margin
    """
    model.eval()
    model.to(device)
    ref_model.to(device)

    total_margin = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing reward margin"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Get log probs
            policy_chosen_logps = model.get_logprobs(
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
                batch["chosen_labels"],
            )
            policy_rejected_logps = model.get_logprobs(
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
                batch["rejected_labels"],
            )
            ref_chosen_logps = ref_model.get_logprobs(
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
                batch["chosen_labels"],
            )
            ref_rejected_logps = ref_model.get_logprobs(
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
                batch["rejected_labels"],
            )

            # Compute rewards
            chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
            rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)

            # Accumulate margin
            total_margin += (chosen_rewards - rejected_rewards).sum().item()
            total_samples += len(chosen_rewards)

    return total_margin / total_samples if total_samples > 0 else 0.0


def compute_perplexity(
    model,
    dataloader: DataLoader,
    device: str = "cuda",
) -> float:
    """
    Compute perplexity on chosen responses.

    Args:
        model: Model to evaluate
        dataloader: DataLoader (should have input_ids, attention_mask, labels)
        device: Device to run on

    Returns:
        Perplexity
    """
    model.eval()
    model.to(device)

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing perplexity"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # For SFT data
            if "input_ids" in batch:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
                # Count non-masked tokens
                num_tokens = (batch["labels"] != -100).sum().item()
            # For preference data (use chosen)
            elif "chosen_input_ids" in batch:
                outputs = model(
                    input_ids=batch["chosen_input_ids"],
                    attention_mask=batch["chosen_attention_mask"],
                    labels=batch["chosen_labels"],
                )
                loss = outputs.loss
                num_tokens = (batch["chosen_labels"] != -100).sum().item()
            else:
                raise ValueError("Unknown batch format")

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return perplexity


def compute_kl_divergence(
    model,
    ref_model,
    dataloader: DataLoader,
    device: str = "cuda",
) -> float:
    """
    Compute average KL divergence: KL(policy || reference).

    Args:
        model: Policy model
        ref_model: Reference model
        dataloader: DataLoader
        device: Device to run on

    Returns:
        Average KL divergence
    """
    model.eval()
    ref_model.to(device)
    model.to(device)

    total_kl = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing KL divergence"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Use chosen responses
            policy_logps = model.get_logprobs(
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
                batch["chosen_labels"],
            )
            ref_logps = ref_model.get_logprobs(
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
                batch["chosen_labels"],
            )

            # KL(policy || ref) = E[log policy - log ref]
            # Since we have log probs for full sequences, approximate as difference
            kl = (policy_logps - ref_logps).abs()  # Approximation

            total_kl += kl.sum().item()
            total_samples += len(kl)

    return total_kl / total_samples if total_samples > 0 else 0.0
