"""DPO loss implementation.

Direct Preference Optimization (DPO) loss from the paper:
"Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Dict


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
    loss_type: str = "sigmoid",
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute DPO loss.

    The DPO loss is:
        L_DPO = -E[log σ(β * (log π_θ(y_w|x)/π_ref(y_w|x)
                                - log π_θ(y_l|x)/π_ref(y_l|x)))]

    Where:
    - π_θ is the policy model being trained
    - π_ref is the frozen reference model
    - y_w is the chosen (preferred) response
    - y_l is the rejected (dispreferred) response
    - β is a temperature parameter controlling deviation from reference
    - σ is the sigmoid function

    Args:
        policy_chosen_logps: Log probs of chosen responses under policy [batch_size]
        policy_rejected_logps: Log probs of rejected responses under policy [batch_size]
        reference_chosen_logps: Log probs of chosen responses under reference [batch_size]
        reference_rejected_logps: Log probs of rejected responses under reference [batch_size]
        beta: Temperature parameter (default: 0.1)
        label_smoothing: Label smoothing factor (default: 0.0)
        loss_type: Type of loss - "sigmoid" (standard DPO), "ipo", or "hinge"

    Returns:
        loss: Scalar loss value
        metrics: Dictionary of auxiliary metrics for logging
    """
    # Compute log ratios: log(π_θ(y|x) / π_ref(y|x))
    chosen_logratios = policy_chosen_logps - reference_chosen_logps
    rejected_logratios = policy_rejected_logps - reference_rejected_logps

    # Implicit rewards: r(x,y) = β * log(π_θ(y|x) / π_ref(y|x))
    chosen_rewards = beta * chosen_logratios
    rejected_rewards = beta * rejected_logratios

    # Preference logits: r(x,y_w) - r(x,y_l)
    logits = chosen_rewards - rejected_rewards

    # Compute loss based on type
    if loss_type == "sigmoid":
        # Standard DPO loss: -log(σ(logits))
        # With label smoothing: (1-ε) * -log(σ(logits)) + ε * -log(σ(-logits))
        losses = (
            -F.logsigmoid(logits) * (1 - label_smoothing)
            - F.logsigmoid(-logits) * label_smoothing
        )
    elif loss_type == "ipo":
        # IPO loss (Identity Preference Optimization)
        # (logits - 1/(2*β))^2
        losses = (logits - 1.0 / (2 * beta)) ** 2
    elif loss_type == "hinge":
        # Hinge loss: max(0, 1 - logits)
        losses = F.relu(1 - logits)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    loss = losses.mean()

    # Compute metrics for logging
    with torch.no_grad():
        # Accuracy: fraction of examples where chosen is preferred
        accuracy = (logits > 0).float().mean()

        # Reward margin: average difference in rewards
        reward_margin = (chosen_rewards - rejected_rewards).mean()

        # Individual reward statistics
        chosen_rewards_mean = chosen_rewards.mean()
        rejected_rewards_mean = rejected_rewards.mean()

        # Log ratio statistics (measure of deviation from reference)
        chosen_logratios_mean = chosen_logratios.mean()
        rejected_logratios_mean = rejected_logratios.mean()

    metrics = {
        "loss": loss.detach(),
        "accuracy": accuracy,
        "reward_margin": reward_margin,
        "chosen_rewards": chosen_rewards_mean,
        "rejected_rewards": rejected_rewards_mean,
        "chosen_logratios": chosen_logratios_mean,
        "rejected_logratios": rejected_logratios_mean,
    }

    return loss, metrics


def dpo_loss_with_margin(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    margin: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    DPO loss with explicit margin.

    This variant adds a margin to the preference logits:
        L = -log(σ(β * logratios - margin))

    Useful for making the optimization more conservative.

    Args:
        Same as dpo_loss, plus:
        margin: Minimum margin to enforce between chosen and rejected

    Returns:
        loss: Scalar loss value
        metrics: Dictionary of auxiliary metrics
    """
    chosen_logratios = policy_chosen_logps - reference_chosen_logps
    rejected_logratios = policy_rejected_logps - reference_rejected_logps

    chosen_rewards = beta * chosen_logratios
    rejected_rewards = beta * rejected_logratios

    # Apply margin
    logits = chosen_rewards - rejected_rewards - margin

    losses = -F.logsigmoid(logits)
    loss = losses.mean()

    with torch.no_grad():
        accuracy = (logits > 0).float().mean()
        reward_margin = (chosen_rewards - rejected_rewards).mean()

    metrics = {
        "loss": loss.detach(),
        "accuracy": accuracy,
        "reward_margin": reward_margin,
        "chosen_rewards": chosen_rewards.mean(),
        "rejected_rewards": rejected_rewards.mean(),
    }

    return loss, metrics
