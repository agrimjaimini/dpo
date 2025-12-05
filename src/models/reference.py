"""Reference model for DPO training."""

import torch
import copy
from typing import Optional
from .policy import PolicyModel


class ReferenceModel:
    """
    Frozen reference model for DPO training.

    This is a copy of the policy model that is kept frozen (no gradients)
    and used to compute the KL penalty in the DPO loss.
    """

    def __init__(self, policy_model: PolicyModel):
        """
        Initialize reference model as a frozen copy of the policy.

        Args:
            policy_model: Policy model to copy
        """
        # Deep copy the policy model
        self.model = copy.deepcopy(policy_model)

        # Set to eval mode
        self.model.eval()

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def get_logprobs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-sequence log probabilities (without gradients).

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Labels [batch_size, seq_len]

        Returns:
            Log probabilities for each sequence [batch_size]
        """
        return self.model.get_logprobs(input_ids, attention_mask, labels)

    def to(self, device):
        """Move model to device."""
        self.model = self.model.to(device)
        return self

    def cuda(self):
        """Move model to CUDA."""
        return self.to("cuda")

    def cpu(self):
        """Move model to CPU."""
        return self.to("cpu")


class SharedReferenceModel:
    """
    Memory-efficient reference model that shares weights with the policy.

    Instead of copying the entire model, this keeps a reference to the
    policy model but ensures it's in eval mode when computing reference
    log probabilities.

    Note: This is more memory efficient but requires careful handling to
    avoid computing gradients through the reference model.
    """

    def __init__(self, policy_model: PolicyModel):
        """
        Initialize shared reference model.

        Args:
            policy_model: Policy model to share weights with
        """
        self.model = policy_model

    @torch.no_grad()
    def get_logprobs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-sequence log probabilities (without gradients).

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Labels [batch_size, seq_len]

        Returns:
            Log probabilities for each sequence [batch_size]
        """
        # Ensure model is in eval mode
        training_mode = self.model.training
        self.model.eval()

        # Compute log probs
        logprobs = self.model.get_logprobs(input_ids, attention_mask, labels)

        # Restore training mode
        self.model.train(training_mode)

        return logprobs

    def to(self, device):
        """No-op since weights are shared."""
        return self

    def cuda(self):
        """No-op since weights are shared."""
        return self

    def cpu(self):
        """No-op since weights are shared."""
        return self
