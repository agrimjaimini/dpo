"""SFT (Supervised Fine-Tuning) trainer."""

import torch
from typing import Dict
from .base_trainer import BaseTrainer


class SFTTrainer(BaseTrainer):
    """
    Trainer for Supervised Fine-Tuning (SFT).

    Trains the model on chosen responses using standard language modeling loss.
    This creates the initial policy model before DPO training.
    """

    def training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Perform a single SFT training step.

        For SFT, we simply compute the language modeling loss on the
        chosen responses (ignoring the rejected responses).

        Args:
            batch: Batch containing input_ids, attention_mask, and labels

        Returns:
            Loss tensor
        """
        # Forward pass through model
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        return outputs.loss
