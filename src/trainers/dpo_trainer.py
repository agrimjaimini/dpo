"""DPO (Direct Preference Optimization) trainer."""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Dict, Any, Optional
from pathlib import Path

from .base_trainer import BaseTrainer
from ..models.policy import PolicyModel
from ..models.reference import ReferenceModel
from ..losses.dpo_loss import dpo_loss


class DPOTrainer(BaseTrainer):
    """
    Trainer for Direct Preference Optimization (DPO).

    Trains a policy model using preference data by optimizing the DPO objective.
    Requires a frozen reference model for computing the KL penalty.
    """

    def __init__(
        self,
        policy_model: PolicyModel,
        reference_model: ReferenceModel,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset],
        collate_fn: Any,
        config: Dict[str, Any],
        output_dir: str,
    ):
        """
        Initialize DPO trainer.

        Args:
            policy_model: Policy model to train
            reference_model: Frozen reference model
            train_dataset: Training dataset with preference pairs
            eval_dataset: Evaluation dataset
            collate_fn: Data collator
            config: Training configuration
            output_dir: Directory to save checkpoints
        """
        # Initialize base trainer
        super().__init__(
            model=policy_model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            collate_fn=collate_fn,
            config=config,
            output_dir=output_dir,
        )

        # Store reference model
        self.ref_model = reference_model
        self.ref_model.to(self.device)

        # DPO-specific config
        self.beta = config.get("beta", 0.1)
        self.loss_type = config.get("loss_type", "sigmoid")
        self.label_smoothing = config.get("label_smoothing", 0.0)

        self.logger.info(f"DPO Config: beta={self.beta}, loss_type={self.loss_type}")

    def training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Perform a single DPO training step.

        Computes log probabilities for both chosen and rejected responses
        under the policy and reference models, then optimizes the DPO loss.

        Args:
            batch: Batch containing chosen and rejected sequences

        Returns:
            Loss tensor
        """
        # Get log probs from policy model for chosen responses
        policy_chosen_logps = self.model.get_logprobs(
            input_ids=batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"],
            labels=batch["chosen_labels"],
        )

        # Get log probs from policy model for rejected responses
        policy_rejected_logps = self.model.get_logprobs(
            input_ids=batch["rejected_input_ids"],
            attention_mask=batch["rejected_attention_mask"],
            labels=batch["rejected_labels"],
        )

        # Get log probs from reference model for chosen responses
        ref_chosen_logps = self.ref_model.get_logprobs(
            input_ids=batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"],
            labels=batch["chosen_labels"],
        )

        # Get log probs from reference model for rejected responses
        ref_rejected_logps = self.ref_model.get_logprobs(
            input_ids=batch["rejected_input_ids"],
            attention_mask=batch["rejected_attention_mask"],
            labels=batch["rejected_labels"],
        )

        # Compute DPO loss
        loss, metrics = dpo_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            reference_chosen_logps=ref_chosen_logps,
            reference_rejected_logps=ref_rejected_logps,
            beta=self.beta,
            label_smoothing=self.label_smoothing,
            loss_type=self.loss_type,
        )

        # Log DPO-specific metrics
        if self.global_step % self.logging_steps == 0:
            for key, value in metrics.items():
                if key != "loss":  # loss is already logged by base trainer
                    if self.wandb_logger:
                        self.wandb_logger.log({f"train/{key}": value.item()}, step=self.global_step)

        return loss

    def evaluate(self) -> Dict[str, float]:
        """
        Run evaluation with DPO-specific metrics.

        Returns:
            Dictionary of evaluation metrics including DPO metrics
        """
        self.model.eval()
        self.ref_model.to(self.device)  # Ensure reference model is on device

        total_loss = 0.0
        total_accuracy = 0.0
        total_reward_margin = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.eval_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # Get log probs
                policy_chosen_logps = self.model.get_logprobs(
                    batch["chosen_input_ids"],
                    batch["chosen_attention_mask"],
                    batch["chosen_labels"],
                )
                policy_rejected_logps = self.model.get_logprobs(
                    batch["rejected_input_ids"],
                    batch["rejected_attention_mask"],
                    batch["rejected_labels"],
                )
                ref_chosen_logps = self.ref_model.get_logprobs(
                    batch["chosen_input_ids"],
                    batch["chosen_attention_mask"],
                    batch["chosen_labels"],
                )
                ref_rejected_logps = self.ref_model.get_logprobs(
                    batch["rejected_input_ids"],
                    batch["rejected_attention_mask"],
                    batch["rejected_labels"],
                )

                # Compute loss and metrics
                loss, metrics = dpo_loss(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    ref_chosen_logps,
                    ref_rejected_logps,
                    beta=self.beta,
                    label_smoothing=self.label_smoothing,
                    loss_type=self.loss_type,
                )

                total_loss += loss.item()
                total_accuracy += metrics["accuracy"].item()
                total_reward_margin += metrics["reward_margin"].item()
                num_batches += 1

        return {
            "loss": total_loss / num_batches,
            "accuracy": total_accuracy / num_batches,
            "reward_margin": total_reward_margin / num_batches,
        }
