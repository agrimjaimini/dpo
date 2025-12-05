"""Base trainer with common training infrastructure."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

from ..utils.logging_utils import WandBLogger, MetricsTracker
from ..utils.device import get_device


class BaseTrainer(ABC):
    """
    Base trainer class with common training infrastructure.

    Provides:
    - Optimizer and scheduler setup
    - Gradient accumulation
    - Mixed precision training
    - Checkpointing
    - Logging (console + wandb)
    - Evaluation loop
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset],
        collate_fn: Any,
        config: Dict[str, Any],
        output_dir: str,
    ):
        """
        Initialize base trainer.

        Args:
            model: Model to train
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            collate_fn: Data collator function
            config: Training configuration
            output_dir: Directory to save checkpoints
        """
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.collate_fn = collate_fn
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup device (prioritize MPS for Mac, then CUDA, then CPU)
        self.device = get_device()
        self.model.to(self.device)

        # Training config
        training_config = config.get("training", config)  # Handle nested config
        self.num_epochs = training_config.get("num_epochs", 1)
        self.per_device_batch_size = training_config.get("per_device_batch_size", 8)
        self.gradient_accumulation_steps = training_config.get("gradient_accumulation_steps", 1)
        self.learning_rate = training_config.get("learning_rate", 2e-4)
        self.warmup_ratio = training_config.get("warmup_ratio", 0.1)
        self.weight_decay = training_config.get("weight_decay", 0.01)
        self.max_grad_norm = training_config.get("max_grad_norm", 1.0)
        self.fp16 = training_config.get("fp16", True)
        self.logging_steps = training_config.get("logging_steps", 10)
        self.eval_steps = training_config.get("eval_steps", 500)
        self.save_steps = training_config.get("save_steps", 500)
        self.save_total_limit = training_config.get("save_total_limit", 3)
        self.seed = training_config.get("seed", 42)

        # Set seed
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.per_device_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        self.eval_loader = None
        if eval_dataset is not None:
            self.eval_loader = DataLoader(
                eval_dataset,
                batch_size=self.per_device_batch_size,
                shuffle=False,
                collate_fn=collate_fn,
            )

        # Setup optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Setup mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if self.fp16 else None

        # Setup logging
        self.logger = logging.getLogger(__name__)
        logging_config = config.get("logging", {})
        self.wandb_logger = None
        if logging_config.get("use_wandb", False):
            self.wandb_logger = WandBLogger(
                project=logging_config.get("wandb_project", "dpo-hh"),
                name=logging_config.get("wandb_run_name"),
                config=config,
            )

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float("inf")
        self.checkpoints = []

    def _create_optimizer(self) -> AdamW:
        """Create AdamW optimizer."""
        # Separate parameters with and without weight decay
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]

        return AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

    def _create_scheduler(self):
        """Create learning rate scheduler with linear warmup."""
        num_training_steps = len(self.train_loader) * self.num_epochs // self.gradient_accumulation_steps
        num_warmup_steps = int(num_training_steps * self.warmup_ratio)

        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

    @abstractmethod
    def training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Perform a single training step.

        Must be implemented by subclasses.

        Args:
            batch: Batch of data

        Returns:
            Loss tensor
        """
        pass

    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        self.logger.info(f"  Num examples: {len(self.train_dataset)}")
        self.logger.info(f"  Num epochs: {self.num_epochs}")
        self.logger.info(f"  Batch size: {self.per_device_batch_size}")
        self.logger.info(f"  Gradient accumulation steps: {self.gradient_accumulation_steps}")
        self.logger.info(f"  Total optimization steps: {len(self.train_loader) * self.num_epochs // self.gradient_accumulation_steps}")

        self.model.train()
        metrics_tracker = MetricsTracker()

        for epoch in range(self.num_epochs):
            self.epoch = epoch
            self.logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs}")

            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")

            for step, batch in enumerate(progress_bar):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # Forward pass with mixed precision
                if self.fp16:
                    with torch.cuda.amp.autocast():
                        loss = self.training_step(batch)
                        loss = loss / self.gradient_accumulation_steps
                else:
                    loss = self.training_step(batch)
                    loss = loss / self.gradient_accumulation_steps

                # Backward pass
                if self.fp16:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Update metrics
                metrics_tracker.update({"loss": loss.item() * self.gradient_accumulation_steps})

                # Optimizer step (after gradient accumulation)
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    if self.fp16:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.optimizer.step()

                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    # Logging
                    if self.global_step % self.logging_steps == 0:
                        metrics = metrics_tracker.compute()
                        metrics["learning_rate"] = self.scheduler.get_last_lr()[0]
                        metrics["epoch"] = epoch

                        progress_bar.set_postfix(metrics)

                        if self.wandb_logger:
                            self.wandb_logger.log(metrics, step=self.global_step)

                        metrics_tracker.reset()

                    # Evaluation
                    if self.eval_loader and self.global_step % self.eval_steps == 0:
                        eval_metrics = self.evaluate()
                        self.logger.info(f"Eval metrics: {eval_metrics}")

                        if self.wandb_logger:
                            eval_metrics_prefixed = {f"eval/{k}": v for k, v in eval_metrics.items()}
                            self.wandb_logger.log(eval_metrics_prefixed, step=self.global_step)

                        self.model.train()

                    # Checkpointing
                    if self.global_step % self.save_steps == 0:
                        self.save_checkpoint()

        # Final evaluation and checkpoint
        if self.eval_loader:
            eval_metrics = self.evaluate()
            self.logger.info(f"Final eval metrics: {eval_metrics}")

        self.save_checkpoint(is_final=True)

        if self.wandb_logger:
            self.wandb_logger.finish()

        self.logger.info("Training completed!")

    def evaluate(self) -> Dict[str, float]:
        """
        Run evaluation.

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        metrics_tracker = MetricsTracker()

        with torch.no_grad():
            for batch in tqdm(self.eval_loader, desc="Evaluating"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                if self.fp16:
                    with torch.cuda.amp.autocast():
                        loss = self.training_step(batch)
                else:
                    loss = self.training_step(batch)

                metrics_tracker.update({"loss": loss.item()})

        return metrics_tracker.compute()

    def save_checkpoint(self, is_final: bool = False):
        """
        Save model checkpoint.

        Args:
            is_final: Whether this is the final checkpoint
        """
        if is_final:
            checkpoint_dir = self.output_dir / "final"
        else:
            checkpoint_dir = self.output_dir / f"checkpoint-{self.global_step}"

        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save_pretrained(str(checkpoint_dir))

        # Track checkpoints for cleanup
        if not is_final:
            self.checkpoints.append(checkpoint_dir)

            # Remove old checkpoints if exceeding limit
            if len(self.checkpoints) > self.save_total_limit:
                oldest_checkpoint = self.checkpoints.pop(0)
                if oldest_checkpoint.exists():
                    import shutil
                    shutil.rmtree(oldest_checkpoint)

        self.logger.info(f"Saved checkpoint to {checkpoint_dir}")
