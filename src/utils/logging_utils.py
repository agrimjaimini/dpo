"""Logging utilities."""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


def setup_logging(
    log_level: str = "info",
    log_file: Optional[str] = None,
    name: Optional[str] = None,
) -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        log_level: Logging level (debug, info, warning, error, critical)
        log_file: Optional path to log file
        name: Logger name (defaults to root logger)

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)

    # Convert log level string to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # Clear existing handlers
    logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class WandBLogger:
    """Wrapper for Weights & Biases logging."""

    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
    ):
        """
        Initialize WandB logger.

        Args:
            project: WandB project name
            name: Run name (auto-generated if None)
            config: Configuration dictionary to log
            enabled: Whether to actually log to wandb
        """
        self.enabled = enabled

        if self.enabled:
            try:
                import wandb

                self.wandb = wandb

                # Initialize run
                self.run = wandb.init(
                    project=project,
                    name=name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    config=config,
                )
            except ImportError:
                logging.warning("wandb not installed. Disabling wandb logging.")
                self.enabled = False

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to wandb."""
        if self.enabled:
            self.wandb.log(metrics, step=step)

    def finish(self):
        """Finish wandb run."""
        if self.enabled:
            self.wandb.finish()


class MetricsTracker:
    """Track and aggregate training metrics."""

    def __init__(self):
        self.metrics = {}
        self.counts = {}

    def update(self, metrics: Dict[str, float]):
        """Update metrics with new values."""
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = 0.0
                self.counts[key] = 0

            self.metrics[key] += value
            self.counts[key] += 1

    def compute(self) -> Dict[str, float]:
        """Compute averaged metrics."""
        return {
            key: self.metrics[key] / self.counts[key]
            for key in self.metrics
            if self.counts[key] > 0
        }

    def reset(self):
        """Reset all metrics."""
        self.metrics = {}
        self.counts = {}

    def __repr__(self) -> str:
        computed = self.compute()
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in computed.items()])
        return f"Metrics({metrics_str})"
