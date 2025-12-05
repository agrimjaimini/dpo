"""Device selection utilities."""

import torch
import logging


def get_device() -> torch.device:
    """
    Get the best available device for training/inference.

    Priority:
    1. MPS (Metal Performance Shaders) for Apple Silicon
    2. CUDA for NVIDIA GPUs
    3. CPU as fallback

    Returns:
        torch.device: The selected device

    Note:
        MPS works best with fp16=true in the config.
        Some operations may fall back to CPU if not supported on MPS.
    """
    logger = logging.getLogger(__name__)

    if torch.backends.mps.is_available():
        logger.info("Using MPS (Metal Performance Shaders) for GPU acceleration on Mac")
        logger.info("For best performance, ensure fp16=true in your config")
        return torch.device("mps")
    elif torch.cuda.is_available():
        logger.info("Using CUDA for GPU acceleration")
        return torch.device("cuda")
    else:
        logger.info("Using CPU (no GPU acceleration available)")
        return torch.device("cpu")


def get_device_name() -> str:
    """
    Get the name of the best available device.

    Returns:
        str: Device name ("mps", "cuda", or "cpu")
    """
    device = get_device()
    return device.type
