"""Policy model wrapper."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Tuple


class PolicyModel(nn.Module):
    """
    Wrapper around a HuggingFace causal language model for DPO/SFT training.

    Provides methods to compute log probabilities for sequences.
    """

    def __init__(
        self,
        model_name_or_path: str,
        use_gradient_checkpointing: bool = True,
        torch_dtype: str = "float16",
    ):
        """
        Initialize policy model.

        Args:
            model_name_or_path: HuggingFace model name or path to checkpoint
            use_gradient_checkpointing: Enable gradient checkpointing to save memory
            torch_dtype: Data type for model weights
        """
        super().__init__()

        # Map string dtype to torch dtype
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map.get(torch_dtype, torch.float16)

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
        )

        # Enable gradient checkpointing
        if use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Standard forward pass (for SFT training).

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Labels for loss computation [batch_size, seq_len]

        Returns:
            Model outputs (including loss if labels provided)
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def get_logprobs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-sequence log probabilities.

        This is used for DPO training where we need the log probability
        of the entire response (not individual tokens).

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Labels [batch_size, seq_len] (use -100 to mask prompt)

        Returns:
            Log probabilities for each sequence [batch_size]
        """
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]

        # Shift logits and labels for next-token prediction
        # logits: predict token at position i from tokens 0..i-1
        # labels: token at position i
        shift_logits = logits[:, :-1, :].contiguous()  # [batch_size, seq_len-1, vocab_size]
        shift_labels = labels[:, 1:].contiguous()  # [batch_size, seq_len-1]

        # Get log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)  # [batch_size, seq_len-1, vocab_size]

        # Mask out prompt tokens (where labels == -100)
        mask = (shift_labels != -100).float()  # [batch_size, seq_len-1]

        # Replace -100 with 0 for gathering (will be masked out anyway)
        shift_labels_masked = shift_labels.clone()
        shift_labels_masked[shift_labels == -100] = 0

        # Gather log probs at label positions
        # For each position, get log_prob of the actual token
        gathered_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=shift_labels_masked.unsqueeze(-1),
        ).squeeze(-1)  # [batch_size, seq_len-1]

        # Sum log probs over sequence (only for response tokens)
        # This gives us log P(response | prompt)
        sequence_log_probs = (gathered_log_probs * mask).sum(dim=-1)  # [batch_size]

        return sequence_log_probs

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        **kwargs,
    ):
        """
        Generate text.

        Args:
            input_ids: Prompt token IDs
            attention_mask: Attention mask
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **kwargs: Additional generation arguments

        Returns:
            Generated token IDs
        """
        if "do_sample" in kwargs:
            do_sample = kwargs.pop("do_sample")
        if "pad_token_id" in kwargs:
            pad_token_id = kwargs.pop("pad_token_id")
        if pad_token_id is None:
            pad_token_id = self.model.config.eos_token_id

        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=pad_token_id,
            **kwargs,
        )

    def save_pretrained(self, path: str):
        """Save model to disk."""
        self.model.save_pretrained(path)

    @classmethod
    def from_pretrained(cls, path: str, **kwargs):
        """Load model from disk."""
        instance = cls.__new__(cls)
        super(PolicyModel, instance).__init__()
        instance.model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
        return instance
