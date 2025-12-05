"""Preference dataset for DPO training."""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional
from transformers import PreTrainedTokenizer
from datasets import Dataset as HFDataset


class PreferenceDataset(Dataset):
    """
    PyTorch Dataset for preference pairs.

    Returns tokenized (prompt, chosen, rejected) triples for DPO training.
    """

    def __init__(
        self,
        data: HFDataset,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        max_prompt_length: int = 256,
    ):
        """
        Initialize preference dataset.

        Args:
            data: HuggingFace dataset with 'prompt', 'chosen', 'rejected' fields
            tokenizer: HuggingFace tokenizer
            max_length: Maximum total sequence length
            max_prompt_length: Maximum prompt length (to ensure space for response)
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length

        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single example.

        Returns:
            Dictionary with tokenized prompt, chosen, and rejected sequences
        """
        example = self.data[idx]

        # Format inputs: prompt + response
        # For HH format, we add the assistant marker
        prompt = example["prompt"]
        if not prompt.strip().endswith("Assistant:"):
            prompt = prompt + "\n\nAssistant:"

        chosen_text = prompt + " " + example["chosen"]
        rejected_text = prompt + " " + example["rejected"]

        # Tokenize
        prompt_tokens = self.tokenizer(
            prompt,
            max_length=self.max_prompt_length,
            truncation=True,
            add_special_tokens=True,
        )

        chosen_tokens = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=True,
        )

        rejected_tokens = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=True,
        )

        # Get prompt length for masking
        prompt_len = len(prompt_tokens["input_ids"])

        # Create labels (mask prompt tokens with -100 so they don't contribute to loss)
        chosen_labels = chosen_tokens["input_ids"].copy()
        chosen_labels[:prompt_len] = [-100] * prompt_len

        rejected_labels = rejected_tokens["input_ids"].copy()
        rejected_labels[:prompt_len] = [-100] * prompt_len

        return {
            "prompt_input_ids": torch.tensor(prompt_tokens["input_ids"], dtype=torch.long),
            "prompt_attention_mask": torch.tensor(prompt_tokens["attention_mask"], dtype=torch.long),
            "chosen_input_ids": torch.tensor(chosen_tokens["input_ids"], dtype=torch.long),
            "chosen_attention_mask": torch.tensor(chosen_tokens["attention_mask"], dtype=torch.long),
            "chosen_labels": torch.tensor(chosen_labels, dtype=torch.long),
            "rejected_input_ids": torch.tensor(rejected_tokens["input_ids"], dtype=torch.long),
            "rejected_attention_mask": torch.tensor(rejected_tokens["attention_mask"], dtype=torch.long),
            "rejected_labels": torch.tensor(rejected_labels, dtype=torch.long),
        }


class SFTDataset(Dataset):
    """
    Dataset for SFT training (supervised fine-tuning on chosen responses only).
    """

    def __init__(
        self,
        data: HFDataset,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        max_prompt_length: int = 256,
    ):
        """
        Initialize SFT dataset.

        Args:
            data: HuggingFace dataset with 'prompt', 'chosen' fields
            tokenizer: HuggingFace tokenizer
            max_length: Maximum total sequence length
            max_prompt_length: Maximum prompt length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length

        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example (prompt + chosen response)."""
        example = self.data[idx]

        # Format input
        prompt = example["prompt"]
        if not prompt.strip().endswith("Assistant:"):
            prompt = prompt + "\n\nAssistant:"

        text = prompt + " " + example["chosen"]

        # Tokenize
        prompt_tokens = self.tokenizer(
            prompt,
            max_length=self.max_prompt_length,
            truncation=True,
            add_special_tokens=True,
        )

        tokens = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=True,
        )

        # Create labels (mask prompt tokens)
        prompt_len = len(prompt_tokens["input_ids"])
        labels = tokens["input_ids"].copy()
        labels[:prompt_len] = [-100] * prompt_len

        return {
            "input_ids": torch.tensor(tokens["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(tokens["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
