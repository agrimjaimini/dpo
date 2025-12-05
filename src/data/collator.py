"""Data collators for batching."""

import torch
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class PreferenceDataCollator:
    """
    Collator for preference dataset.

    Pads sequences to the maximum length in the batch.
    """

    pad_token_id: int = 0

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of preference examples.

        Args:
            features: List of examples from PreferenceDataset

        Returns:
            Batched tensors with padding
        """
        # Extract all fields
        batch = {}

        # For each field, pad to max length in batch
        for key in features[0].keys():
            # Stack tensors
            tensors = [f[key] for f in features]

            # Pad sequences
            if "input_ids" in key or "labels" in key or "attention_mask" in key:
                # Determine padding value
                if "labels" in key:
                    padding_value = -100
                elif "attention_mask" in key:
                    padding_value = 0
                else:
                    padding_value = self.pad_token_id

                # Pad to max length in batch
                batch[key] = self._pad_sequence(tensors, padding_value)
            else:
                batch[key] = torch.stack(tensors)

        return batch

    def _pad_sequence(
        self,
        sequences: List[torch.Tensor],
        padding_value: int,
    ) -> torch.Tensor:
        """
        Pad sequences to the same length.

        Args:
            sequences: List of 1D tensors
            padding_value: Value to use for padding

        Returns:
            2D tensor of shape (batch_size, max_length)
        """
        # Find max length
        max_length = max(len(seq) for seq in sequences)

        # Pad each sequence
        padded = []
        for seq in sequences:
            padding_length = max_length - len(seq)
            if padding_length > 0:
                padding = torch.full(
                    (padding_length,),
                    padding_value,
                    dtype=seq.dtype,
                )
                padded_seq = torch.cat([seq, padding])
            else:
                padded_seq = seq
            padded.append(padded_seq)

        return torch.stack(padded)


@dataclass
class SFTDataCollator:
    """
    Collator for SFT dataset.

    Pads sequences to the maximum length in the batch.
    """

    pad_token_id: int = 0

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of SFT examples.

        Args:
            features: List of examples from SFTDataset

        Returns:
            Batched tensors with padding
        """
        batch = {}

        for key in features[0].keys():
            tensors = [f[key] for f in features]

            # Determine padding value
            if key == "labels":
                padding_value = -100
            elif key == "attention_mask":
                padding_value = 0
            else:
                padding_value = self.pad_token_id

            # Pad to max length in batch
            batch[key] = self._pad_sequence(tensors, padding_value)

        return batch

    def _pad_sequence(
        self,
        sequences: List[torch.Tensor],
        padding_value: int,
    ) -> torch.Tensor:
        """Pad sequences to the same length."""
        max_length = max(len(seq) for seq in sequences)

        padded = []
        for seq in sequences:
            padding_length = max_length - len(seq)
            if padding_length > 0:
                padding = torch.full(
                    (padding_length,),
                    padding_value,
                    dtype=seq.dtype,
                )
                padded_seq = torch.cat([seq, padding])
            else:
                padded_seq = seq
            padded.append(padded_seq)

        return torch.stack(padded)
