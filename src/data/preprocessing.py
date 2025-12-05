"""Anthropic HH data preprocessing."""

import re
from typing import List, Dict, Tuple
from datasets import Dataset, DatasetDict


def parse_conversation(text: str) -> List[Dict[str, str]]:
    """
    Parse a conversation string into a list of turns.

    The Anthropic HH format uses:
    - "\n\nHuman: ..." for human turns
    - "\n\nAssistant: ..." for assistant turns

    Args:
        text: Raw conversation string

    Returns:
        List of dicts with 'role' and 'content' keys
    """
    # Split by "\n\n" to get segments
    segments = text.split("\n\n")

    turns = []
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue

        if segment.startswith("Human:"):
            turns.append({
                "role": "human",
                "content": segment[6:].strip()  # Remove "Human:"
            })
        elif segment.startswith("Assistant:"):
            turns.append({
                "role": "assistant",
                "content": segment[10:].strip()  # Remove "Assistant:"
            })

    return turns


def extract_preference_triple(chosen: str, rejected: str) -> Dict[str, str]:
    """
    Extract (prompt, chosen_response, rejected_response) from HH format.

    The format has multi-turn conversations where the last turn differs
    between chosen and rejected. We extract:
    - prompt: All turns up to (but not including) the final assistant turn
    - chosen_response: The final assistant turn in the chosen conversation
    - rejected_response: The final assistant turn in the rejected conversation

    Args:
        chosen: Chosen conversation string
        rejected: Rejected conversation string

    Returns:
        Dict with 'prompt', 'chosen', 'rejected' keys
    """
    chosen_turns = parse_conversation(chosen)
    rejected_turns = parse_conversation(rejected)

    # Validate format
    if not chosen_turns or not rejected_turns:
        raise ValueError("Empty conversation")

    if chosen_turns[-1]["role"] != "assistant" or rejected_turns[-1]["role"] != "assistant":
        raise ValueError("Last turn must be assistant")

    # Extract final assistant responses
    chosen_response = chosen_turns[-1]["content"]
    rejected_response = rejected_turns[-1]["content"]

    # Build prompt from all turns except the last one
    # We'll use the chosen conversation for the prompt (they should be identical except for the last turn)
    prompt_turns = chosen_turns[:-1]

    # Format prompt as conversation
    prompt_parts = []
    for turn in prompt_turns:
        if turn["role"] == "human":
            prompt_parts.append(f"Human: {turn['content']}")
        else:
            prompt_parts.append(f"Assistant: {turn['content']}")

    prompt = "\n\n".join(prompt_parts)

    return {
        "prompt": prompt,
        "chosen": chosen_response,
        "rejected": rejected_response,
    }


def preprocess_hh_dataset(
    dataset: Dataset,
    num_proc: int = 4,
) -> Dataset:
    """
    Preprocess Anthropic HH dataset to extract preference triples.

    Args:
        dataset: Raw HH dataset with 'chosen' and 'rejected' fields
        num_proc: Number of processes for parallel processing

    Returns:
        Processed dataset with 'prompt', 'chosen', 'rejected' fields
    """

    def process_example(example):
        try:
            return extract_preference_triple(example["chosen"], example["rejected"])
        except Exception as e:
            # If parsing fails, return None (will be filtered out)
            print(f"Warning: Failed to parse example: {e}")
            return None

    # Apply preprocessing
    processed = dataset.map(
        process_example,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc="Preprocessing HH data",
    )

    # Filter out None values (failed parses)
    processed = processed.filter(lambda x: x["prompt"] is not None)

    return processed


def create_splits(
    dataset: Dataset,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Split dataset into train/val/test.

    Args:
        dataset: Full dataset
        val_ratio: Fraction for validation
        test_ratio: Fraction for test
        seed: Random seed

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # First split off test set
    split = dataset.train_test_split(test_size=test_ratio, seed=seed)
    test_dataset = split["test"]
    remaining = split["train"]

    # Split remaining into train and val
    val_size = val_ratio / (1 - test_ratio)  # Adjust ratio for remaining data
    split = remaining.train_test_split(test_size=val_size, seed=seed)
    train_dataset = split["train"]
    val_dataset = split["test"]

    return train_dataset, val_dataset, test_dataset


def format_prompt_for_generation(prompt: str) -> str:
    """
    Format a prompt for generation (add trailing Assistant: marker).

    Args:
        prompt: Conversation prompt

    Returns:
        Formatted prompt ready for generation
    """
    if not prompt.strip().endswith("Human:"):
        # Add final Human turn marker if not present
        # This ensures the model generates an Assistant response
        return prompt + "\n\nAssistant:"
    else:
        return prompt + "\n\nAssistant:"


def debug_sample(dataset: Dataset, num_samples: int = 3):
    """
    Print sample examples from the dataset for debugging.

    Args:
        dataset: Dataset to sample from
        num_samples: Number of samples to print
    """
    print(f"\n{'='*80}")
    print(f"Dataset Debug: Showing {num_samples} samples")
    print(f"{'='*80}\n")

    for i in range(min(num_samples, len(dataset))):
        example = dataset[i]
        print(f"Example {i + 1}:")
        print(f"  Prompt: {example['prompt'][:200]}...")
        print(f"  Chosen: {example['chosen'][:200]}...")
        print(f"  Rejected: {example['rejected'][:200]}...")
        print()
