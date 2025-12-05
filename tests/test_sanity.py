"""Sanity checks for the DPO training setup."""

import sys
import os
from pathlib import Path

# Add project root to path (handle both local and Colab environments)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Also try common Colab paths
if '/content/dpo' not in sys.path:
    sys.path.insert(0, '/content/dpo')

# Change to project directory
os.chdir(project_root)

import torch
from transformers import AutoTokenizer
from datasets import load_dataset

from src.data.preprocessing import preprocess_hh_dataset, extract_preference_triple
from src.data.dataset import PreferenceDataset, SFTDataset
from src.data.collator import PreferenceDataCollator, SFTDataCollator
from src.models.policy import PolicyModel
from src.models.reference import ReferenceModel
from src.losses.dpo_loss import dpo_loss


def test_data_loading():
    """Test that data loads and preprocesses correctly."""
    print("\n" + "="*80)
    print("TEST 1: Data Loading")
    print("="*80)

    # Load small sample
    raw_dataset = load_dataset("Anthropic/hh-rlhf", split="train[:10]")
    print(f"✓ Loaded {len(raw_dataset)} raw examples")

    # Test preprocessing
    processed = preprocess_hh_dataset(raw_dataset, num_proc=1)
    print(f"✓ Preprocessed to {len(processed)} examples")

    # Check format
    example = processed[0]
    assert "prompt" in example
    assert "chosen" in example
    assert "rejected" in example
    print(f"✓ Data format is correct")

    # Show sample
    print("\nSample example:")
    print(f"  Prompt: {example['prompt'][:100]}...")
    print(f"  Chosen: {example['chosen'][:100]}...")
    print(f"  Rejected: {example['rejected'][:100]}...")

    print("\n✅ Data loading test PASSED")


def test_tokenization():
    """Test that tokenization works correctly."""
    print("\n" + "="*80)
    print("TEST 2: Tokenization")
    print("="*80)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✓ Loaded tokenizer")

    # Load small dataset
    raw_dataset = load_dataset("Anthropic/hh-rlhf", split="train[:5]")
    processed = preprocess_hh_dataset(raw_dataset, num_proc=1)

    # Create preference dataset
    dataset = PreferenceDataset(processed, tokenizer, max_length=128, max_prompt_length=64)
    print(f"✓ Created PreferenceDataset with {len(dataset)} examples")

    # Get sample
    sample = dataset[0]
    print(f"✓ Sample shapes:")
    print(f"  chosen_input_ids: {sample['chosen_input_ids'].shape}")
    print(f"  rejected_input_ids: {sample['rejected_input_ids'].shape}")

    # Check that labels mask prompt tokens
    assert (sample['chosen_labels'][:10] == -100).any(), "Labels should mask prompt"
    print("✓ Labels correctly mask prompt tokens")

    print("\n✅ Tokenization test PASSED")


def test_model_forward_pass():
    """Test that model forward pass works."""
    print("\n" + "="*80)
    print("TEST 3: Model Forward Pass")
    print("="*80)

    # Load small model
    model = PolicyModel("gpt2", use_gradient_checkpointing=False, torch_dtype="float32")
    print("✓ Loaded model")

    # Create dummy batch
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(0, 1000, (batch_size, seq_len))
    labels[:, :3] = -100  # Mask first 3 tokens

    # Forward pass
    outputs = model(input_ids, attention_mask, labels)
    print(f"✓ Forward pass successful, loss: {outputs.loss.item():.4f}")

    # Test get_logprobs
    logprobs = model.get_logprobs(input_ids, attention_mask, labels)
    print(f"✓ get_logprobs successful, shape: {logprobs.shape}")
    assert logprobs.shape == (batch_size,), "Log probs should be per-sequence"

    print("\n✅ Model forward pass test PASSED")


def test_dpo_loss():
    """Test DPO loss computation."""
    print("\n" + "="*80)
    print("TEST 4: DPO Loss")
    print("="*80)

    # Create dummy log probs
    batch_size = 4
    policy_chosen = torch.randn(batch_size)
    policy_rejected = torch.randn(batch_size)
    ref_chosen = torch.randn(batch_size)
    ref_rejected = torch.randn(batch_size)

    # Make chosen better than rejected (on average)
    policy_chosen += 1.0

    # Compute loss
    loss, metrics = dpo_loss(
        policy_chosen,
        policy_rejected,
        ref_chosen,
        ref_rejected,
        beta=0.1,
    )

    print(f"✓ Loss: {loss.item():.4f}")
    print(f"✓ Accuracy: {metrics['accuracy'].item():.2%}")
    print(f"✓ Reward margin: {metrics['reward_margin'].item():.4f}")

    assert torch.isfinite(loss), "Loss should be finite"
    assert loss.item() >= 0, "Loss should be non-negative"
    print("✓ Loss is finite and non-negative")

    print("\n✅ DPO loss test PASSED")


def test_training_step():
    """Test a single training step."""
    print("\n" + "="*80)
    print("TEST 5: Training Step")
    print("="*80)

    # Load tokenizer and data
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_dataset = load_dataset("Anthropic/hh-rlhf", split="train[:5]")
    processed = preprocess_hh_dataset(raw_dataset, num_proc=1)
    dataset = PreferenceDataset(processed, tokenizer, max_length=128)

    # Create batch
    collator = PreferenceDataCollator(pad_token_id=tokenizer.pad_token_id)
    batch = collator([dataset[0], dataset[1]])

    # Load models
    policy = PolicyModel("gpt2", use_gradient_checkpointing=False, torch_dtype="float32")
    reference = ReferenceModel(policy)
    print("✓ Loaded policy and reference models")

    # Get log probs
    policy_chosen_logps = policy.get_logprobs(
        batch["chosen_input_ids"],
        batch["chosen_attention_mask"],
        batch["chosen_labels"],
    )
    ref_chosen_logps = reference.get_logprobs(
        batch["chosen_input_ids"],
        batch["chosen_attention_mask"],
        batch["chosen_labels"],
    )
    print("✓ Computed log probabilities")

    # Compute loss
    policy_rejected_logps = policy.get_logprobs(
        batch["rejected_input_ids"],
        batch["rejected_attention_mask"],
        batch["rejected_labels"],
    )
    ref_rejected_logps = reference.get_logprobs(
        batch["rejected_input_ids"],
        batch["rejected_attention_mask"],
        batch["rejected_labels"],
    )

    loss, metrics = dpo_loss(
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        beta=0.1,
    )
    print(f"✓ Loss: {loss.item():.4f}")

    # Backward pass
    loss.backward()
    print("✓ Backward pass successful")

    # Check gradients
    has_grads = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in policy.parameters() if p.requires_grad)
    assert has_grads, "Model should have gradients"
    print("✓ Gradients computed successfully")

    print("\n✅ Training step test PASSED")


def test_checkpointing():
    """Test model saving and loading."""
    print("\n" + "="*80)
    print("TEST 6: Checkpointing")
    print("="*80)

    import tempfile
    import shutil

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    print(f"✓ Created temp directory: {temp_dir}")

    try:
        # Load and save model
        model = PolicyModel("gpt2", use_gradient_checkpointing=False, torch_dtype="float32")
        model.save_pretrained(temp_dir)
        print("✓ Saved model")

        # Load model
        loaded_model = PolicyModel.from_pretrained(temp_dir)
        print("✓ Loaded model")

        # Test that loaded model works
        input_ids = torch.randint(0, 1000, (1, 10))
        attention_mask = torch.ones(1, 10)
        labels = torch.randint(0, 1000, (1, 10))

        outputs = loaded_model(input_ids, attention_mask, labels)
        print(f"✓ Loaded model works, loss: {outputs.loss.item():.4f}")

        print("\n✅ Checkpointing test PASSED")

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        print("✓ Cleaned up temp directory")


def run_all_tests():
    """Run all sanity checks."""
    print("\n" + "="*80)
    print("DPO TRAINING SANITY CHECKS")
    print("="*80)

    tests = [
        test_data_loading,
        test_tokenization,
        test_model_forward_pass,
        test_dpo_loss,
        test_training_step,
        test_checkpointing,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n❌ Test FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n🎉 All tests PASSED!")
    else:
        print(f"\n⚠️  {failed} test(s) FAILED")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
