"""Tests for batched per-sample LoRA."""

import pytest
import torch
from torch import nn

from catharsis.batched_lora import BatchedLoRAContext


@pytest.fixture
def simple_model():
    """A tiny model with two linear layers for testing."""
    model = nn.Sequential(
        nn.Linear(8, 16, bias=False),
        nn.ReLU(),
        nn.Linear(16, 4, bias=False),
    )
    model.eval()
    return model


def test_batched_lora_applies_different_corrections(simple_model):
    """Verify that different candidates get different outputs."""
    n_candidates = 3
    batch_per_candidate = 2
    total_batch = n_candidates * batch_per_candidate

    x = torch.randn(total_batch, 1, 8)
    candidate_ids = torch.repeat_interleave(torch.arange(n_candidates), batch_per_candidate)

    # Create different LoRA params for each candidate
    # For the first linear (8 -> 16): A is (n_cand, 1, 8), B is (n_cand, 16, 1)
    A0 = torch.randn(n_candidates, 1, 8) * 0.1
    B0 = torch.randn(n_candidates, 16, 1) * 0.1

    module_lora_params = {"0": (A0, B0)}

    # Output without LoRA
    with torch.no_grad():
        base_output = simple_model(x)

    # Output with batched LoRA
    with torch.no_grad():
        with BatchedLoRAContext(simple_model, module_lora_params, candidate_ids):
            lora_output = simple_model(x)

    # Outputs should differ from base
    assert not torch.allclose(base_output, lora_output, atol=1e-6), "LoRA should change the output"

    # Different candidates should get different corrections
    out_c0 = lora_output[:batch_per_candidate]
    out_c1 = lora_output[batch_per_candidate : 2 * batch_per_candidate]
    out_c2 = lora_output[2 * batch_per_candidate :]

    # At least some candidates should differ
    assert not torch.allclose(out_c0, out_c1, atol=1e-6) or not torch.allclose(out_c1, out_c2, atol=1e-6), (
        "Different candidates should produce different outputs"
    )


def test_batched_lora_zero_params_equal_base(simple_model):
    """Zero LoRA params should produce the same output as base model."""
    n_candidates = 2
    batch_per_candidate = 3
    total_batch = n_candidates * batch_per_candidate

    x = torch.randn(total_batch, 1, 8)
    candidate_ids = torch.repeat_interleave(torch.arange(n_candidates), batch_per_candidate)

    # Zero LoRA params
    A0 = torch.zeros(n_candidates, 1, 8)
    B0 = torch.zeros(n_candidates, 16, 1)
    module_lora_params = {"0": (A0, B0)}

    with torch.no_grad():
        base_output = simple_model(x)

    with torch.no_grad():
        with BatchedLoRAContext(simple_model, module_lora_params, candidate_ids):
            lora_output = simple_model(x)

    assert torch.allclose(base_output, lora_output, atol=1e-6), "Zero LoRA should equal base output"


def test_batched_lora_hooks_cleaned_up(simple_model):
    """Hooks should be removed after context exits."""
    x = torch.randn(2, 1, 8)
    candidate_ids = torch.tensor([0, 0])
    A0 = torch.randn(1, 1, 8)
    B0 = torch.randn(1, 16, 1)
    module_lora_params = {"0": (A0, B0)}

    with torch.no_grad():
        base_before = simple_model(x).clone()

    with BatchedLoRAContext(simple_model, module_lora_params, candidate_ids):
        pass  # hooks active here

    with torch.no_grad():
        base_after = simple_model(x)

    assert torch.allclose(base_before, base_after, atol=1e-6), "Hooks should be cleaned up after context exit"


def test_batched_lora_dtype_mismatch(simple_model):
    """LoRA params in float32 should work with bfloat16 model."""
    simple_model = simple_model.to(torch.bfloat16)
    x = torch.randn(2, 1, 8, dtype=torch.bfloat16)
    candidate_ids = torch.tensor([0, 1])

    # LoRA params in float32 (as they would be from the optimizer)
    A0 = torch.randn(2, 1, 8, dtype=torch.float32) * 0.01
    B0 = torch.randn(2, 16, 1, dtype=torch.float32) * 0.01
    module_lora_params = {"0": (A0, B0)}

    with torch.no_grad():
        with BatchedLoRAContext(simple_model, module_lora_params, candidate_ids):
            output = simple_model(x)

    assert output.dtype == torch.bfloat16, "Output should match model dtype"
    assert output.shape == (2, 1, 4)
