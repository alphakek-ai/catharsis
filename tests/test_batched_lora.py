"""Tests for batched per-sample noise hooks."""

import torch
from torch import nn

from catharsis.batched_lora import BatchedNoiseContext


def _simple_model():
    model = nn.Sequential(nn.Linear(8, 16, bias=False), nn.ReLU(), nn.Linear(16, 4, bias=False))
    model.eval()
    return model


def test_noise_applies_different_corrections():
    model = _simple_model()
    n_candidates = 3
    batch_per_candidate = 2
    total_batch = n_candidates * batch_per_candidate

    x = torch.randn(total_batch, 1, 8)
    candidate_ids = torch.repeat_interleave(torch.arange(n_candidates), batch_per_candidate)

    A0 = torch.randn(n_candidates, 1, 8) * 0.1
    B0 = torch.randn(n_candidates, 16, 1) * 0.1
    module_noise_params = {"0": (A0, B0)}

    with torch.no_grad():
        base_output = model(x)

    with torch.no_grad():
        with BatchedNoiseContext(model, module_noise_params, candidate_ids):
            noise_output = model(x)

    assert not torch.allclose(base_output, noise_output, atol=1e-6)


def test_zero_noise_equals_base():
    model = _simple_model()
    x = torch.randn(4, 1, 8)
    candidate_ids = torch.tensor([0, 0, 1, 1])

    A0 = torch.zeros(2, 1, 8)
    B0 = torch.zeros(2, 16, 1)
    module_noise_params = {"0": (A0, B0)}

    with torch.no_grad():
        base_output = model(x)

    with torch.no_grad():
        with BatchedNoiseContext(model, module_noise_params, candidate_ids):
            noise_output = model(x)

    assert torch.allclose(base_output, noise_output, atol=1e-6)


def test_hooks_cleaned_up():
    model = _simple_model()
    x = torch.randn(2, 1, 8)
    candidate_ids = torch.tensor([0, 0])

    A0 = torch.randn(1, 1, 8)
    B0 = torch.randn(1, 16, 1)
    module_noise_params = {"0": (A0, B0)}

    with torch.no_grad():
        base_before = model(x).clone()

    with BatchedNoiseContext(model, module_noise_params, candidate_ids):
        pass

    with torch.no_grad():
        base_after = model(x)

    assert torch.allclose(base_before, base_after, atol=1e-6)


def test_dtype_mismatch():
    model = _simple_model().to(torch.bfloat16)
    x = torch.randn(2, 1, 8, dtype=torch.bfloat16)
    candidate_ids = torch.tensor([0, 1])

    A0 = torch.randn(2, 1, 8, dtype=torch.float32) * 0.01
    B0 = torch.randn(2, 16, 1, dtype=torch.float32) * 0.01
    module_noise_params = {"0": (A0, B0)}

    with torch.no_grad():
        with BatchedNoiseContext(model, module_noise_params, candidate_ids):
            output = model(x)

    assert output.dtype == torch.bfloat16
    assert output.shape == (2, 1, 4)
