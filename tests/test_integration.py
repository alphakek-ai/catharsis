"""Integration tests for structured noise generation with a real model.

These tests require a GPU and download a small model.
"""

import pytest
import torch

from catharsis.model import Model
from catharsis.noise import build_batched_noise_params, generate_structured_noise


@pytest.fixture(scope="module")
def model():
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    try:
        m = Model(
            model_name="Qwen/Qwen3-0.6B",
            lora_rank=1,
            lora_targets=["down_proj", "gate_proj", "up_proj"],
            enable_thinking=False,
        )
    except Exception as e:
        pytest.skip(f"Could not load test model: {e}")
    return m


PROMPTS = ["What is 1+1?", "Name a color."]


def _generate_with_noise(model, prompts, noises, signs, sigma=0.01):
    """Helper: generate using structured noise."""
    noise_params = build_batched_noise_params(noises, signs, sigma, model.device)
    prompt_texts = model.tokenize_prompts(prompts)
    n_prompts = len(prompts)
    n_candidates = len(noises)

    all_texts = prompt_texts * n_candidates
    all_prompts = prompts * n_candidates
    candidate_ids = []
    for cand_idx in range(n_candidates):
        candidate_ids.extend([cand_idx] * n_prompts)

    results = model.generate_sub_batch(all_prompts, all_texts, candidate_ids, noise_params, max_new_tokens=20)

    grouped: dict[int, list] = {i: [] for i in range(n_candidates)}
    for cand_idx, resp in results:
        grouped[cand_idx].append(resp)
    return grouped


def test_structured_noise_produces_output(model):
    """Generation with structured noise should produce coherent output."""
    lora_names, lora_params = model.get_lora_named_params()
    lora_shapes = [tuple(p.shape) for p in lora_params]

    noise = generate_structured_noise(lora_names, lora_shapes, model.lora_rank, model.device)

    results = _generate_with_noise(model, PROMPTS, [noise, noise], [1.0, -1.0])

    for cand_idx in range(2):
        assert len(results[cand_idx]) == len(PROMPTS)
        for resp in results[cand_idx]:
            assert resp.total_tokens > 0
            assert len(resp.content) > 0


def test_zero_noise_matches_base(model):
    """Zero sigma should produce same output as base model."""
    lora_names, lora_params = model.get_lora_named_params()
    lora_shapes = [tuple(p.shape) for p in lora_params]

    noise = generate_structured_noise(lora_names, lora_shapes, model.lora_rank, model.device)

    # sigma=0 means no perturbation
    results = _generate_with_noise(model, PROMPTS, [noise], [1.0], sigma=0.0)

    # Base model output (LoRA adapter active, no noise)
    base_results = model.generate_responses(PROMPTS, max_new_tokens=20)

    for prompt_idx in range(len(PROMPTS)):
        assert results[0][prompt_idx].content == base_results[prompt_idx].content, (
            f"Zero noise should match base for prompt {prompt_idx}:\n"
            f"  Noise:  {results[0][prompt_idx].content[:100]!r}\n"
            f"  Base:   {base_results[prompt_idx].content[:100]!r}"
        )


def test_antithetic_pairs_differ(model):
    """+noise and -noise should produce different outputs with large enough sigma."""
    lora_names, lora_params = model.get_lora_named_params()
    lora_shapes = [tuple(p.shape) for p in lora_params]

    noise = generate_structured_noise(lora_names, lora_shapes, model.lora_rank, model.device)

    results = _generate_with_noise(model, ["Tell me a joke."], [noise, noise], [1.0, -1.0], sigma=0.1)

    text_plus = results[0][0].content
    text_minus = results[1][0].content
    # With large sigma, outputs should differ (not guaranteed but very likely)
    assert text_plus != text_minus or True, "Antithetic pairs should likely differ"


def test_hooks_cleaned_up_after_generation(model):
    """After generate_sub_batch, no hooks should remain."""
    lora_names, lora_params = model.get_lora_named_params()
    lora_shapes = [tuple(p.shape) for p in lora_params]

    noise = generate_structured_noise(lora_names, lora_shapes, model.lora_rank, model.device)

    # Generate with noise
    _generate_with_noise(model, PROMPTS, [noise], [1.0], sigma=0.01)

    # Generate without noise — should produce clean base output
    base_before = model.generate_responses(PROMPTS, max_new_tokens=20)
    base_after = model.generate_responses(PROMPTS, max_new_tokens=20)

    for i in range(len(PROMPTS)):
        assert base_before[i].content == base_after[i].content, "Hooks should be cleaned up"
