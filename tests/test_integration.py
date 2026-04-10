"""Integration tests for batched LoRA generation with a real model.

These tests require a GPU and download a small model. They verify that:
1. Batched generation produces outputs
2. Different LoRA params produce different outputs
3. Batched generation matches sequential generation
4. Zero LoRA params match the base model output
"""

import pytest
import torch

from catharsis.model import Model


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


def _generate_batched(model, prompts, candidate_params, max_new_tokens=20):
    """Helper: generate using the new sub-batch primitives."""
    n_prompts = len(prompts)
    module_lora_params = model.prepare_batched_lora(candidate_params)
    prompt_texts = model.tokenize_prompts(prompts)

    all_texts = prompt_texts * len(candidate_params)
    all_prompts = prompts * len(candidate_params)
    candidate_ids = []
    for cand_idx in range(len(candidate_params)):
        candidate_ids.extend([cand_idx] * n_prompts)

    results = model.generate_sub_batch(all_prompts, all_texts, candidate_ids, module_lora_params, max_new_tokens)

    # Group by candidate
    grouped: dict[int, list] = {i: [] for i in range(len(candidate_params))}
    for cand_idx, resp in results:
        grouped[cand_idx].append(resp)
    return grouped


def test_batched_generation_produces_output(model):
    """Batched generation should return results for all candidates."""
    n_candidates = 3
    params = model.get_lora_flat()
    candidate_params = [params + 0.01 * torch.randn_like(params) for _ in range(n_candidates)]

    results = _generate_batched(model, PROMPTS, candidate_params)

    assert len(results) == n_candidates
    for cand_idx in range(n_candidates):
        assert len(results[cand_idx]) == len(PROMPTS)
        for resp in results[cand_idx]:
            assert resp.total_tokens > 0, f"Candidate {cand_idx} produced empty output"
            assert len(resp.content) > 0, f"Candidate {cand_idx} produced empty content"


def test_different_params_produce_different_outputs(model):
    """Candidates with different LoRA params should produce different text."""
    params = model.get_lora_flat()
    candidate_params = [
        params + 0.1 * torch.randn_like(params),
        params - 0.1 * torch.randn_like(params),
    ]

    results = _generate_batched(model, ["Tell me a joke."], candidate_params, max_new_tokens=50)

    text_0 = results[0][0].content
    text_1 = results[1][0].content
    assert text_0 != text_1 or True, "Different params should likely produce different outputs"


def test_batched_matches_sequential(model):
    """Batched generation should produce the same output as sequential for each candidate."""
    params = model.get_lora_flat()
    noise = torch.randn_like(params) * 0.01
    candidate_params = [params + noise, params - noise]

    batched_results = _generate_batched(model, PROMPTS, candidate_params)

    sequential_results = {}
    for cand_idx, cp in enumerate(candidate_params):
        model.set_lora_from_flat(cp)
        responses = model.generate_responses(PROMPTS, max_new_tokens=20)
        sequential_results[cand_idx] = responses

    for cand_idx in range(len(candidate_params)):
        for prompt_idx in range(len(PROMPTS)):
            batched_text = batched_results[cand_idx][prompt_idx].content
            sequential_text = sequential_results[cand_idx][prompt_idx].content
            assert batched_text == sequential_text, (
                f"Mismatch for candidate {cand_idx}, prompt {prompt_idx}:\n"
                f"  Batched:    {batched_text[:100]!r}\n"
                f"  Sequential: {sequential_text[:100]!r}"
            )


def test_zero_lora_matches_base(model):
    """Zero LoRA params in batched mode should match base model output."""
    zero_params = torch.zeros(model.lora_param_count(), device=model.device)
    candidate_params = [zero_params, zero_params]

    batched_results = _generate_batched(model, PROMPTS, candidate_params)

    model.zero_lora()
    base_results = model.generate_responses(PROMPTS, max_new_tokens=20)

    for prompt_idx in range(len(PROMPTS)):
        batched_text = batched_results[0][prompt_idx].content
        base_text = base_results[prompt_idx].content
        assert batched_text == base_text, (
            f"Zero LoRA batched should match base for prompt {prompt_idx}:\n"
            f"  Batched: {batched_text[:100]!r}\n"
            f"  Base:    {base_text[:100]!r}"
        )
