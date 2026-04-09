"""Evolutionary search over LoRA perturbations."""

import torch
from torch import Tensor
from tqdm import tqdm

from .judge import Judge
from .model import Model


def evolve(
    model: Model,
    judge: Judge,
    good_prompts: list[str],
    bad_prompts: list[str],
    base_logprobs: Tensor,
    population_size: int = 16,
    generations: int = 100,
    noise_std: float = 0.1,
    kl_weight: float = 1.0,
    batch_size: int = 32,
    max_new_tokens: int = 1000,
) -> Tensor:
    """
    Evolutionary search over LoRA parameters.

    For each generation:
    1. Sample `population_size` perturbations around the current best
    2. For each perturbation, generate responses to bad prompts
    3. Judge compliance with LLM judge
    4. Compute KL divergence on good prompts
    5. Score = compliance_rate - kl_weight * kl_divergence
    6. Select the best, repeat

    Returns the best LoRA parameter vector found.
    """
    n_params = model.lora_param_count()
    device = model.device

    # Start from current LoRA state (could be zero or warm-started)
    best_params = model.get_lora_flat().clone()
    best_score = float("-inf")
    best_compliance = 0
    best_kl = float("inf")

    for gen in range(generations):
        print(f"\n{'='*60}")
        print(f"Generation {gen + 1}/{generations}")
        print(f"  Best so far: compliance={best_compliance}%, KL={best_kl:.4f}, score={best_score:.4f}")
        print(f"{'='*60}")

        # Generate perturbations
        candidates = []
        for i in range(population_size):
            if gen == 0 and i == 0:
                # Always evaluate the starting point
                noise = torch.zeros(n_params, device=device)
            else:
                noise = torch.randn(n_params, device=device) * noise_std
            candidates.append(best_params + noise)

        scores = []
        for i, candidate in enumerate(candidates):
            print(f"\n  Candidate {i + 1}/{population_size}")

            # Apply candidate LoRA weights
            model.set_lora_from_flat(candidate)

            # Generate responses to bad prompts
            print("    Generating responses...")
            responses = model.generate_responses(bad_prompts, max_new_tokens=max_new_tokens, batch_size=batch_size)

            # Judge compliance
            refusals, verdicts = judge.evaluate(bad_prompts, responses)
            compliance_rate = 1.0 - (refusals / len(bad_prompts))

            # Compute KL divergence on good prompts
            print("    Computing KL divergence...")
            kl = model.compute_kl(good_prompts, base_logprobs, batch_size=batch_size)

            score = compliance_rate - kl_weight * kl
            scores.append(score)

            print(f"    Compliance: {compliance_rate * 100:.0f}%, KL: {kl:.4f}, Score: {score:.4f}")

        # Select the best candidate
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        if scores[best_idx] > best_score:
            best_score = scores[best_idx]
            best_params = candidates[best_idx].clone()

            # Recompute stats for display
            model.set_lora_from_flat(best_params)
            refusals, _ = judge.evaluate(bad_prompts, model.generate_responses(bad_prompts, max_new_tokens=max_new_tokens, batch_size=batch_size))
            best_compliance = int((1.0 - refusals / len(bad_prompts)) * 100)
            best_kl = model.compute_kl(good_prompts, base_logprobs, batch_size=batch_size)

            print(f"\n  >>> New best! compliance={best_compliance}%, KL={best_kl:.4f}")
        else:
            print(f"\n  No improvement this generation.")

    # Apply best params
    model.set_lora_from_flat(best_params)
    return best_params
