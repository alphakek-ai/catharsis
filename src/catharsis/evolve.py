"""Evolutionary search over LoRA perturbations."""

import time

import torch
from torch import Tensor

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

    gen_start = time.perf_counter()

    for gen in range(generations):
        gen_t0 = time.perf_counter()
        elapsed_total = gen_t0 - gen_start

        print(f"\n{'='*60}")
        print(f"Generation {gen + 1}/{generations}")
        print(f"  Best so far: compliance={best_compliance}%, KL={best_kl:.4f}, score={best_score:.4f}")
        if gen > 0:
            avg_gen = elapsed_total / gen
            eta = avg_gen * (generations - gen)
            print(f"  Elapsed: {_fmt(elapsed_total)} | ETA: {_fmt(eta)} | Avg/gen: {_fmt(avg_gen)}")
        print(f"{'='*60}")

        # Generate perturbations
        candidates = []
        for i in range(population_size):
            if gen == 0 and i == 0:
                noise = torch.zeros(n_params, device=device)
            else:
                noise = torch.randn(n_params, device=device) * noise_std
            candidates.append(best_params + noise)

        scores = []
        for i, candidate in enumerate(candidates):
            cand_t0 = time.perf_counter()
            print(f"\n  Candidate {i + 1}/{population_size}")

            # Apply candidate LoRA weights
            model.set_lora_from_flat(candidate)

            # Generate responses to bad prompts
            t0 = time.perf_counter()
            responses = model.generate_responses(bad_prompts, max_new_tokens=max_new_tokens, batch_size=batch_size)
            t_gen = time.perf_counter() - t0
            print(f"    Generate: {t_gen:.1f}s ({len(bad_prompts)} prompts, {len(bad_prompts)/t_gen:.1f} prompts/s)")

            # Judge compliance
            t0 = time.perf_counter()
            refusals, verdicts = judge.evaluate(bad_prompts, responses)
            t_judge = time.perf_counter() - t0
            compliance_rate = 1.0 - (refusals / len(bad_prompts))
            print(f"    Judge:    {t_judge:.1f}s ({refusals} refusals, {len(bad_prompts)-refusals} compliant)")

            # Compute KL divergence on good prompts
            t0 = time.perf_counter()
            kl = model.compute_kl(good_prompts, base_logprobs, batch_size=batch_size)
            t_kl = time.perf_counter() - t0
            print(f"    KL:       {t_kl:.1f}s (KL={kl:.4f})")

            score = compliance_rate - kl_weight * kl
            scores.append(score)

            cand_total = time.perf_counter() - cand_t0
            print(f"    Total:    {cand_total:.1f}s | Score: {score:.4f}")

        # Select the best candidate
        # Track per-candidate stats for reporting
        best_idx = max(range(len(scores)), key=lambda idx: scores[idx])
        if scores[best_idx] > best_score:
            best_score = scores[best_idx]
            best_params = candidates[best_idx].clone()
            model.set_lora_from_flat(best_params)
            # Recover compliance/KL from the score: score = compliance - kl_weight * kl
            # We stored them during evaluation, so just recompute KL (cheap)
            best_kl = model.compute_kl(good_prompts, base_logprobs, batch_size=batch_size)
            best_compliance = round((best_score + kl_weight * best_kl) * 100)

            print(f"\n  >>> New best! compliance={best_compliance}%, KL={best_kl:.4f}")
        else:
            print(f"\n  No improvement this generation.")

        gen_total = time.perf_counter() - gen_t0
        print(f"\n  Generation time: {_fmt(gen_total)}")

    total = time.perf_counter() - gen_start
    print(f"\nEvolution complete. Total time: {_fmt(total)}")
    model.set_lora_from_flat(best_params)
    return best_params


def _fmt(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"
