"""Evolutionary search over LoRA perturbations."""

import time

import torch
from torch import Tensor
from tqdm import tqdm

from .judge import Judge
from .log import log
from .model import Model


def _fmt(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


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
    n_params = model.lora_param_count()
    device = model.device

    best_params = model.get_lora_flat().clone()
    best_score = float("-inf")
    best_compliance = 0
    best_kl = float("inf")

    # Total work units: each candidate processes len(bad_prompts) prompts (generate+judge)
    total_prompts = generations * population_size * len(bad_prompts)
    pbar = tqdm(total=total_prompts, desc="Gen 1 | Cand 1 | best=0%", unit="prompt")

    gen_start = time.perf_counter()

    for gen in range(generations):
        gen_t0 = time.perf_counter()

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
            pbar.set_description(
                f"Gen {gen + 1}/{generations} | Cand {i + 1}/{population_size} | best={best_compliance}%"
            )

            model.set_lora_from_flat(candidate)

            # Generate + judge in pipeline
            t0 = time.perf_counter()
            gen_iter = model.generate_responses_iter(bad_prompts, max_new_tokens=max_new_tokens, batch_size=batch_size)
            refusals, verdicts, responses = judge.evaluate_streaming(gen_iter, total=len(bad_prompts), pbar=pbar)
            t_gen_judge = time.perf_counter() - t0
            compliance_rate = 1.0 - (refusals / len(bad_prompts))

            # Compute KL
            t0 = time.perf_counter()
            kl = model.compute_kl(good_prompts, base_logprobs, batch_size=batch_size)
            t_kl = time.perf_counter() - t0

            score = compliance_rate - kl_weight * kl
            scores.append(score)

            cand_total = time.perf_counter() - cand_t0

            log.info(
                "candidate_eval",
                generation=gen + 1,
                candidate=i + 1,
                population=population_size,
                compliance=round(compliance_rate * 100),
                refusals=refusals,
                kl=round(kl, 4),
                score=round(score, 4),
                t_gen_judge=f"{t_gen_judge:.1f}s",
                t_kl=f"{t_kl:.1f}s",
                t_total=f"{cand_total:.1f}s",
            )

        # Select best
        best_idx = max(range(len(scores)), key=lambda idx: scores[idx])
        if scores[best_idx] > best_score:
            best_score = scores[best_idx]
            best_params = candidates[best_idx].clone()
            model.set_lora_from_flat(best_params)
            best_kl = model.compute_kl(good_prompts, base_logprobs, batch_size=batch_size)
            best_compliance = round((best_score + kl_weight * best_kl) * 100)
            log.info("new_best", compliance=best_compliance, kl=round(best_kl, 4), score=round(best_score, 4))
        else:
            log.info("no_improvement", generation=gen + 1)

        gen_total = time.perf_counter() - gen_t0
        elapsed_total = time.perf_counter() - gen_start
        eta = _fmt((elapsed_total / (gen + 1)) * (generations - gen - 1))
        log.info("generation_done", generation=gen + 1, time=_fmt(gen_total), eta=eta)

    pbar.close()
    total = time.perf_counter() - gen_start
    log.info("evolution_complete", total_time=_fmt(total), best_compliance=best_compliance, best_kl=round(best_kl, 4))
    model.set_lora_from_flat(best_params)
    return best_params
