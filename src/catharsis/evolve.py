"""Evolutionary search over LoRA perturbations."""

import asyncio
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

        # Phase 1: GPU work (sequential) — generate responses + compute KL per candidate
        # Judge calls are fired off in background as responses come in
        candidate_data: list[dict] = []
        all_judge_tasks: list[list[asyncio.Task[bool]]] = []

        gpu_bar = tqdm(
            total=population_size,
            desc=f"Gen {gen + 1}/{generations} GPU | best={best_compliance}%",
            unit="cand",
            leave=False,
        )

        for candidate in candidates:
            model.set_lora_from_flat(candidate)

            # Generate responses
            t0 = time.perf_counter()
            responses = model.generate_responses(bad_prompts, max_new_tokens=max_new_tokens, batch_size=batch_size)
            t_gen = time.perf_counter() - t0

            # Compute KL while this candidate's weights are still loaded
            t0 = time.perf_counter()
            kl = model.compute_kl(good_prompts, base_logprobs, batch_size=batch_size)
            t_kl = time.perf_counter() - t0

            # Fire off judge calls (non-blocking)
            tasks = [judge.submit(p, r) for p, r in zip(bad_prompts, responses, strict=True)]
            all_judge_tasks.append(tasks)

            # Completion length stats
            lengths = sorted(len(r) for r in responses)
            p50 = lengths[len(lengths) // 2] if lengths else 0
            p95_idx = min(int(len(lengths) * 0.95), len(lengths) - 1)
            p95 = lengths[p95_idx] if lengths else 0
            max_len = lengths[-1] if lengths else 0

            candidate_data.append(
                {
                    "kl": kl,
                    "responses": responses,
                    "t_gen": t_gen,
                    "t_kl": t_kl,
                    "len_p50": p50,
                    "len_p95": p95,
                    "len_max": max_len,
                }
            )

            # Let pending judge calls progress while GPU was busy
            judge.run_pending()

            gpu_bar.update(1)

        gpu_bar.close()

        # Phase 2: Await all judge results
        judge_bar = tqdm(
            total=population_size * len(bad_prompts),
            desc=f"Gen {gen + 1}/{generations} Judge | best={best_compliance}%",
            unit="prompt",
            leave=False,
        )

        scores = []
        for i, (tasks, data) in enumerate(zip(all_judge_tasks, candidate_data, strict=True)):
            verdicts = judge.await_all(tasks, pbar=judge_bar)
            refusals = sum(verdicts)
            compliance_rate = 1.0 - (refusals / len(bad_prompts))
            score = compliance_rate - kl_weight * data["kl"]
            scores.append(score)

            log.info(
                "candidate_eval",
                generation=gen + 1,
                candidate=i + 1,
                population=population_size,
                compliance=round(compliance_rate * 100),
                refusals=refusals,
                kl=round(data["kl"], 4),
                score=round(score, 4),
                len_p50=data["len_p50"],
                len_p95=data["len_p95"],
                len_max=data["len_max"],
                t_gen=f"{data['t_gen']:.1f}s",
                t_kl=f"{data['t_kl']:.1f}s",
            )

        judge_bar.close()

        # Select best
        best_idx = max(range(len(scores)), key=lambda idx: scores[idx])
        if scores[best_idx] > best_score:
            best_score = scores[best_idx]
            best_params = candidates[best_idx].clone()
            model.set_lora_from_flat(best_params)
            best_kl = candidate_data[best_idx]["kl"]
            best_compliance = round((best_score + kl_weight * best_kl) * 100)
            log.info("new_best", compliance=best_compliance, kl=round(best_kl, 4), score=round(best_score, 4))
        else:
            log.info("no_improvement", generation=gen + 1)

        gen_total = time.perf_counter() - gen_t0
        elapsed_total = time.perf_counter() - gen_start
        eta = _fmt((elapsed_total / (gen + 1)) * (generations - gen - 1))
        log.info("generation_done", generation=gen + 1, time=_fmt(gen_total), eta=eta)

    total = time.perf_counter() - gen_start
    log.info("evolution_complete", total_time=_fmt(total), best_compliance=best_compliance, best_kl=round(best_kl, 4))
    model.set_lora_from_flat(best_params)
    return best_params
