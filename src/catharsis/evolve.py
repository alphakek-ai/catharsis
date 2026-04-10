"""Evolution strategies with antithetic sampling over LoRA perturbations."""

import random
import time

import torch
from torch import Tensor
from tqdm import tqdm

from .judge import Judge
from .log import log
from .model import Model
from .trace import ResponseLengths, TraceWriter


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
    noise_std: float = 0.01,
    kl_weight: float = 1.0,
    learning_rate: float = 0.01,
    batch_size: int = 32,
    max_new_tokens: int = 2000,
    prompts_per_step: int | None = None,
) -> Tensor:
    """
    Evolution strategies with antithetic sampling over LoRA parameters.

    Uses transformers continuous batching for fast generation — short responses
    (refusals) finish immediately and free up the batch while long responses
    continue generating. Judge calls fire as each response completes.
    """
    n_params = model.lora_param_count()
    device = model.device
    n_pairs = population_size // 2
    n_prompts = prompts_per_step or len(bad_prompts)

    params = model.get_lora_flat().clone()
    best_score = float("-inf")
    best_compliance = 0
    best_kl = float("inf")
    best_params = params.clone()

    trace = TraceWriter()
    trace.write_meta(
        model=model.model_name,
        lora_rank=model.lora_rank,
        lora_targets=model.lora_targets,
        method="antithetic_es",
        backend="continuous_batching",
        population_size=population_size,
        n_pairs=n_pairs,
        generations=generations,
        noise_std=noise_std,
        learning_rate=learning_rate,
        kl_weight=kl_weight,
        n_good=len(good_prompts),
        n_bad=len(bad_prompts),
        prompts_per_step=n_prompts,
        max_new_tokens=max_new_tokens,
    )
    log.info("trace_dir", path=str(trace.base_dir))

    # Total steps: each pair = 2 candidates, each candidate = n_prompts (generate + judge)
    total_steps = generations * n_pairs * 2 * n_prompts * 2
    pbar = tqdm(total=total_steps, desc="best=0%", unit="step")

    gen_start = time.perf_counter()

    for gen in range(generations):
        gen_t0 = time.perf_counter()
        pbar.set_description(f"Gen {gen + 1}/{generations} | best={best_compliance}%")

        # Sample mini-batch
        step_prompts = random.sample(bad_prompts, min(n_prompts, len(bad_prompts)))

        # Generate noise vectors
        noise_vectors = []
        for _ in range(n_pairs):
            noise_vectors.append(torch.randn(n_params, device=device))

        # Build all candidate param vectors (antithetic pairs)
        candidate_params = []
        candidate_labels = []  # (sign_label, pair_idx)
        for pair_idx, noise in enumerate(noise_vectors):
            candidate_params.append(params + noise_std * noise)
            candidate_labels.append(("+", pair_idx))
            candidate_params.append(params - noise_std * noise)
            candidate_labels.append(("-", pair_idx))

        # Generate ALL candidates in ONE forward pass
        t0 = time.perf_counter()
        all_results = model.generate_batched_candidates(step_prompts, candidate_params, max_new_tokens=max_new_tokens)
        t_gen = time.perf_counter() - t0
        pbar.update(len(candidate_params) * len(step_prompts))  # generation steps

        log.info(
            "batched_generation_done",
            n_candidates=len(candidate_params),
            n_prompts=len(step_prompts),
            t_gen=f"{t_gen:.1f}s",
        )

        # Score each candidate: write traces, fire judge calls, compute KL
        scores_plus = []
        scores_minus = []

        all_judge_tasks = []
        for cand_idx, (sign_label, pair_idx) in enumerate(candidate_labels):
            gen_results = all_results[cand_idx]

            # Write traces
            for prompt_idx, resp in enumerate(gen_results):
                rl = ResponseLengths(
                    reasoning_tokens=resp.reasoning_tokens,
                    content_tokens=resp.content_tokens,
                    total_tokens=resp.total_tokens,
                )
                trace.write_response(
                    generation=gen + 1,
                    candidate=cand_idx + 1,
                    prompt_idx=prompt_idx,
                    prompt=resp.prompt,
                    response=resp.content,
                    response_lengths=rl,
                    raw_response=resp.raw if resp.raw != resp.content else None,
                )

            # Fire judge calls (non-blocking)
            tasks = [judge.submit(p, r.content) for p, r in zip(step_prompts, gen_results, strict=True)]
            all_judge_tasks.append((cand_idx, sign_label, pair_idx, gen_results, tasks))

        # Compute KL per candidate (requires loading adapter)
        candidate_kls = []
        for cand_idx in range(len(candidate_params)):
            model.set_lora_from_flat(candidate_params[cand_idx])
            kl = model.compute_kl(good_prompts, base_logprobs, batch_size=batch_size)
            candidate_kls.append(kl)

        # Await all judge results
        for cand_idx, sign_label, pair_idx, gen_results, tasks in all_judge_tasks:
            judge.run_pending()
            results = judge.await_all(tasks, pbar=pbar)

            for prompt_idx, r in enumerate(results):
                trace.write_verdict(
                    generation=gen + 1,
                    candidate=cand_idx + 1,
                    prompt_idx=prompt_idx,
                    is_refusal=r.is_refusal,
                    judge_lengths=r.lengths,
                    judge_error=r.error,
                )

            refusals = sum(1 for r in results if r.is_refusal is True or r.is_refusal is None)
            compliance_rate = 1.0 - (refusals / len(step_prompts))
            kl = candidate_kls[cand_idx]
            score = compliance_rate - kl_weight * kl

            if sign_label == "+":
                scores_plus.append(score)
            else:
                scores_minus.append(score)

            total_tok = sorted(r.total_tokens for r in gen_results)
            reasoning_tok = sorted(r.reasoning_tokens for r in gen_results)
            judge_reasoning_tok = sorted(r.lengths.reasoning_tokens for r in results if r.lengths is not None)
            judge_total_tok = sorted(r.lengths.total_tokens for r in results if r.lengths is not None)
            n_errors = sum(1 for r in results if r.error is not None)

            log.info(
                "candidate_eval",
                generation=gen + 1,
                candidate=f"{sign_label}{pair_idx + 1}",
                compliance=round(compliance_rate * 100),
                refusals=refusals,
                judge_errors=n_errors,
                kl=round(kl, 4),
                score=round(score, 4),
                student_tok_p50=total_tok[len(total_tok) // 2] if total_tok else 0,
                student_tok_max=total_tok[-1] if total_tok else 0,
                student_reasoning_p50=reasoning_tok[len(reasoning_tok) // 2] if reasoning_tok else 0,
                judge_reasoning_p50=judge_reasoning_tok[len(judge_reasoning_tok) // 2] if judge_reasoning_tok else 0,
                judge_tok_max=judge_total_tok[-1] if judge_total_tok else 0,
            )

            for r in results:
                if r.error is not None:
                    log.warning("judge_error", gen=gen + 1, cand=cand_idx + 1, error=r.error)

        # Compute ES gradient
        grad = torch.zeros(n_params, device=device)
        for noise, sp, sm in zip(noise_vectors, scores_plus, scores_minus, strict=True):
            grad += (sp - sm) * noise
        grad /= n_pairs * noise_std

        # Gradient ascent
        params = params + learning_rate * grad

        mean_score = sum(scores_plus) / len(scores_plus)
        grad_norm = grad.norm().item()

        log.info(
            "es_update",
            generation=gen + 1,
            grad_norm=round(grad_norm, 6),
            param_norm=round(params.norm().item(), 4),
            mean_score_plus=round(sum(scores_plus) / len(scores_plus), 4),
            mean_score_minus=round(sum(scores_minus) / len(scores_minus), 4),
        )

        if mean_score > best_score:
            best_score = mean_score
            best_params = params.clone()
            model.set_lora_from_flat(best_params)
            best_kl = model.compute_kl(good_prompts, base_logprobs, batch_size=batch_size)
            best_compliance = round((best_score + kl_weight * best_kl) * 100)
            log.info("new_best", compliance=best_compliance, kl=round(best_kl, 4), score=round(best_score, 4))

        trace.write_generation_summary(gen + 1, best_score, best_compliance, best_kl)

        gen_total = time.perf_counter() - gen_t0
        elapsed_total = time.perf_counter() - gen_start
        eta = _fmt((elapsed_total / (gen + 1)) * (generations - gen - 1))
        log.info("generation_done", generation=gen + 1, time=_fmt(gen_total), eta=eta)

    pbar.close()
    trace.close()
    total = time.perf_counter() - gen_start
    log.info("evolution_complete", total_time=_fmt(total), best_compliance=best_compliance, best_kl=round(best_kl, 4))
    model.set_lora_from_flat(best_params)
    return best_params
