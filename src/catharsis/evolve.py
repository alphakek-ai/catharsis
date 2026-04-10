"""Evolution strategies with antithetic sampling over LoRA perturbations."""

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


def _evaluate_candidate(
    model: Model,
    judge: Judge,
    bad_prompts: list[str],
    good_prompts: list[str],
    base_logprobs: Tensor,
    kl_weight: float,
    max_new_tokens: int,
    batch_size: int,
    trace: TraceWriter,
    generation: int,
    candidate_idx: int,
    pbar: tqdm,
) -> tuple[float, dict]:
    """Evaluate a single candidate. Returns (score, gpu_data)."""
    t0 = time.perf_counter()
    gen_results = []
    response_lengths = []
    for prompt_idx, resp in enumerate(
        model.generate_responses_iter(bad_prompts, max_new_tokens=max_new_tokens, batch_size=batch_size)
    ):
        gen_results.append(resp)
        pbar.update(1)
        rl = ResponseLengths(
            reasoning_tokens=resp.reasoning_tokens,
            content_tokens=resp.content_tokens,
            total_tokens=resp.total_tokens,
        )
        response_lengths.append(rl)
        trace.write_response(
            generation=generation,
            candidate=candidate_idx,
            prompt_idx=prompt_idx,
            prompt=resp.prompt,
            response=resp.content,
            response_lengths=rl,
            raw_response=resp.raw if resp.raw != resp.content else None,
        )
    t_gen = time.perf_counter() - t0

    t0 = time.perf_counter()
    kl = model.compute_kl(good_prompts, base_logprobs, batch_size=batch_size)
    t_kl = time.perf_counter() - t0

    # Fire judge calls
    tasks = [judge.submit(p, r.content) for p, r in zip(bad_prompts, gen_results, strict=True)]
    judge.run_pending()

    # Await results
    results = judge.await_all(tasks, pbar=pbar)
    for prompt_idx, r in enumerate(results):
        trace.write_verdict(
            generation=generation,
            candidate=candidate_idx,
            prompt_idx=prompt_idx,
            is_refusal=r.is_refusal,
            judge_lengths=r.lengths,
            judge_error=r.error,
        )

    refusals = sum(1 for r in results if r.is_refusal is True or r.is_refusal is None)
    n_errors = sum(1 for r in results if r.error is not None)
    compliance_rate = 1.0 - (refusals / len(bad_prompts))
    score = compliance_rate - kl_weight * kl

    total_tok = sorted(rl.total_tokens for rl in response_lengths)
    reasoning_tok = sorted(rl.reasoning_tokens for rl in response_lengths)
    content_tok = sorted(rl.content_tokens for rl in response_lengths)
    judge_reasoning_tok = sorted(r.lengths.reasoning_tokens for r in results if r.lengths is not None)
    judge_total_tok = sorted(r.lengths.total_tokens for r in results if r.lengths is not None)

    gpu_data = {
        "kl": kl,
        "compliance": round(compliance_rate * 100),
        "refusals": refusals,
        "judge_errors": n_errors,
        "t_gen": t_gen,
        "t_kl": t_kl,
        "total_tok_p50": total_tok[len(total_tok) // 2] if total_tok else 0,
        "total_tok_p95": total_tok[min(int(len(total_tok) * 0.95), len(total_tok) - 1)] if total_tok else 0,
        "total_tok_max": total_tok[-1] if total_tok else 0,
        "reasoning_tok_p50": reasoning_tok[len(reasoning_tok) // 2] if reasoning_tok else 0,
        "content_tok_p50": content_tok[len(content_tok) // 2] if content_tok else 0,
        "judge_reasoning_tok_p50": judge_reasoning_tok[len(judge_reasoning_tok) // 2] if judge_reasoning_tok else 0,
        "judge_total_tok_max": judge_total_tok[-1] if judge_total_tok else 0,
    }

    trace.write_candidate_summary(
        generation=generation,
        candidate=candidate_idx,
        refusals=refusals,
        compliance=len(bad_prompts) - refusals,
        judge_errors=n_errors,
        kl=kl,
        score=score,
    )

    for r in results:
        if r.error is not None:
            log.warning("judge_error", gen=generation, cand=candidate_idx, error=r.error)

    return score, gpu_data


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
    learning_rate: float = 0.01,
    batch_size: int = 32,
    max_new_tokens: int = 1000,
    prompts_per_step: int | None = None,
) -> Tensor:
    """
    Evolution strategies with antithetic sampling over LoRA parameters.

    For each generation:
    1. Sample a random mini-batch of bad prompts
    2. Sample N/2 noise vectors
    3. Evaluate both +noise and -noise (antithetic pairs)
    4. Compute ES gradient: grad = mean((score_plus - score_minus) * noise) / sigma
    5. Update params with gradient ascent
    """
    import random

    n_params = model.lora_param_count()
    device = model.device
    n_pairs = population_size // 2  # antithetic pairs
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

    # Total steps: each pair = 2 candidates, each candidate = n_prompts generate + n_prompts judge
    total_steps = generations * n_pairs * 2 * n_prompts * 2
    pbar = tqdm(total=total_steps, desc="best=0%", unit="step")

    gen_start = time.perf_counter()

    for gen in range(generations):
        gen_t0 = time.perf_counter()
        pbar.set_description(f"Gen {gen + 1}/{generations} | best={best_compliance}%")

        # Sample mini-batch of bad prompts for this step
        if n_prompts < len(bad_prompts):
            step_prompts = random.sample(bad_prompts, n_prompts)
        else:
            step_prompts = bad_prompts

        # Sample noise vectors and evaluate antithetic pairs
        noise_vectors = []
        scores_plus = []
        scores_minus = []

        for pair_idx in range(n_pairs):
            noise = torch.randn(n_params, device=device)
            noise_vectors.append(noise)

            # Evaluate +noise
            model.set_lora_from_flat(params + noise_std * noise)
            score_plus, data_plus = _evaluate_candidate(
                model,
                judge,
                step_prompts,
                good_prompts,
                base_logprobs,
                kl_weight,
                max_new_tokens,
                batch_size,
                trace,
                generation=gen + 1,
                candidate_idx=pair_idx * 2 + 1,
                pbar=pbar,
            )
            scores_plus.append(score_plus)

            log.info(
                "candidate_eval",
                generation=gen + 1,
                candidate=f"+{pair_idx + 1}",
                score=round(score_plus, 4),
                **{k: round(v, 4) if isinstance(v, float) else v for k, v in data_plus.items()},
            )

            # Evaluate -noise
            model.set_lora_from_flat(params - noise_std * noise)
            score_minus, data_minus = _evaluate_candidate(
                model,
                judge,
                step_prompts,
                good_prompts,
                base_logprobs,
                kl_weight,
                max_new_tokens,
                batch_size,
                trace,
                generation=gen + 1,
                candidate_idx=pair_idx * 2 + 2,
                pbar=pbar,
            )
            scores_minus.append(score_minus)

            log.info(
                "candidate_eval",
                generation=gen + 1,
                candidate=f"-{pair_idx + 1}",
                score=round(score_minus, 4),
                **{k: round(v, 4) if isinstance(v, float) else v for k, v in data_minus.items()},
            )

        # Compute ES gradient estimate
        # grad = (1 / (n_pairs * sigma)) * sum((score_plus - score_minus) * noise)
        grad = torch.zeros(n_params, device=device)
        for noise, sp, sm in zip(noise_vectors, scores_plus, scores_minus, strict=True):
            grad += (sp - sm) * noise
        grad /= n_pairs * noise_std

        # Gradient ascent (we want to maximize score)
        params = params + learning_rate * grad

        # Track the best — evaluate current params
        model.set_lora_from_flat(params)
        current_kl = model.compute_kl(good_prompts, base_logprobs, batch_size=batch_size)

        mean_score = sum(scores_plus) / len(scores_plus)

        grad_norm = grad.norm().item()
        log.info(
            "es_update",
            generation=gen + 1,
            grad_norm=round(grad_norm, 6),
            param_norm=round(params.norm().item(), 4),
            mean_score_plus=round(sum(scores_plus) / len(scores_plus), 4),
            mean_score_minus=round(sum(scores_minus) / len(scores_minus), 4),
            current_kl=round(current_kl, 4),
        )

        if mean_score > best_score:
            best_score = mean_score
            best_params = params.clone()
            best_kl = current_kl
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
