"""Evolutionary search over LoRA perturbations."""

import time

import torch
from torch import Tensor
from tqdm import tqdm

from .judge import Judge
from .log import log
from .model import Model
from .trace import TraceWriter, measure_response


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

    trace = TraceWriter()
    trace.write_meta(
        model=model.model_name,
        lora_rank=model.lora_rank,
        lora_targets=model.lora_targets,
        population_size=population_size,
        generations=generations,
        noise_std=noise_std,
        kl_weight=kl_weight,
        n_good=len(good_prompts),
        n_bad=len(bad_prompts),
        max_new_tokens=max_new_tokens,
    )
    log.info("trace_dir", path=str(trace.base_dir))

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

        # Phase 1: GPU work — generate responses + compute KL, fire judge calls
        candidate_gpu_data: list[dict] = []
        all_judge_tasks: list[list] = []

        gpu_bar = tqdm(
            total=population_size,
            desc=f"Gen {gen + 1}/{generations} GPU | best={best_compliance}%",
            unit="cand",
            leave=False,
        )

        for cand_idx, candidate in enumerate(candidates):
            model.set_lora_from_flat(candidate)

            t0 = time.perf_counter()
            gen_results = list(
                model.generate_responses_iter(bad_prompts, max_new_tokens=max_new_tokens, batch_size=batch_size)
            )
            t_gen = time.perf_counter() - t0

            clean_responses = [clean for _, clean, _raw in gen_results]

            # Write responses to trace immediately (raw includes reasoning tokens)
            response_lengths = []
            for prompt_idx, (_prompt, clean, raw) in enumerate(gen_results):
                rl = measure_response(raw)
                response_lengths.append(rl)
                trace.write_response(
                    generation=gen + 1,
                    candidate=cand_idx + 1,
                    prompt_idx=prompt_idx,
                    prompt=bad_prompts[prompt_idx],
                    response=clean,
                    response_lengths=rl,
                    raw_response=raw,
                )

            t0 = time.perf_counter()
            kl = model.compute_kl(good_prompts, base_logprobs, batch_size=batch_size)
            t_kl = time.perf_counter() - t0

            # Fire judge calls with clean text (non-blocking)
            tasks = [judge.submit(p, r) for p, r in zip(bad_prompts, clean_responses, strict=True)]
            all_judge_tasks.append(tasks)

            content_lens = sorted(rl.content for rl in response_lengths)
            reasoning_lens = sorted(rl.reasoning for rl in response_lengths)

            candidate_gpu_data.append(
                {
                    "kl": kl,
                    "t_gen": t_gen,
                    "t_kl": t_kl,
                    "content_p50": content_lens[len(content_lens) // 2] if content_lens else 0,
                    "content_p95": content_lens[min(int(len(content_lens) * 0.95), len(content_lens) - 1)]
                    if content_lens
                    else 0,
                    "reasoning_p50": reasoning_lens[len(reasoning_lens) // 2] if reasoning_lens else 0,
                    "reasoning_max": reasoning_lens[-1] if reasoning_lens else 0,
                }
            )

            judge.run_pending()
            gpu_bar.update(1)

        gpu_bar.close()

        # Phase 2: Await judge results
        judge_bar = tqdm(
            total=population_size * len(bad_prompts),
            desc=f"Gen {gen + 1}/{generations} Judge | best={best_compliance}%",
            unit="prompt",
            leave=False,
        )

        scores = []
        for cand_idx, (tasks, data) in enumerate(zip(all_judge_tasks, candidate_gpu_data, strict=True)):
            results = judge.await_all(tasks, pbar=judge_bar)

            # Write verdicts to trace immediately
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
            n_errors = sum(1 for r in results if r.error is not None)
            compliance_rate = 1.0 - (refusals / len(bad_prompts))
            score = compliance_rate - kl_weight * data["kl"]
            scores.append(score)

            judge_content_lens = sorted(r.lengths.content for r in results if r.lengths is not None)
            judge_reasoning_lens = sorted(r.lengths.reasoning for r in results if r.lengths is not None)
            judge_total_lens = sorted(r.lengths.total for r in results if r.lengths is not None)

            trace.write_candidate_summary(
                generation=gen + 1,
                candidate=cand_idx + 1,
                refusals=refusals,
                compliance=len(bad_prompts) - refusals,
                judge_errors=n_errors,
                kl=data["kl"],
                score=score,
            )

            log.info(
                "candidate_eval",
                generation=gen + 1,
                candidate=cand_idx + 1,
                population=population_size,
                compliance=round(compliance_rate * 100),
                refusals=refusals,
                judge_errors=n_errors,
                kl=round(data["kl"], 4),
                score=round(score, 4),
                student_content_p50=data["content_p50"],
                student_content_p95=data["content_p95"],
                student_reasoning_p50=data["reasoning_p50"],
                judge_content_p50=judge_content_lens[len(judge_content_lens) // 2] if judge_content_lens else 0,
                judge_reasoning_p50=judge_reasoning_lens[len(judge_reasoning_lens) // 2] if judge_reasoning_lens else 0,
                judge_total_max=judge_total_lens[-1] if judge_total_lens else 0,
                t_gen=f"{data['t_gen']:.1f}s",
                t_kl=f"{data['t_kl']:.1f}s",
            )

            for r in results:
                if r.error is not None:
                    log.warning("judge_error", gen=gen + 1, cand=cand_idx + 1, error=r.error)

        judge_bar.close()

        # Select best
        best_idx = max(range(len(scores)), key=lambda idx: scores[idx])
        if scores[best_idx] > best_score:
            best_score = scores[best_idx]
            best_params = candidates[best_idx].clone()
            model.set_lora_from_flat(best_params)
            best_kl = candidate_gpu_data[best_idx]["kl"]
            best_compliance = round((best_score + kl_weight * best_kl) * 100)
            log.info("new_best", compliance=best_compliance, kl=round(best_kl, 4), score=round(best_score, 4))
        else:
            log.info("no_improvement", generation=gen + 1)

        trace.write_generation_summary(gen + 1, best_score, best_compliance, best_kl)

        gen_total = time.perf_counter() - gen_t0
        elapsed_total = time.perf_counter() - gen_start
        eta = _fmt((elapsed_total / (gen + 1)) * (generations - gen - 1))
        log.info("generation_done", generation=gen + 1, time=_fmt(gen_total), eta=eta)

    trace.close()
    total = time.perf_counter() - gen_start
    log.info("evolution_complete", total_time=_fmt(total), best_compliance=best_compliance, best_kl=round(best_kl, 4))
    model.set_lora_from_flat(best_params)
    return best_params
