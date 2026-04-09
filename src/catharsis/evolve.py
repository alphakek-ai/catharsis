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
        candidate_data: list[dict] = []
        all_judge_tasks = []

        gpu_bar = tqdm(
            total=population_size,
            desc=f"Gen {gen + 1}/{generations} GPU | best={best_compliance}%",
            unit="cand",
            leave=False,
        )

        for candidate in candidates:
            model.set_lora_from_flat(candidate)

            t0 = time.perf_counter()
            responses = model.generate_responses(bad_prompts, max_new_tokens=max_new_tokens, batch_size=batch_size)
            t_gen = time.perf_counter() - t0

            t0 = time.perf_counter()
            kl = model.compute_kl(good_prompts, base_logprobs, batch_size=batch_size)
            t_kl = time.perf_counter() - t0

            # Measure student model response lengths
            response_lengths = [measure_response(r) for r in responses]
            content_lens = sorted(rl.content for rl in response_lengths)
            reasoning_lens = sorted(rl.reasoning for rl in response_lengths)

            # Fire judge calls (non-blocking)
            tasks = [judge.submit(p, r) for p, r in zip(bad_prompts, responses, strict=True)]
            all_judge_tasks.append(tasks)

            candidate_data.append(
                {
                    "kl": kl,
                    "responses": responses,
                    "response_lengths": response_lengths,
                    "t_gen": t_gen,
                    "t_kl": t_kl,
                    "content_p50": content_lens[len(content_lens) // 2] if content_lens else 0,
                    "content_p95": content_lens[min(int(len(content_lens) * 0.95), len(content_lens) - 1)]
                    if content_lens
                    else 0,
                    "content_max": content_lens[-1] if content_lens else 0,
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
        for i, (tasks, data) in enumerate(zip(all_judge_tasks, candidate_data, strict=True)):
            results = judge.await_all(tasks, pbar=judge_bar)

            verdicts = [r.is_refusal for r in results]
            judge_lengths = [r.lengths for r in results]
            judge_errors = [r.error for r in results]

            # Count refusals — judge errors count as refusals
            refusals = sum(1 for v in verdicts if v is True or v is None)
            compliance_rate = 1.0 - (refusals / len(bad_prompts))
            score = compliance_rate - kl_weight * data["kl"]
            scores.append(score)

            # Judge length stats (exclude errors)
            judge_content_lens = sorted(jl.content for jl in judge_lengths if jl is not None)
            judge_reasoning_lens = sorted(jl.reasoning for jl in judge_lengths if jl is not None)
            judge_total_lens = sorted(jl.total for jl in judge_lengths if jl is not None)
            n_errors = sum(1 for e in judge_errors if e is not None)

            # Write trace to disk
            trace.write_candidate(
                generation=gen + 1,
                candidate=i + 1,
                prompts=bad_prompts,
                responses=data["responses"],
                response_lengths=data["response_lengths"],
                verdicts=verdicts,
                judge_lengths=judge_lengths,
                judge_errors=judge_errors,
                kl=data["kl"],
                score=score,
            )

            log.info(
                "candidate_eval",
                generation=gen + 1,
                candidate=i + 1,
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

            # Log any judge errors
            for e in judge_errors:
                if e is not None:
                    log.warning("judge_error", generation=gen + 1, candidate=i + 1, error=e)

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

        trace.write_generation_summary(gen + 1, best_score, best_compliance, best_kl)

        gen_total = time.perf_counter() - gen_t0
        elapsed_total = time.perf_counter() - gen_start
        eta = _fmt((elapsed_total / (gen + 1)) * (generations - gen - 1))
        log.info("generation_done", generation=gen + 1, time=_fmt(gen_total), eta=eta)

    total = time.perf_counter() - gen_start
    log.info("evolution_complete", total_time=_fmt(total), best_compliance=best_compliance, best_kl=round(best_kl, 4))
    model.set_lora_from_flat(best_params)
    return best_params
