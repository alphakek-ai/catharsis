"""Evolution strategies with antithetic sampling over LoRA perturbations."""

import asyncio
import random
import time

import torch
from torch import Tensor
from tqdm import tqdm

from .judge import Judge, JudgeResult
from .log import log
from .model import Model
from .trace import ResponseLengths, TraceWriter
from .vllm_backend import VLLMBackend


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
    vllm_base_url: str | None = None,
) -> Tensor:
    """
    Evolution strategies with antithetic sampling over LoRA parameters.

    If vllm_base_url is provided, uses vLLM multi-LoRA for parallel generation.
    Otherwise falls back to local sequential generation.
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

    # Set up vLLM backend if available
    backend: VLLMBackend | None = None
    if vllm_base_url:
        backend = VLLMBackend(
            base_url=vllm_base_url,
            base_model_name=model.model_name,
            lora_config=model.peft_config,
        )

    trace = TraceWriter()
    trace.write_meta(
        model=model.model_name,
        lora_rank=model.lora_rank,
        lora_targets=model.lora_targets,
        method="antithetic_es",
        backend="vllm" if backend else "local",
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

    total_steps = generations * n_pairs * 2 * n_prompts * 2  # generate + judge
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

        if backend:
            scores_plus, scores_minus = _run_generation_vllm(
                model,
                backend,
                judge,
                params,
                noise_vectors,
                noise_std,
                step_prompts,
                good_prompts,
                base_logprobs,
                kl_weight,
                max_new_tokens,
                batch_size,
                trace,
                gen + 1,
                pbar,
            )
        else:
            scores_plus, scores_minus = _run_generation_local(
                model,
                judge,
                params,
                noise_vectors,
                noise_std,
                step_prompts,
                good_prompts,
                base_logprobs,
                kl_weight,
                max_new_tokens,
                batch_size,
                trace,
                gen + 1,
                pbar,
            )

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

    if backend:
        backend.cleanup()
    pbar.close()
    trace.close()
    total = time.perf_counter() - gen_start
    log.info("evolution_complete", total_time=_fmt(total), best_compliance=best_compliance, best_kl=round(best_kl, 4))
    model.set_lora_from_flat(best_params)
    return best_params


def _run_generation_vllm(
    model: Model,
    backend: VLLMBackend,
    judge: Judge,
    params: Tensor,
    noise_vectors: list[Tensor],
    noise_std: float,
    step_prompts: list[str],
    good_prompts: list[str],
    base_logprobs: Tensor,
    kl_weight: float,
    max_new_tokens: int,
    batch_size: int,
    trace: TraceWriter,
    generation: int,
    pbar: tqdm,
) -> tuple[list[float], list[float]]:
    """Run a generation step using vLLM multi-LoRA — all candidates in parallel."""
    n_pairs = len(noise_vectors)
    param_names, _ = model.get_lora_named_params()

    # 1. Save all adapters to disk and hot-load into vLLM
    adapter_names: list[str] = []
    adapter_sign: list[str] = []  # "+1", "-1", "+2", "-2", ...

    t0 = time.perf_counter()
    for pair_idx, noise in enumerate(noise_vectors):
        for sign, sign_label in [(1.0, "+"), (-1.0, "-")]:
            name = f"gen{generation}_{sign_label}{pair_idx + 1}"
            perturbed = params + sign * noise_std * noise

            # Build param tensors from flat vector
            lora_params = []
            offset = 0
            for p in model.get_lora_params():
                n = p.numel()
                lora_params.append(perturbed[offset : offset + n].view(p.shape))
                offset += n

            backend.save_adapter(name, lora_params, param_names)
            adapter_names.append(name)
            adapter_sign.append(f"{sign_label}{pair_idx + 1}")

    # Hot-load all adapters
    async def load_all():
        for name in adapter_names:
            path = backend._adapter_dir / name
            await backend.load_adapter(name, path)

    asyncio.run(load_all())
    t_save_load = time.perf_counter() - t0
    log.info("adapters_loaded", count=len(adapter_names), time=f"{t_save_load:.1f}s")

    # 2. Generate ALL responses in parallel (all adapters × all prompts)
    t0 = time.perf_counter()

    async def generate_and_judge():
        sem_gen = asyncio.Semaphore(backend.concurrency)
        sem_judge = asyncio.Semaphore(judge.concurrency)

        # Results storage: adapter_name -> prompt_idx -> (response, judge_result)
        gen_results: dict[str, list[dict | None]] = {name: [None] * len(step_prompts) for name in adapter_names}
        judge_results: dict[str, list[JudgeResult | None]] = {
            name: [None] * len(step_prompts) for name in adapter_names
        }

        async def process_one(adapter_name: str, prompt_idx: int, prompt: str):
            # Generate
            async with sem_gen:
                resp = await backend.generate_one(adapter_name, prompt, max_tokens=max_new_tokens)
                gen_results[adapter_name][prompt_idx] = resp
                pbar.update(1)

            # Write to trace immediately
            rl = ResponseLengths(
                reasoning_tokens=resp.get("reasoning_tokens", 0),
                content_tokens=resp["total_tokens"] - resp.get("reasoning_tokens", 0),
                total_tokens=resp["total_tokens"],
            )
            cand_idx = adapter_names.index(adapter_name) + 1
            trace.write_response(
                generation=generation,
                candidate=cand_idx,
                prompt_idx=prompt_idx,
                prompt=prompt,
                response=resp["content"],
                response_lengths=rl,
            )

            # Judge immediately (piped)
            async with sem_judge:
                jr = await judge._judge_one(prompt, resp["content"])
                judge_results[adapter_name][prompt_idx] = (
                    JudgeResult(is_refusal=jr, lengths=None, error=None) if isinstance(jr, bool) else jr
                )
                pbar.update(1)

                trace.write_verdict(
                    generation=generation,
                    candidate=cand_idx,
                    prompt_idx=prompt_idx,
                    is_refusal=jr if isinstance(jr, bool) else jr.is_refusal,
                    judge_lengths=None,
                    judge_error=None,
                )

        # Fire everything at once
        tasks = [
            process_one(adapter_name, idx, prompt)
            for adapter_name in adapter_names
            for idx, prompt in enumerate(step_prompts)
        ]
        await asyncio.gather(*tasks)

        return gen_results, judge_results

    all_gen, all_judge = asyncio.run(generate_and_judge())
    t_gen_judge = time.perf_counter() - t0
    log.info(
        "generation_judging_done", time=f"{t_gen_judge:.1f}s", total_requests=len(adapter_names) * len(step_prompts)
    )

    # 3. Compute scores
    scores_plus = []
    scores_minus = []

    for pair_idx in range(n_pairs):
        for sign_label, score_list in [("+", scores_plus), ("-", scores_minus)]:
            name = f"gen{generation}_{sign_label}{pair_idx + 1}"
            judge_res = all_judge[name]

            refusals = sum(1 for r in judge_res if r is not None and (r.is_refusal is True or r.is_refusal is None))
            compliance_rate = 1.0 - (refusals / len(step_prompts))

            # KL: compute locally (fast, 1 token per prompt)
            model.set_lora_from_flat(
                params + (1.0 if sign_label == "+" else -1.0) * noise_std * noise_vectors[pair_idx]
            )
            kl = model.compute_kl(good_prompts, base_logprobs, batch_size=batch_size)

            score = compliance_rate - kl_weight * kl
            score_list.append(score)

            log.info(
                "candidate_eval",
                generation=generation,
                candidate=f"{sign_label}{pair_idx + 1}",
                compliance=round(compliance_rate * 100),
                refusals=refusals,
                kl=round(kl, 4),
                score=round(score, 4),
            )

    # 4. Cleanup adapters
    async def unload_all():
        for name in adapter_names:
            await backend.unload_adapter(name)

    asyncio.run(unload_all())

    return scores_plus, scores_minus


def _run_generation_local(
    model: Model,
    judge: Judge,
    params: Tensor,
    noise_vectors: list[Tensor],
    noise_std: float,
    step_prompts: list[str],
    good_prompts: list[str],
    base_logprobs: Tensor,
    kl_weight: float,
    max_new_tokens: int,
    batch_size: int,
    trace: TraceWriter,
    generation: int,
    pbar: tqdm,
) -> tuple[list[float], list[float]]:
    """Run a generation step locally (sequential, for when no vLLM server)."""
    scores_plus = []
    scores_minus = []

    for pair_idx, noise in enumerate(noise_vectors):
        for sign, sign_label, score_list in [(1.0, "+", scores_plus), (-1.0, "-", scores_minus)]:
            model.set_lora_from_flat(params + sign * noise_std * noise)
            cand_idx = pair_idx * 2 + (1 if sign > 0 else 2)

            # Generate
            t0 = time.perf_counter()
            gen_results = []
            for prompt_idx, resp in enumerate(
                model.generate_responses_iter(step_prompts, max_new_tokens=max_new_tokens, batch_size=batch_size)
            ):
                gen_results.append(resp)
                pbar.update(1)
                rl = ResponseLengths(
                    reasoning_tokens=resp.reasoning_tokens,
                    content_tokens=resp.content_tokens,
                    total_tokens=resp.total_tokens,
                )
                trace.write_response(
                    generation=generation,
                    candidate=cand_idx,
                    prompt_idx=prompt_idx,
                    prompt=resp.prompt,
                    response=resp.content,
                    response_lengths=rl,
                    raw_response=resp.raw if resp.raw != resp.content else None,
                )
            t_gen = time.perf_counter() - t0

            # KL
            kl = model.compute_kl(good_prompts, base_logprobs, batch_size=batch_size)

            # Judge
            tasks = [judge.submit(p, r.content) for p, r in zip(step_prompts, gen_results, strict=True)]
            judge.run_pending()
            results = judge.await_all(tasks, pbar=pbar)

            for prompt_idx, r in enumerate(results):
                trace.write_verdict(
                    generation=generation,
                    candidate=cand_idx,
                    prompt_idx=prompt_idx,
                    is_refusal=r.is_refusal,
                    judge_lengths=r.lengths,
                    judge_error=r.error,
                )

            refusals = sum(1 for r in results if r.is_refusal is True or r.is_refusal is None)
            compliance_rate = 1.0 - (refusals / len(step_prompts))
            score = compliance_rate - kl_weight * kl
            score_list.append(score)

            log.info(
                "candidate_eval",
                generation=generation,
                candidate=f"{sign_label}{pair_idx + 1}",
                compliance=round(compliance_rate * 100),
                refusals=refusals,
                kl=round(kl, 4),
                score=round(score, 4),
                t_gen=f"{t_gen:.1f}s",
            )

            for r in results:
                if r.error is not None:
                    log.warning("judge_error", gen=generation, cand=cand_idx, error=r.error)

    return scores_plus, scores_minus
