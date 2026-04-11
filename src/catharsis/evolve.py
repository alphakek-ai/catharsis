"""Evolution strategies with antithetic sampling and structured noise.

Pipeline per generation:
1. Generate structured per-module noise (like EGGROLL)
2. For each sub-batch of candidates:
   a. Generate responses with noise hooks (GPU, heavy)
   b. Write traces + fire judge calls (immediate)
   c. Compute KL for these candidates (GPU, light — judge in background)
3. Await judge results, compute scores
4. Compute ES gradient per-module, update LoRA weights
"""

import random
import time
from collections import Counter

import torch
from torch import Tensor
from tqdm import tqdm

from .judge import Judge
from .log import log
from .model import Model
from .noise import StructuredNoise, build_batched_noise_params, generate_structured_noise
from .trace import ResponseLengths, TraceWriter


def calibrate_sigma(
    model: Model,
    good_prompts: list[str],
    base_logprobs: Tensor,
    lora_names: list[str],
    lora_shapes: list[tuple[int, ...]],
    lora_rank: int,
    target_kl: float = 0.1,
    max_iterations: int = 15,
    batch_size: int = 32,
) -> float:
    """Binary search for the sigma that produces target_kl divergence.

    Finds the noise magnitude where the model's outputs change meaningfully
    (KL ~0.1) without being destroyed (KL >> 1).
    """
    device = model.device
    lo, hi = 1e-4, 1.0

    log.info("calibrating_sigma", target_kl=target_kl)

    for iteration in range(max_iterations):
        mid = (lo + hi) / 2

        # Generate one noise sample and apply as +noise
        noise = generate_structured_noise(lora_names, lora_shapes, lora_rank, device)
        noise_params = build_batched_noise_params([noise], [1.0], mid, device)

        # Run a quick forward pass to measure KL
        # We need to temporarily apply noise via hooks
        prompt_texts = model.tokenize_prompts(good_prompts[:10])  # just 10 prompts for speed
        candidate_ids = [0] * len(prompt_texts)

        from .batched_lora import BatchedNoiseContext

        inputs = model.processor(text=prompt_texts, return_tensors="pt", padding=True).to(device)
        with BatchedNoiseContext(model.model, noise_params, torch.tensor(candidate_ids, device=device)):
            with torch.no_grad():
                outputs = model.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    output_scores=True,
                    return_dict_in_generate=True,
                    pad_token_id=model.tokenizer.pad_token_id,
                    do_sample=False,
                )
        logits = outputs.scores[0]
        noised_logprobs = torch.nn.functional.log_softmax(logits, dim=-1).cpu()
        kl = torch.nn.functional.kl_div(
            noised_logprobs, base_logprobs[:10], reduction="batchmean", log_target=True
        ).item()

        log.info("sigma_probe", iteration=iteration + 1, sigma=round(mid, 6), kl=round(kl, 4))

        if abs(kl - target_kl) / target_kl < 0.3:  # within 30% of target
            log.info("sigma_calibrated", sigma=round(mid, 6), kl=round(kl, 4))
            return mid

        if kl > target_kl:
            hi = mid
        else:
            lo = mid

    final_sigma = (lo + hi) / 2
    log.info("sigma_calibrated", sigma=round(final_sigma, 6), note="max iterations reached")
    return final_sigma


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
    max_batch_sequences: int = 64,
) -> Tensor:
    n_pairs = population_size // 2
    n_prompts = prompts_per_step or len(bad_prompts)
    n_candidates = n_pairs * 2
    device = model.device

    # Get LoRA structure info for noise generation
    lora_names, lora_shapes = model.get_lora_named_params()
    lora_names_list = [n for n in lora_names]
    lora_shapes_list = [tuple(p.shape) for p in lora_shapes]
    lora_rank = model.lora_rank

    # Auto-calibrate sigma if requested (noise_std=0 means auto)
    if noise_std <= 0:
        noise_std = calibrate_sigma(model, good_prompts, base_logprobs, lora_names_list, lora_shapes_list, lora_rank)
    else:
        # Validate the provided sigma with a quick KL check
        test_noise = generate_structured_noise(lora_names_list, lora_shapes_list, lora_rank, device)
        test_params = build_batched_noise_params([test_noise], [1.0], noise_std, device)

        from .batched_lora import BatchedNoiseContext

        test_texts = model.tokenize_prompts(good_prompts[:10])
        test_inputs = model.processor(text=test_texts, return_tensors="pt", padding=True).to(device)
        with BatchedNoiseContext(
            model.model, test_params, torch.zeros(len(test_texts), dtype=torch.long, device=device)
        ):
            with torch.no_grad():
                test_out = model.model.generate(
                    **test_inputs,
                    max_new_tokens=1,
                    output_scores=True,
                    return_dict_in_generate=True,
                    pad_token_id=model.tokenizer.pad_token_id,
                    do_sample=False,
                )
        test_logprobs = torch.nn.functional.log_softmax(test_out.scores[0], dim=-1).cpu()
        test_kl = torch.nn.functional.kl_div(
            test_logprobs, base_logprobs[:10], reduction="batchmean", log_target=True
        ).item()
        log.info("sigma_check", sigma=round(noise_std, 6), kl=round(test_kl, 4))
        if test_kl > 1.0:
            log.warning(
                "sigma_too_high", sigma=noise_std, kl=test_kl, hint="Consider using --noise-std 0 for auto-calibration"
            )

    log.info("sigma_ready", sigma=round(noise_std, 6))

    best_score = float("-inf")
    best_compliance = 0
    best_kl = float("inf")

    trace = TraceWriter()
    trace.write_meta(
        model=model.model_name,
        lora_rank=lora_rank,
        lora_targets=model.lora_targets,
        method="antithetic_es_structured",
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
        max_batch_sequences=max_batch_sequences,
    )
    log.info("trace_dir", path=str(trace.base_dir))

    total_steps = generations * n_candidates * n_prompts * 2
    pbar = tqdm(total=total_steps, desc="best=0%", unit="step")

    gen_start = time.perf_counter()

    for gen in range(generations):
        gen_t0 = time.perf_counter()
        pbar.set_description(f"Gen {gen + 1}/{generations} | best={best_compliance}%")

        # Sample mini-batch of prompts
        step_prompts = random.sample(bad_prompts, min(n_prompts, len(bad_prompts)))
        prompt_texts = model.tokenize_prompts(step_prompts)

        # Generate structured noise (per-module, like EGGROLL)
        noise_samples = [
            generate_structured_noise(lora_names_list, lora_shapes_list, lora_rank, device) for _ in range(n_pairs)
        ]

        # Build antithetic candidate list: +noise, -noise for each pair
        candidate_noises: list[StructuredNoise] = []
        candidate_signs: list[float] = []
        candidate_labels: list[tuple[str, int]] = []
        for pair_idx, noise in enumerate(noise_samples):
            candidate_noises.append(noise)
            candidate_signs.append(1.0)
            candidate_labels.append(("+", pair_idx))
            candidate_noises.append(noise)  # same noise, opposite sign
            candidate_signs.append(-1.0)
            candidate_labels.append(("-", pair_idx))

        # Build stacked noise params for hooks (sigma decays linearly)
        current_sigma = noise_std * (1.0 - gen / generations)
        all_noise_params = build_batched_noise_params(candidate_noises, candidate_signs, current_sigma, device)

        # How many candidates per sub-batch
        cands_per_sub_batch = max(1, max_batch_sequences // n_prompts)

        # Storage
        candidate_responses: dict[int, list] = {i: [] for i in range(n_candidates)}
        candidate_judge_tasks: dict[int, list] = {i: [] for i in range(n_candidates)}
        candidate_kls: dict[int, float] = {}

        # --- Pipeline: generate → trace + judge → KL ---
        for batch_start in range(0, n_candidates, cands_per_sub_batch):
            batch_end = min(batch_start + cands_per_sub_batch, n_candidates)
            batch_cand_indices = list(range(batch_start, batch_end))
            batch_n = len(batch_cand_indices)

            sub_batch_texts = prompt_texts * batch_n
            sub_batch_prompts = step_prompts * batch_n
            sub_batch_candidate_ids = []
            for cand_idx in batch_cand_indices:
                sub_batch_candidate_ids.extend([cand_idx] * n_prompts)

            # (a) Generate with noise hooks (base LoRA stays active)
            t0 = time.perf_counter()
            sub_results = model.generate_sub_batch(
                sub_batch_prompts, sub_batch_texts, sub_batch_candidate_ids, all_noise_params, max_new_tokens
            )
            t_gen = time.perf_counter() - t0

            # (b) Write traces + fire judge calls
            for cand_idx, resp in sub_results:
                candidate_responses[cand_idx].append(resp)
                prompt_idx = len(candidate_responses[cand_idx]) - 1

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

                task = judge.submit(resp.prompt, resp.content)
                candidate_judge_tasks[cand_idx].append(task)
                pbar.update(1)

            # (c) KL — no adapter swap needed, just compute with base LoRA
            # (noise was additive via hooks, not baked into weights)
            # KL for all candidates in this sub-batch is the SAME (base LoRA unchanged)
            # So we compute once and share
            kl = model.compute_kl(good_prompts, base_logprobs, batch_size=batch_size)
            for cand_idx in batch_cand_indices:
                candidate_kls[cand_idx] = kl

            judge.run_pending()

            log.info(
                "sub_batch_done",
                generation=gen + 1,
                candidates=f"{batch_start + 1}-{batch_end}/{n_candidates}",
                t_gen=f"{t_gen:.1f}s",
            )

        # --- Await judge + score ---
        scores_plus: list[float] = []
        scores_minus: list[float] = []

        for cand_idx, (sign_label, pair_idx) in enumerate(candidate_labels):
            tasks = candidate_judge_tasks[cand_idx]
            results = judge.await_all(tasks, pbar=pbar)

            for prompt_idx, r in enumerate(results):
                trace.write_verdict(
                    generation=gen + 1,
                    candidate=cand_idx + 1,
                    prompt_idx=prompt_idx,
                    category=r.category,
                    reward=r.reward,
                    is_refusal=r.is_refusal,
                    judge_lengths=r.lengths,
                    judge_reasoning=r.reasoning,
                    judge_error=r.error,
                )

            # Compute mean reward from judge categories
            mean_reward = sum(r.reward for r in results) / len(results)
            refusals = sum(1 for r in results if r.is_refusal)
            n_errors = sum(1 for r in results if r.error is not None)
            kl = candidate_kls[cand_idx]
            score = mean_reward - kl_weight * kl

            categories = Counter(r.category for r in results)

            if sign_label == "+":
                scores_plus.append(score)
            else:
                scores_minus.append(score)

            gen_results = candidate_responses[cand_idx]
            total_tok = sorted(r.total_tokens for r in gen_results)
            judge_total_tok = sorted(r.lengths.total_tokens for r in results if r.lengths is not None)

            log.info(
                "candidate_eval",
                generation=gen + 1,
                candidate=f"{sign_label}{pair_idx + 1}",
                mean_reward=round(mean_reward, 3),
                categories=dict(categories),
                refusals=refusals,
                judge_errors=n_errors,
                kl=round(kl, 4),
                score=round(score, 4),
                student_tok_p50=total_tok[len(total_tok) // 2] if total_tok else 0,
                student_tok_max=total_tok[-1] if total_tok else 0,
                judge_tok_max=judge_total_tok[-1] if judge_total_tok else 0,
            )

            for r in results:
                if r.error is not None:
                    log.warning("judge_error", gen=gen + 1, cand=cand_idx + 1, error=r.error)

        # --- Fitness normalization (like EGGROLL) ---
        # Normalize all scores to zero mean, unit variance.
        # This makes gradient magnitude independent of absolute reward scale.
        all_scores = scores_plus + scores_minus
        score_mean = sum(all_scores) / len(all_scores)
        score_var = sum((s - score_mean) ** 2 for s in all_scores) / len(all_scores)
        score_std = max((score_var + 1e-8) ** 0.5, 1e-8)
        normalized_plus = [(s - score_mean) / score_std for s in scores_plus]
        normalized_minus = [(s - score_mean) / score_std for s in scores_minus]

        # --- ES gradient update (per-module, structured) ---
        # Sigma decays linearly over generations (like EGGROLL)
        current_sigma = noise_std * (1.0 - gen / generations)

        for name, param in model.model.named_parameters():
            if "lora_" not in name or not param.requires_grad:
                continue

            base_name = name.split(".lora_A.")[0] if ".lora_A." in name else name.split(".lora_B.")[0]
            is_A = ".lora_A." in name

            grad = torch.zeros_like(param)
            for pair_idx, noise in enumerate(noise_samples):
                if base_name not in noise.module_noise:
                    continue
                noise_A, noise_B = noise.module_noise[base_name]
                # Use normalized fitness-weighted noise (like EGGROLL)
                np_score = normalized_plus[pair_idx]
                nm_score = normalized_minus[pair_idx]

                if is_A:
                    noise_param = noise_A.to(param.device, param.dtype)
                    grad += np_score * noise_param  # +noise direction
                    grad += nm_score * (-noise_param)  # -noise direction (sign is in the noise)
                else:
                    noise_param = noise_B.to(param.device, param.dtype)
                    grad += np_score * noise_param
                    grad += nm_score * noise_param  # B is not sign-flipped in EGGROLL

            grad /= n_candidates
            param.data += learning_rate * grad

        mean_score = sum(scores_plus) / len(scores_plus)
        log.info(
            "es_update",
            generation=gen + 1,
            sigma=round(current_sigma, 6),
            mean_score_plus=round(sum(scores_plus) / len(scores_plus), 4),
            mean_score_minus=round(sum(scores_minus) / len(scores_minus), 4),
            score_std=round(score_std, 4),
        )

        if mean_score > best_score:
            best_score = mean_score
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
    return model.get_lora_flat()
