"""CLI entry point for catharsis."""

import argparse

import torch

from .data import load_default_prompts
from .evolve import evolve
from .judge import Judge
from .model import Model


def main():
    parser = argparse.ArgumentParser(description="Catharsis — LLM re-alignment toolkit")
    parser.add_argument("--model", required=True, help="Model ID or path")
    parser.add_argument("--judge-api-base", required=True, help="LLM judge API base URL")
    parser.add_argument("--judge-model", default=None, help="Judge model name (auto-detected if not set)")
    parser.add_argument("--judge-concurrency", type=int, default=32)
    parser.add_argument("--lora-rank", type=int, default=1)
    parser.add_argument("--lora-targets", nargs="+", default=["down_proj", "gate_proj", "up_proj"])
    parser.add_argument("--population-size", type=int, default=16)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--kl-weight", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=1000)
    parser.add_argument("--n-eval-good", type=int, default=100)
    parser.add_argument("--n-eval-bad", type=int, default=100)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    print("Loading model...")
    model = Model(
        model_name=args.model,
        lora_rank=args.lora_rank,
        lora_targets=args.lora_targets,
    )
    print(f"  LoRA params: {model.lora_param_count():,}")

    print("Setting up judge...")
    judge = Judge(
        api_base=args.judge_api_base,
        model=args.judge_model,
        concurrency=args.judge_concurrency,
    )

    print("Loading prompts...")
    good_train, bad_train, good_eval, bad_eval = load_default_prompts(
        n_eval=args.n_eval_bad,
    )

    print("Computing base model logprobs on good prompts...")
    base_logprobs = model.get_logprobs(good_eval, batch_size=args.batch_size)

    print("Starting evolutionary search...")
    best_params = evolve(
        model=model,
        judge=judge,
        good_prompts=good_eval,
        bad_prompts=bad_eval,
        base_logprobs=base_logprobs,
        population_size=args.population_size,
        generations=args.generations,
        noise_std=args.noise_std,
        kl_weight=args.kl_weight,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    output_dir = args.output_dir or f"{args.model.replace('/', '--')}-catharsis"
    print(f"\nSaving model to {output_dir}...")
    merged = model.model.merge_and_unload()
    merged.save_pretrained(output_dir, safe_serialization=True)
    model.tokenizer.save_pretrained(output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
