"""CLI entry point for catharsis."""

from typing import Annotated, Optional

import typer

app = typer.Typer(help="Catharsis — LLM re-alignment toolkit")


@app.command()
def run(
    model: Annotated[str, typer.Option(help="Model ID or path")],
    judge_api_base: Annotated[str, typer.Option(help="LLM judge API base URL")],
    judge_model: Annotated[Optional[str], typer.Option(help="Judge model name")] = None,
    judge_concurrency: Annotated[int, typer.Option(help="Max concurrent judge requests")] = 32,
    lora_rank: Annotated[int, typer.Option(help="LoRA rank")] = 1,
    lora_targets: Annotated[Optional[list[str]], typer.Option(help="LoRA target modules")] = None,
    population_size: Annotated[int, typer.Option(help="Candidates per generation")] = 16,
    generations: Annotated[int, typer.Option(help="Number of generations")] = 100,
    noise_std: Annotated[float, typer.Option(help="Perturbation noise std")] = 0.001,
    kl_weight: Annotated[float, typer.Option(help="KL divergence penalty weight")] = 1.0,
    batch_size: Annotated[int, typer.Option(help="Inference batch size")] = 32,
    max_new_tokens: Annotated[int, typer.Option(help="Max tokens per response")] = 1000,
    n_eval: Annotated[int, typer.Option(help="Number of eval prompts")] = 100,
    output_dir: Annotated[Optional[str], typer.Option(help="Output directory")] = None,
):
    """Run evolutionary LoRA search with LLM judge fitness."""
    from .data import load_default_prompts
    from .evolve import evolve
    from .judge import Judge
    from .log import log, setup_logging
    from .model import Model

    setup_logging()

    log.info("loading_model", model=model, lora_rank=lora_rank, lora_targets=lora_targets)
    m = Model(model_name=model, lora_rank=lora_rank, lora_targets=lora_targets or None)
    log.info("model_loaded", lora_params=m.lora_param_count())

    log.info("setting_up_judge", api_base=judge_api_base, model=judge_model)
    judge = Judge(api_base=judge_api_base, model=judge_model, concurrency=judge_concurrency)
    log.info("judge_ready", model=judge.model)

    log.info("loading_prompts")
    good_train, bad_train, good_eval, bad_eval = load_default_prompts(n_eval=n_eval)
    log.info("prompts_loaded", good_eval=len(good_eval), bad_eval=len(bad_eval))

    log.info("computing_base_logprobs")
    base_logprobs = m.get_logprobs(good_eval, batch_size=batch_size)
    log.info("base_logprobs_ready")

    log.info("starting_evolution", population_size=population_size, generations=generations, noise_std=noise_std)
    evolve(
        model=m,
        judge=judge,
        good_prompts=good_eval,
        bad_prompts=bad_eval,
        base_logprobs=base_logprobs,
        population_size=population_size,
        generations=generations,
        noise_std=noise_std,
        kl_weight=kl_weight,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )

    out = output_dir or f"{model.replace('/', '--')}-catharsis"
    log.info("saving_model", output_dir=out)
    merged = m.model.merge_and_unload()
    merged.save_pretrained(out, safe_serialization=True)
    m.tokenizer.save_pretrained(out)
    log.info("done", output_dir=out)


def main():
    app()


if __name__ == "__main__":
    main()
