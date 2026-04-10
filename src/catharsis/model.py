"""Model loading, LoRA perturbation, and generation primitives."""

import re
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoProcessor, PreTrainedModel

from .arch import ArchConfig, detect_arch
from .batched_lora import BatchedLoRAContext, build_module_lora_params
from .log import log


@dataclass
class GeneratedResponse:
    """A single generated response with reasoning/content split."""

    prompt: str
    content: str
    reasoning: str
    raw: str
    total_tokens: int
    reasoning_tokens: int
    content_tokens: int


class Model:
    def __init__(
        self,
        model_name: str,
        lora_rank: int = 1,
        lora_targets: list[str] | None = None,
        dtype: str = "auto",
        device_map: str = "auto",
        enable_thinking: bool = True,
    ):
        self.model_name = model_name
        self.lora_rank = lora_rank
        self.enable_thinking = enable_thinking

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.base_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, device_map=device_map
        )
        self.base_model.eval()

        self.arch: ArchConfig = detect_arch(self.base_model.config)
        self.lora_targets = lora_targets or self.arch.default_lora_targets
        log.info(
            "arch_detected",
            model_type=getattr(self.base_model.config, "model_type", "unknown"),
            layers_path=self.arch.layers_path,
            lora_targets=self.lora_targets,
        )

        leaf_names = "|".join(re.escape(t) for t in self.lora_targets)
        target_modules = f"{re.escape(self.arch.layers_path)}\\..*\\.({leaf_names})"

        self.peft_config = LoraConfig(
            r=lora_rank,
            target_modules=target_modules,
            lora_alpha=lora_rank,
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = cast(PeftModel, get_peft_model(self.base_model, self.peft_config))

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    # --- LoRA parameter management ---

    def get_lora_params(self) -> list[Tensor]:
        return [p for _, p in self.model.named_parameters() if "lora_" in _ and p.requires_grad]

    def get_lora_named_params(self) -> tuple[list[str], list[Tensor]]:
        names, params = [], []
        for name, p in self.model.named_parameters():
            if "lora_" in name and p.requires_grad:
                names.append(name)
                params.append(p)
        return names, params

    def get_lora_param_shapes(self) -> list[tuple[int, ...]]:
        return [tuple(p.shape) for p in self.get_lora_params()]

    def zero_lora(self):
        for param in self.get_lora_params():
            param.data.zero_()

    def set_lora_from_flat(self, flat: Tensor):
        offset = 0
        for param in self.get_lora_params():
            n = param.numel()
            param.data.copy_(flat[offset : offset + n].view(param.shape))
            offset += n

    def get_lora_flat(self) -> Tensor:
        return torch.cat([p.data.flatten() for p in self.get_lora_params()])

    def lora_param_count(self) -> int:
        return sum(p.numel() for p in self.get_lora_params())

    # --- Batched LoRA preparation ---

    def prepare_batched_lora(self, candidate_params: list[Tensor]) -> dict[str, tuple[Tensor, Tensor]]:
        """Precompute per-module stacked LoRA params for batched generation.

        Call once per generation, reuse across sub-batches.
        """
        lora_names, _ = self.get_lora_named_params()
        lora_shapes = self.get_lora_param_shapes()
        return build_module_lora_params(self.model, candidate_params, lora_names, lora_shapes, self.device)

    # --- Generation primitives ---

    def _parse_output(self, prompt: str, generated_tokens: list[int]) -> GeneratedResponse:
        tokenizer = self.tokenizer
        raw = tokenizer.decode(generated_tokens, skip_special_tokens=False)

        content = raw
        reasoning = ""
        if hasattr(self.processor, "parse_response") and hasattr(self.processor, "response_schema"):
            try:
                parsed = self.processor.parse_response(raw)
                if isinstance(parsed, dict):
                    content = parsed.get("content", raw)
                    reasoning = parsed.get("thinking", "")
            except (AttributeError, Exception):
                pass
        if content == raw:
            content = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        reasoning_tokens = len(tokenizer.encode(reasoning, add_special_tokens=False)) if reasoning else 0
        content_tokens = len(tokenizer.encode(content, add_special_tokens=False)) if content else 0

        return GeneratedResponse(
            prompt=prompt,
            content=content,
            reasoning=reasoning,
            raw=raw,
            total_tokens=len(generated_tokens),
            reasoning_tokens=reasoning_tokens,
            content_tokens=content_tokens,
        )

    def tokenize_prompts(self, prompts: list[str], system_prompt: str = "You are a helpful assistant.") -> list[str]:
        """Apply chat template to prompts. Returns list of formatted text strings."""
        chats = [[{"role": "system", "content": system_prompt}, {"role": "user", "content": p}] for p in prompts]
        return self.processor.apply_chat_template(
            chats, add_generation_prompt=True, tokenize=False, enable_thinking=self.enable_thinking
        )

    def generate_sub_batch(
        self,
        prompts: list[str],
        prompt_texts: list[str],
        candidate_indices: list[int],
        module_lora_params: dict[str, tuple[Tensor, Tensor]],
        max_new_tokens: int = 2000,
    ) -> list[tuple[int, GeneratedResponse]]:
        """Generate one sub-batch with per-sample LoRA hooks.

        Args:
            prompts: Original prompt strings (for parse_output).
            prompt_texts: Tokenizable texts (from tokenize_prompts), repeated per candidate.
            candidate_indices: Which candidate each sequence belongs to.
            module_lora_params: Precomputed from prepare_batched_lora.
            max_new_tokens: Max tokens to generate.

        Returns:
            List of (candidate_index, GeneratedResponse) pairs.
        """
        inputs = self.processor(text=prompt_texts, return_tensors="pt", padding=True).to(self.device)
        input_len = inputs["input_ids"].shape[-1]
        candidate_ids = torch.tensor(candidate_indices, device=self.device)

        self.model.disable_adapter_layers()
        try:
            with BatchedLoRAContext(self.model, module_lora_params, candidate_ids):
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
        finally:
            self.model.enable_adapter_layers()

        pad_id = self.tokenizer.pad_token_id
        results = []
        for idx, output in enumerate(outputs):
            generated_ids = output[input_len:]
            if pad_id is not None:
                generated_ids = generated_ids[generated_ids != pad_id]
            cand_idx = candidate_indices[idx]
            prompt = prompts[idx % len(prompts)] if len(prompts) < len(prompt_texts) else prompts[idx]
            results.append((cand_idx, self._parse_output(prompt, generated_ids.tolist())))
        return results

    # --- KL divergence ---

    def compute_kl(self, good_prompts: list[str], base_logprobs: Tensor, batch_size: int = 32) -> float:
        """Compute KL divergence from base model on good prompts. Uses current LoRA weights."""
        current_logprobs = self._get_logprobs(good_prompts, batch_size=batch_size)
        return F.kl_div(current_logprobs, base_logprobs, reduction="batchmean", log_target=True).item()

    def get_base_logprobs(self, good_prompts: list[str], batch_size: int = 32) -> Tensor:
        """Compute base model logprobs (call once at startup)."""
        return self._get_logprobs(good_prompts, batch_size=batch_size)

    def _get_logprobs(self, prompts: list[str], batch_size: int = 32) -> Tensor:
        all_logprobs = []
        texts = self.tokenize_prompts(prompts)
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = self.processor(text=batch_texts, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    output_scores=True,
                    return_dict_in_generate=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    do_sample=False,
                )
            logits = outputs.scores[0]
            all_logprobs.append(F.log_softmax(logits, dim=-1).cpu())
        return torch.cat(all_logprobs, dim=0)

    # --- Convenience (for non-batched use like saving) ---

    def generate_responses(
        self,
        prompts: list[str],
        system_prompt: str = "You are a helpful assistant.",
        max_new_tokens: int = 2000,
        batch_size: int = 32,
    ) -> list[GeneratedResponse]:
        """Simple sequential generation with current LoRA weights."""
        all_results = []
        texts = self.tokenize_prompts(prompts, system_prompt)
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_prompts = prompts[i : i + batch_size]
            inputs = self.processor(text=batch_texts, return_tensors="pt", padding=True).to(self.device)
            input_len = inputs["input_ids"].shape[-1]
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.tokenizer.pad_token_id
                )
            pad_id = self.tokenizer.pad_token_id
            for j, output in enumerate(outputs):
                generated_ids = output[input_len:]
                if pad_id is not None:
                    generated_ids = generated_ids[generated_ids != pad_id]
                all_results.append(self._parse_output(batch_prompts[j], generated_ids.tolist()))
        return all_results
