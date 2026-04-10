"""Model loading, LoRA perturbation, and generation with continuous batching."""

import re
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig, PreTrainedModel
from transformers.generation import ContinuousBatchingConfig

from .arch import ArchConfig, detect_arch
from .log import log


@dataclass
class GeneratedResponse:
    """A single generated response with reasoning/content split."""

    prompt: str
    content: str  # The actual answer (post-thinking)
    reasoning: str  # The thinking trace (if any)
    raw: str  # Full raw output with special tokens
    total_tokens: int  # Total generated tokens
    reasoning_tokens: int  # Tokens in the thinking trace
    content_tokens: int  # Tokens in the actual answer


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
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self.processor.tokenizer.padding_side = "left"

        self.base_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=device_map,
            attn_implementation="flash_attention_2",
        )
        self.base_model.eval()

        # Detect architecture and configure LoRA targets
        self.arch: ArchConfig = detect_arch(self.base_model.config)
        self.lora_targets = lora_targets or self.arch.default_lora_targets
        log.info(
            "arch_detected",
            model_type=getattr(self.base_model.config, "model_type", "unknown"),
            layers_path=self.arch.layers_path,
            lora_targets=self.lora_targets,
        )

        # Scope LoRA to the transformer layers path to avoid unsupported
        # module types in vision/audio towers (e.g. Gemma4ClippableLinear).
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

        # Generation config
        self.generation_config = GenerationConfig(
            max_new_tokens=2000,
            do_sample=False,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )

        # Continuous batching config
        self.cb_config = ContinuousBatchingConfig(
            use_cuda_graph=True,
            return_logprobs=False,
        )

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def get_lora_params(self) -> list[Tensor]:
        """Get all LoRA A and B weight tensors."""
        return [p for _, p in self.model.named_parameters() if "lora_" in _ and p.requires_grad]

    def get_lora_named_params(self) -> tuple[list[str], list[Tensor]]:
        """Get LoRA param names and tensors (for saving adapters)."""
        names = []
        params = []
        for name, p in self.model.named_parameters():
            if "lora_" in name and p.requires_grad:
                names.append(name)
                params.append(p)
        return names, params

    def zero_lora(self):
        """Reset all LoRA weights to zero (identity)."""
        for param in self.get_lora_params():
            param.data.zero_()

    def set_lora_from_flat(self, flat: Tensor):
        """Set LoRA params from a flat vector."""
        offset = 0
        for param in self.get_lora_params():
            n = param.numel()
            param.data.copy_(flat[offset : offset + n].view(param.shape))
            offset += n

    def get_lora_flat(self) -> Tensor:
        """Get all LoRA params as a single flat vector."""
        return torch.cat([p.data.flatten() for p in self.get_lora_params()])

    def lora_param_count(self) -> int:
        return sum(p.numel() for p in self.get_lora_params())

    def _tokenize_prompts(
        self, prompts: list[str], system_prompt: str = "You are a helpful assistant."
    ) -> list[list[int]]:
        """Tokenize prompts with chat template. Returns list of token ID lists."""
        chats = [[{"role": "system", "content": system_prompt}, {"role": "user", "content": p}] for p in prompts]
        texts = self.processor.apply_chat_template(
            chats,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=self.enable_thinking,
        )
        return [self.processor.tokenizer.encode(t) for t in texts]

    def _parse_output(self, prompt: str, generated_tokens: list[int]) -> GeneratedResponse:
        """Parse generated token IDs into a GeneratedResponse."""
        tokenizer = self.processor.tokenizer
        raw = tokenizer.decode(generated_tokens, skip_special_tokens=False)

        parsed = self.processor.parse_response(raw)
        if isinstance(parsed, dict):
            content = parsed.get("content", raw)
            reasoning = parsed.get("thinking", "")
        else:
            content = str(parsed)
            reasoning = ""

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

    def generate_responses(
        self,
        prompts: list[str],
        system_prompt: str = "You are a helpful assistant.",
        max_new_tokens: int = 2000,
    ) -> list[GeneratedResponse]:
        """Generate responses using continuous batching. Returns when all are done."""
        inputs = self._tokenize_prompts(prompts, system_prompt)

        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )

        with torch.no_grad():
            outputs = self.model.generate_batch(
                inputs=inputs,
                generation_config=gen_config,
                continuous_batching_config=self.cb_config,
                progress_bar=False,
            )

        # Map results back to prompts (outputs keyed by request_id)
        results = []
        for i, prompt in enumerate(prompts):
            request_id = str(i)
            if request_id in outputs:
                output = outputs[request_id]
                results.append(self._parse_output(prompt, output.generated_tokens))
            else:
                # Fallback — shouldn't happen
                results.append(
                    GeneratedResponse(
                        prompt=prompt,
                        content="",
                        reasoning="",
                        raw="",
                        total_tokens=0,
                        reasoning_tokens=0,
                        content_tokens=0,
                    )
                )
        return results

    def generate_responses_streaming(
        self,
        prompts: list[str],
        system_prompt: str = "You are a helpful assistant.",
        max_new_tokens: int = 2000,
    ):
        """Generate responses with continuous batching, yielding each as it completes.

        Short responses (e.g. refusals) finish first and are yielded immediately,
        while long responses continue generating. No batch-level blocking.
        """
        inputs = self._tokenize_prompts(prompts, system_prompt)
        prompt_map = {}  # request_id -> prompt text

        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )

        with self.model.continuous_batching_context_manager(
            generation_config=gen_config,
            continuous_batching_config=self.cb_config,
        ) as manager:
            for i, (prompt, input_ids) in enumerate(zip(prompts, inputs, strict=True)):
                request_id = str(i)
                prompt_map[request_id] = prompt
                manager.add_request(input_ids=input_ids, request_id=request_id)

            for result in manager:
                prompt = prompt_map[result.request_id]
                yield self._parse_output(prompt, result.generated_tokens)

    def get_logprobs(
        self,
        prompts: list[str],
        system_prompt: str = "You are a helpful assistant.",
        batch_size: int = 32,
    ) -> Tensor:
        """Get first-token log probability distributions for KL computation."""
        all_logprobs = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            chats = [[{"role": "system", "content": system_prompt}, {"role": "user", "content": p}] for p in batch]
            texts = self.processor.apply_chat_template(
                chats,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=self.enable_thinking,
            )
            inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    output_scores=True,
                    return_dict_in_generate=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    do_sample=False,
                )
            logits = outputs.scores[0]
            all_logprobs.append(F.log_softmax(logits, dim=-1).cpu())
        return torch.cat(all_logprobs, dim=0)

    def compute_kl(self, good_prompts: list[str], base_logprobs: Tensor, batch_size: int = 32) -> float:
        """Compute KL divergence from base model on good prompts."""
        current_logprobs = self.get_logprobs(good_prompts, batch_size=batch_size)
        return F.kl_div(current_logprobs, base_logprobs, reduction="batchmean", log_target=True).item()
