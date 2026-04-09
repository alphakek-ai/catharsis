"""Model loading, LoRA perturbation, and generation."""

import math
import re

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from .arch import ArchConfig, detect_arch
from .log import log


class Model:
    def __init__(
        self,
        model_name: str,
        lora_rank: int = 1,
        lora_targets: list[str] | None = None,
        dtype: str = "auto",
        device_map: str = "auto",
    ):
        self.model_name = model_name
        self.lora_rank = lora_rank

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.base_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, device_map=device_map
        )
        self.base_model.eval()

        # Detect architecture and configure LoRA targets
        self.arch: ArchConfig = detect_arch(self.base_model.config)
        self.lora_targets = lora_targets or self.arch.default_lora_targets
        log.info("arch_detected", model_type=getattr(self.base_model.config, "model_type", "unknown"), layers_path=self.arch.layers_path, lora_targets=self.lora_targets)

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
        self.model: PeftModel = get_peft_model(self.base_model, self.peft_config)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def get_lora_params(self) -> list[Tensor]:
        """Get all LoRA A and B weight tensors."""
        return [p for _, p in self.model.named_parameters() if "lora_" in _ and p.requires_grad]

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

    def generate_responses(
        self,
        prompts: list[str],
        system_prompt: str = "You are a helpful assistant.",
        max_new_tokens: int = 1000,
        batch_size: int = 32,
    ) -> list[str]:
        """Generate responses for a list of prompts."""
        return list(self.generate_responses_iter(prompts, system_prompt, max_new_tokens, batch_size))

    def generate_responses_iter(
        self,
        prompts: list[str],
        system_prompt: str = "You are a helpful assistant.",
        max_new_tokens: int = 1000,
        batch_size: int = 32,
    ):
        """Yield (prompt, response) pairs as each batch completes."""
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            chats = [[{"role": "system", "content": system_prompt}, {"role": "user", "content": p}] for p in batch]
            chat_texts = self.tokenizer.apply_chat_template(chats, add_generation_prompt=True, tokenize=False)
            inputs = self.tokenizer(chat_texts, return_tensors="pt", padding=True, return_token_type_ids=False).to(
                self.device
            )
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            for j, output in enumerate(outputs):
                response = self.tokenizer.decode(output[inputs["input_ids"].shape[1] :], skip_special_tokens=True)
                yield batch[j], response

    def get_logprobs(
        self,
        prompts: list[str],
        system_prompt: str = "You are a helpful assistant.",
        batch_size: int = 32,
    ) -> Tensor:
        """Get first-token log probability distributions for KL computation."""
        all_logprobs = []
        n_batches = math.ceil(len(prompts) / batch_size)
        for i in tqdm(range(0, len(prompts), batch_size), total=n_batches, desc="Logprobs", leave=False):
            batch = prompts[i : i + batch_size]
            chats = [[{"role": "system", "content": system_prompt}, {"role": "user", "content": p}] for p in batch]
            chat_texts = self.tokenizer.apply_chat_template(chats, add_generation_prompt=True, tokenize=False)
            inputs = self.tokenizer(chat_texts, return_tensors="pt", padding=True, return_token_type_ids=False).to(
                self.device
            )
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

    def compute_kl(self, good_prompts: list[str], base_logprobs: Tensor, batch_size: int = 32) -> float:
        """Compute KL divergence from base model on good prompts."""
        current_logprobs = self.get_logprobs(good_prompts, batch_size=batch_size)
        return F.kl_div(current_logprobs, base_logprobs, reduction="batchmean", log_target=True).item()
