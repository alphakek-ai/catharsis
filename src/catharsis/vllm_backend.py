"""vLLM multi-LoRA backend — parallel generation across candidates."""

import asyncio
import json
import shutil
import tempfile
from pathlib import Path

import torch
from openai import AsyncOpenAI
from peft import LoraConfig
from safetensors.torch import save_file
from tqdm import tqdm

from .log import log


class VLLMBackend:
    """Handles multi-LoRA generation via a vLLM server."""

    def __init__(
        self,
        base_url: str,
        base_model_name: str,
        lora_config: LoraConfig,
        concurrency: int = 64,
    ):
        self.client = AsyncOpenAI(base_url=base_url, api_key="unused", timeout=86400.0)
        self.base_model_name = base_model_name
        self.lora_config = lora_config
        self.concurrency = concurrency
        self._adapter_dir = Path(tempfile.mkdtemp(prefix="catharsis_adapters_"))
        self._loaded_adapters: set[str] = set()
        log.info("vllm_backend_init", base_url=base_url, adapter_dir=str(self._adapter_dir))

    def save_adapter(self, name: str, lora_params: list[torch.Tensor], param_names: list[str]) -> Path:
        """Save LoRA weights as a peft-compatible adapter directory."""
        adapter_path = self._adapter_dir / name
        if adapter_path.exists():
            shutil.rmtree(adapter_path)
        adapter_path.mkdir(parents=True)

        # Build state dict with peft naming convention
        state_dict = {}
        for param_name, param in zip(param_names, lora_params, strict=True):
            # param_name is like "base_model.model.model.language_model.layers.0.mlp.down_proj.lora_A.default.weight"
            # We need to keep it as-is for peft compatibility
            state_dict[param_name] = param.cpu().contiguous()

        save_file(state_dict, adapter_path / "adapter_model.safetensors")

        # Save adapter config
        config_dict = {
            "base_model_name_or_path": self.base_model_name,
            "peft_type": "LORA",
            "r": self.lora_config.r,
            "lora_alpha": self.lora_config.lora_alpha,
            "target_modules": [self.lora_config.target_modules]
            if isinstance(self.lora_config.target_modules, str)
            else list(self.lora_config.target_modules or []),
            "bias": self.lora_config.bias,
            "task_type": self.lora_config.task_type,
            "lora_dropout": self.lora_config.lora_dropout,
        }
        with open(adapter_path / "adapter_config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

        return adapter_path

    async def load_adapter(self, name: str, path: Path):
        """Hot-load an adapter into the running vLLM server."""
        import httpx

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self.client.base_url}load_lora_adapter".replace("/v1/", "/v1/"),
                json={"lora_name": name, "lora_path": str(path)},
            )
            if resp.status_code != 200:
                raise RuntimeError(f"Failed to load adapter {name}: {resp.text}")
        self._loaded_adapters.add(name)

    async def unload_adapter(self, name: str):
        """Unload an adapter from the vLLM server."""
        import httpx

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self.client.base_url}unload_lora_adapter".replace("/v1/", "/v1/"),
                json={"lora_name": name},
            )
            if resp.status_code != 200:
                log.warning("unload_adapter_failed", name=name, status=resp.status_code)
        self._loaded_adapters.discard(name)

    async def generate_one(
        self,
        adapter_name: str,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        max_tokens: int = 2000,
    ) -> dict:
        """Generate a single response. Returns dict with content, reasoning, raw, total_tokens."""
        result = await self.client.chat.completions.create(
            model=adapter_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0,
        )
        msg = result.choices[0].message
        content = msg.content or ""
        reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None) or ""
        total_tokens = result.usage.completion_tokens if result.usage else 0

        return {
            "content": content,
            "reasoning": reasoning,
            "raw": content,  # vLLM strips special tokens by default
            "total_tokens": total_tokens,
            "reasoning_tokens": result.usage.completion_tokens_details.reasoning_tokens
            if result.usage and result.usage.completion_tokens_details
            else 0,
        }

    async def generate_batch(
        self,
        adapter_names: list[str],
        prompts: list[str],
        system_prompt: str = "You are a helpful assistant.",
        max_tokens: int = 2000,
        pbar: tqdm | None = None,
    ) -> dict[str, list[dict]]:
        """Generate responses for all adapter×prompt combinations in parallel.

        Returns: {adapter_name: [response_dict, ...]}
        """
        sem = asyncio.Semaphore(self.concurrency)
        results: dict[str, list[dict | None]] = {name: [None] * len(prompts) for name in adapter_names}

        async def gen_one(adapter_name: str, prompt_idx: int, prompt: str):
            async with sem:
                resp = await self.generate_one(adapter_name, prompt, system_prompt, max_tokens)
                results[adapter_name][prompt_idx] = resp
                if pbar:
                    pbar.update(1)

        tasks = [
            gen_one(adapter_name, idx, prompt) for adapter_name in adapter_names for idx, prompt in enumerate(prompts)
        ]
        await asyncio.gather(*tasks)

        return {name: [r for r in resps if r is not None] for name, resps in results.items()}

    def cleanup(self):
        """Remove temporary adapter files."""
        shutil.rmtree(self._adapter_dir, ignore_errors=True)
