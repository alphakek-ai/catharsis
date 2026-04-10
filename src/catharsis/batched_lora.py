"""Batched per-sample LoRA — different perturbations fused into one forward pass.

Instead of evaluating candidates sequentially (swap adapter, generate, repeat),
we batch all candidates' prompts together and apply per-sample LoRA corrections
via forward hooks. The base weights are shared; only the tiny rank-1 correction
differs per sample.

For rank-1 LoRA on a linear layer y = Wx:
    y_corrected = Wx + (x @ A_i.T) @ B_i.T
where A_i, B_i are the LoRA params for candidate i.

This is two cheap ops (a projection + a scale) per layer per token.
"""

import torch
from torch import Tensor, nn


class BatchedLoRAContext:
    """Context manager that applies per-sample LoRA corrections via hooks.

    Usage:
        with BatchedLoRAContext(model, module_lora_params, candidate_ids) as ctx:
            outputs = model.generate(**inputs)
    """

    def __init__(
        self,
        model: nn.Module,
        module_lora_params: dict[str, tuple[Tensor, Tensor]],
        candidate_ids: Tensor,
    ):
        """
        Args:
            model: The base model (with peft adapter disabled).
            module_lora_params: Maps module name -> (A, B) where:
                A has shape (n_candidates, rank, d_in)
                B has shape (n_candidates, d_out, rank)
            candidate_ids: Shape (batch_size,) mapping each sample to a candidate index.
        """
        self.model = model
        self.module_lora_params = module_lora_params
        self.candidate_ids = candidate_ids
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self):
        # Find and hook target modules
        for module_name, (A, B) in self.module_lora_params.items():
            module = _get_module(self.model, module_name)
            handle = module.register_forward_hook(self._make_hook(A, B))
            self._hooks.append(handle)
        return self

    def __exit__(self, *args):
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def _make_hook(self, A: Tensor, B: Tensor):
        """Create a forward hook that adds per-sample LoRA correction.

        A: (n_candidates, rank, d_in)
        B: (n_candidates, d_out, rank)
        """
        candidate_ids = self.candidate_ids

        def hook(module: nn.Module, inputs: tuple, output: Tensor) -> Tensor:
            x = inputs[0]  # (batch, seq_len, d_in)
            batch_size = x.shape[0]

            # Look up LoRA params for each sample's candidate
            a = A[candidate_ids[:batch_size]]  # (batch, rank, d_in)
            b = B[candidate_ids[:batch_size]]  # (batch, d_out, rank)

            # Compute correction: x @ A.T @ B.T
            # x: (batch, seq, d_in), a.T: (batch, d_in, rank) -> proj: (batch, seq, rank)
            proj = torch.bmm(x, a.transpose(-1, -2))
            # proj: (batch, seq, rank), b.T: (batch, rank, d_out) -> correction: (batch, seq, d_out)
            correction = torch.bmm(proj, b.transpose(-1, -2))

            return output + correction

        return hook


def _get_module(model: nn.Module, name: str) -> nn.Module:
    """Get a submodule by dotted name path."""
    parts = name.split(".")
    m = model
    for part in parts:
        m = getattr(m, part)
    return m


def build_module_lora_params(
    model: nn.Module,
    candidate_flat_params: list[Tensor],
    lora_param_names: list[str],
    lora_param_shapes: list[tuple[int, ...]],
    device: torch.device,
) -> dict[str, tuple[Tensor, Tensor]]:
    """Build per-module stacked LoRA params from flat parameter vectors.

    Groups lora_A and lora_B params by their parent module, stacks across candidates.

    Returns:
        Dict mapping base module name -> (A_stacked, B_stacked) where:
            A_stacked: (n_candidates, rank, d_in)
            B_stacked: (n_candidates, d_out, rank)
    """
    n_candidates = len(candidate_flat_params)

    # Parse param names to group A/B by module
    # Names look like: "base_model.model.model.language_model.layers.0.mlp.down_proj.lora_A.default.weight"
    module_params: dict[str, dict[str, list[Tensor]]] = {}

    for flat in candidate_flat_params:
        offset = 0
        for name, shape in zip(lora_param_names, lora_param_shapes, strict=True):
            n = 1
            for s in shape:
                n *= s
            param = flat[offset : offset + n].view(shape)
            offset += n

            # Extract base module path and whether this is A or B
            # "...down_proj.lora_A.default.weight" -> base="...down_proj", type="A"
            if ".lora_A." in name:
                base_name = name.split(".lora_A.")[0]
                ab_key = "A"
            elif ".lora_B." in name:
                base_name = name.split(".lora_B.")[0]
                ab_key = "B"
            else:
                continue

            if base_name not in module_params:
                module_params[base_name] = {"A": [], "B": []}
            module_params[base_name][ab_key].append(param)

    # Stack across candidates
    result: dict[str, tuple[Tensor, Tensor]] = {}
    for base_name, ab_dict in module_params.items():
        if len(ab_dict["A"]) == n_candidates and len(ab_dict["B"]) == n_candidates:
            A = torch.stack(ab_dict["A"]).to(device)  # (n_candidates, rank, d_in)
            B = torch.stack(ab_dict["B"]).to(device)  # (n_candidates, d_out, rank)
            result[base_name] = (A, B)

    return result
