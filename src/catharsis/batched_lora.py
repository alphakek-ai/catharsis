"""Batched per-sample noise corrections via forward hooks.

Applies structured low-rank noise corrections on top of the base model
(with LoRA adapter active). Each sample in the batch gets a different
noise correction based on which candidate it belongs to.

For a linear layer with base output y = (W + lora_B @ lora_A) @ x:
    y_corrected = y + (x @ noise_A_i.T) @ noise_B_i.T

The base LoRA adapter stays enabled and shared. Only the noise differs
per sample. This is EGGROLL-style structured perturbation.
"""

import torch
from torch import Tensor, nn


class BatchedNoiseContext:
    """Context manager that applies per-sample noise corrections via hooks.

    The base model (with LoRA adapter) runs normally. This adds an
    EXTRA per-sample correction on top.

    Usage:
        with BatchedNoiseContext(model, module_noise_params, candidate_ids):
            outputs = model.generate(**inputs)
    """

    def __init__(
        self,
        model: nn.Module,
        module_noise_params: dict[str, tuple[Tensor, Tensor]],
        candidate_ids: Tensor,
    ):
        """
        Args:
            model: The model (with LoRA adapter active).
            module_noise_params: Maps module name -> (noise_A, noise_B) where:
                noise_A: (n_candidates, rank, d_in)
                noise_B: (n_candidates, d_out, rank)
            candidate_ids: Shape (batch_size,) mapping each sample to a candidate.
        """
        self.model = model
        self.module_noise_params = module_noise_params
        self.candidate_ids = candidate_ids
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self):
        for module_name, (A, B) in self.module_noise_params.items():
            module = _get_module(self.model, module_name)
            handle = module.register_forward_hook(self._make_hook(A, B))
            self._hooks.append(handle)
        return self

    def __exit__(self, *args):
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def _make_hook(self, A: Tensor, B: Tensor):
        """Create a hook that adds per-sample noise correction.

        A: (n_candidates, rank, d_in)
        B: (n_candidates, d_out, rank)
        """
        candidate_ids = self.candidate_ids

        def hook(module: nn.Module, inputs: tuple, output: Tensor) -> Tensor:
            x = inputs[0]  # (batch, seq_len, d_in)
            batch_size = x.shape[0]

            a = A[candidate_ids[:batch_size]].to(x.dtype)
            b = B[candidate_ids[:batch_size]].to(x.dtype)

            # correction = (x @ A.T) @ B.T
            proj = torch.bmm(x, a.transpose(-1, -2))
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
