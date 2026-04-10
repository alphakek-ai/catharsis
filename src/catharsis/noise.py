"""Structured low-rank noise generation for evolution strategies.

Instead of flat noise across all LoRA parameters, generates structured
per-module rank-r perturbations like EGGROLL:

  delta_y = sigma * (x @ noise_A.T) @ noise_B.T

This is additive on top of whatever the base LoRA adapter produces.
Each module gets independent noise, scaled by 1/sqrt(rank) for
magnitude normalization.
"""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class StructuredNoise:
    """Per-module noise matrices for one perturbation direction.

    module_noise maps base_module_name -> (noise_A, noise_B) where:
        noise_A: (rank, d_in)  — projects input to noise subspace
        noise_B: (d_out, rank) — projects noise subspace to output
    """

    module_noise: dict[str, tuple[Tensor, Tensor]]


def generate_structured_noise(
    lora_param_names: list[str],
    lora_param_shapes: list[tuple[int, ...]],
    rank: int,
    device: torch.device,
) -> StructuredNoise:
    """Generate one structured noise sample.

    For each target module, generates random noise_A and noise_B.
    Scaled by 1/sqrt(rank) so the perturbation magnitude is
    independent of rank (following EGGROLL).
    """
    module_d_out: dict[str, int] = {}
    module_d_in: dict[str, int] = {}

    for name, shape in zip(lora_param_names, lora_param_shapes, strict=True):
        if ".lora_A." in name:
            base_name = name.split(".lora_A.")[0]
            module_d_in[base_name] = shape[-1]
        elif ".lora_B." in name:
            base_name = name.split(".lora_B.")[0]
            module_d_out[base_name] = shape[0]

    module_noise = {}
    scale = 1.0 / (rank**0.5)
    for base_name in module_d_out:
        d_out = module_d_out.get(base_name, 0)
        d_in = module_d_in.get(base_name, 0)
        if d_out > 0 and d_in > 0:
            noise_A = torch.randn(rank, d_in, device=device) * scale
            noise_B = torch.randn(d_out, rank, device=device)
            module_noise[base_name] = (noise_A, noise_B)

    return StructuredNoise(module_noise=module_noise)


def build_batched_noise_params(
    noises: list[StructuredNoise],
    signs: list[float],
    sigma: float,
    device: torch.device,
) -> dict[str, tuple[Tensor, Tensor]]:
    """Stack noise across candidates for batched hook application.

    Returns dict mapping base_module_name -> (stacked_A, stacked_B) where:
        stacked_A: (n_candidates, rank, d_in)
        stacked_B: (n_candidates, d_out, rank)

    Each candidate's contribution is: sign * sigma * noise_A, noise_B
    (sigma scales A only, like EGGROLL)
    """
    all_module_names = set()
    for n in noises:
        all_module_names.update(n.module_noise.keys())

    result: dict[str, tuple[Tensor, Tensor]] = {}
    for mod_name in all_module_names:
        As = []
        Bs = []
        for noise, sign in zip(noises, signs, strict=True):
            noise_A, noise_B = noise.module_noise[mod_name]
            As.append(sign * sigma * noise_A)
            Bs.append(noise_B)  # B not scaled by sigma, like EGGROLL
        result[mod_name] = (torch.stack(As).to(device), torch.stack(Bs).to(device))

    return result
