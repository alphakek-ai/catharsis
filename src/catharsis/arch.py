"""Model architecture detection and configuration.

Each architecture defines how to find transformer layers and which
modules are safe to target with LoRA. New architectures are registered
by adding an entry to ARCHITECTURES.
"""

from dataclasses import dataclass


@dataclass
class ArchConfig:
    """Configuration for a specific model architecture."""

    # Module path to the transformer decoder layers (e.g. "model.layers")
    layers_path: str

    # Default LoRA target module leaf names
    default_lora_targets: list[str]


# Maps model_type (from config.json) to architecture config.
ARCHITECTURES: dict[str, ArchConfig] = {
    # Gemma 4 (multimodal, has ClippableLinear in vision/audio towers)
    "gemma4": ArchConfig(
        layers_path="model.language_model.layers",
        default_lora_targets=["down_proj", "gate_proj", "up_proj"],
    ),
    # Gemma 3
    "gemma3": ArchConfig(
        layers_path="model.language_model.layers",
        default_lora_targets=["down_proj", "gate_proj", "up_proj"],
    ),
    # Llama / Mistral / most dense models
    "llama": ArchConfig(
        layers_path="model.layers",
        default_lora_targets=["down_proj", "gate_proj", "up_proj"],
    ),
    "mistral": ArchConfig(
        layers_path="model.layers",
        default_lora_targets=["down_proj", "gate_proj", "up_proj"],
    ),
    # Qwen 2/3
    "qwen2": ArchConfig(
        layers_path="model.layers",
        default_lora_targets=["down_proj", "gate_proj", "up_proj"],
    ),
    "qwen3": ArchConfig(
        layers_path="model.layers",
        default_lora_targets=["down_proj", "gate_proj", "up_proj"],
    ),
    # Phi
    "phi3": ArchConfig(
        layers_path="model.layers",
        default_lora_targets=["down_proj", "gate_proj", "up_proj"],
    ),
}

# Fallback for unknown architectures
_DEFAULT = ArchConfig(
    layers_path="model.layers",
    default_lora_targets=["down_proj", "gate_proj", "up_proj"],
)


def detect_arch(model_config) -> ArchConfig:
    """Detect architecture from a transformers model config."""
    model_type = getattr(model_config, "model_type", None)

    # Some multimodal models nest the text config
    if model_type and model_type not in ARCHITECTURES:
        text_config = getattr(model_config, "text_config", None)
        if text_config:
            text_type = getattr(text_config, "model_type", None)
            if text_type and text_type in ARCHITECTURES:
                model_type = text_type

    if model_type and model_type in ARCHITECTURES:
        return ARCHITECTURES[model_type]

    return _DEFAULT
