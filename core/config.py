"""Central configuration: model identifiers, weight paths and prompt template.

Every value can be overridden with an environment variable so the app can be
pointed at a different checkpoint without touching the code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------- #
# Prompt template (must match training byte-for-byte)
# <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n[IMAGE_EMBEDS]{question}<|im_end|>\n<|im_start|>assistant\n
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "<|im_start|>system\nYou are a helpful assistant that can see images and answer "
    "questions directly and concisely.<|im_end|>\n<|im_start|>user\n"
)
USER_SUFFIX_TEMPLATE = "{question}<|im_end|>\n<|im_start|>assistant\n"

# SigLIP-base-patch16-384 -> (384 / 16)^2 = 24 x 24 = 576 patch tokens
IMAGE_GRID_SIZE = 24
NUM_IMAGE_TOKENS = IMAGE_GRID_SIZE * IMAGE_GRID_SIZE


def _resolve_weights_dir() -> Path:
    """Find the folder that contains ``projector.pt`` and ``qwen_lora/``.

    Priority: ``$VLM_WEIGHTS_DIR`` -> ``models/epoch_2`` -> ``models``.
    """
    env_dir = os.getenv("VLM_WEIGHTS_DIR")
    candidates = [Path(env_dir)] if env_dir else []
    candidates += [PROJECT_ROOT / "models" / "epoch_2", PROJECT_ROOT / "models"]
    for cand in candidates:
        if (cand / "projector.pt").is_file() and (cand / "qwen_lora").is_dir():
            return cand
    searched = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Could not find 'projector.pt' + 'qwen_lora/' in any of: {searched}. "
        "Set the VLM_WEIGHTS_DIR environment variable to the checkpoint folder."
    )


@dataclass(frozen=True)
class ModelConfig:
    vision_model_id: str = os.getenv("VLM_VISION_ID", "google/siglip-base-patch16-384")
    llm_model_id: str = os.getenv("VLM_LLM_ID", "Qwen/Qwen2.5-3B-Instruct")
    weights_dir: Path = field(default_factory=_resolve_weights_dir)

    @property
    def projector_path(self) -> Path:
        return self.weights_dir / "projector.pt"

    @property
    def lora_path(self) -> Path:
        return self.weights_dir / "qwen_lora"
