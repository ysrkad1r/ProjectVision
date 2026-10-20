"""Model loading, 4-bit quantisation, projector definition and VRAM management.

Memory budget on a 6 GB GPU (approx.):
    Qwen2.5-3B nf4 + double-quant  ~2.0 GB
    QLoRA adapter (r=16)            ~0.12 GB
    SigLIP *vision tower only*      ~0.19 GB  (text tower is never loaded)
    Projector (bf16)                ~0.01 GB
"""
from __future__ import annotations

import gc
import logging
import threading
from dataclasses import dataclass

import torch
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizerBase,
    SiglipVisionModel,
)

from core.config import ModelConfig

logger = logging.getLogger(__name__)

DTYPE = torch.bfloat16


class VisionProjector(nn.Module):
    """2-layer MLP mapping SigLIP features (768) into Qwen's embedding space (2048).

    The attribute name ``MLP`` is kept identical to training so that the saved
    ``state_dict`` keys (``MLP.0.weight`` ...) load without remapping.
    """

    def __init__(self, vision_dim: int = 768, llm_dim: int = 2048) -> None:
        super().__init__()
        self.MLP = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.MLP(x)


@dataclass
class VLMBundle:
    """Everything needed for a forward pass, loaded once and shared."""

    vision_model: SiglipVisionModel
    projector: VisionProjector
    llm: nn.Module
    tokenizer: PreTrainedTokenizerBase
    processor: AutoProcessor
    device: torch.device
    num_llm_layers: int


# --------------------------------------------------------------------------- #
# Memory helpers
# --------------------------------------------------------------------------- #
def free_memory() -> None:
    """Release Python garbage and cached CUDA blocks."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def vram_report() -> str:
    """Short human readable VRAM usage string."""
    if not torch.cuda.is_available():
        return "CPU"
    alloc = torch.cuda.memory_allocated() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return f"{alloc:.2f} GB aktif · {peak:.2f} GB tepe · {total:.1f} GB toplam"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _freeze(module: nn.Module) -> nn.Module:
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module


def _load_vision(cfg: ModelConfig, device: torch.device):
    # Only the vision tower: saves ~0.4 GB compared with the full SigLIP model.
    vision = SiglipVisionModel.from_pretrained(cfg.vision_model_id, dtype=DTYPE).to(device)
    processor = AutoProcessor.from_pretrained(cfg.vision_model_id)
    return _freeze(vision), processor


def _load_projector(cfg: ModelConfig, device: torch.device) -> VisionProjector:
    projector = VisionProjector()
    state = torch.load(cfg.projector_path, map_location="cpu", weights_only=True)
    projector.load_state_dict(state)
    del state
    return _freeze(projector.to(device=device, dtype=DTYPE))


def _load_llm(cfg: ModelConfig, device: torch.device):
    from peft import PeftModel  # local import keeps module import cheap

    tokenizer = AutoTokenizer.from_pretrained(cfg.llm_model_id)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=DTYPE,
    )
    base = AutoModelForCausalLM.from_pretrained(
        cfg.llm_model_id,
        quantization_config=bnb_config,
        dtype=DTYPE,
        device_map={"": device.index or 0},
        # "eager" is required: SDPA / FlashAttention kernels never materialise
        # the attention matrix, so output_attentions would return None.
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    # Same bf16 lock that was applied during training.
    for name, module in base.named_modules():
        if "norm" in name or "lm_head" in name or "embed_tokens" in name:
            module.to(DTYPE)

    llm = PeftModel.from_pretrained(base, str(cfg.lora_path), is_trainable=False)
    llm.config.use_cache = True
    return _freeze(llm), tokenizer


def load_vlm(cfg: ModelConfig | None = None) -> VLMBundle:
    """Load the full 2-stage VLM onto the GPU."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU bulunamadı. 4-bit (bitsandbytes nf4) Qwen modeli bir NVIDIA GPU gerektirir."
        )
    cfg = cfg or ModelConfig()
    device = torch.device("cuda", 0)
    logger.info("Loading weights from %s", cfg.weights_dir)

    vision, processor = _load_vision(cfg, device)
    projector = _load_projector(cfg, device)
    llm, tokenizer = _load_llm(cfg, device)
    free_memory()
    torch.cuda.reset_peak_memory_stats()

    num_layers = llm.get_base_model().config.num_hidden_layers
    logger.info("VLM ready (%d LLM layers) | %s", num_layers, vram_report())
    return VLMBundle(vision, projector, llm, tokenizer, processor, device, num_layers)


_bundle: VLMBundle | None = None
_bundle_lock = threading.Lock()


def get_vlm() -> VLMBundle:
    """Thread-safe lazy singleton so the 2 GB model is loaded exactly once."""
    global _bundle
    if _bundle is None:
        with _bundle_lock:
            if _bundle is None:
                _bundle = load_vlm()
    return _bundle
