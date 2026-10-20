"""Text generation and the visual question answering pipeline."""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from core.attention import (
    REFERENCE_WORDS,
    GroundingMaps,
    collect_baseline_attention,
    collect_image_attention,
    collect_logit_lens,
    is_meaningful_token,
)
from core.config import SYSTEM_PROMPT, USER_SUFFIX_TEMPLATE
from core.model import DTYPE, VLMBundle, free_memory

_BASELINE_QUESTION = "Describe the image briefly."


@dataclass
class GenerationSettings:
    temperature: float = 0.0  # 0.0 -> greedy / deterministic
    max_new_tokens: int = 35
    repetition_penalty: float = 1.2
    use_top_filtering: bool = False  # enables top-p / top-k (only when sampling)
    top_p: float = 0.9
    top_k: int = 50
    capture_attention: bool = True

    @property
    def do_sample(self) -> bool:
        return self.temperature > 0.0


@dataclass
class InferenceResult:
    answer: str
    attention: GroundingMaps | None
    num_tokens: int
    elapsed_s: float


@dataclass
class _PromptEmbeds:
    inputs_embeds: torch.Tensor  # [1, S, D] bf16
    attention_mask: torch.Tensor  # [1, S]
    image_start: int  # index of the first image token in the sequence
    image_embeds: torch.Tensor  # [1, 576, D] bf16 (kept to avoid re-running SigLIP)


@torch.inference_mode()
def _encode_image(bundle: VLMBundle, image: Image.Image) -> torch.Tensor:
    pixel_values = bundle.processor(images=image, return_tensors="pt")["pixel_values"]
    pixel_values = pixel_values.to(bundle.device, dtype=DTYPE)
    hidden = bundle.vision_model(pixel_values=pixel_values).last_hidden_state  # [1, 576, 768]
    return bundle.projector(hidden)  # [1, 576, 2048] bf16


@torch.inference_mode()
def build_prompt_embeds(
    bundle: VLMBundle,
    image: Image.Image,
    question: str,
    image_embeds: torch.Tensor | None = None,
) -> _PromptEmbeds:
    """[system+user prefix] + [576 image embeds] + [question + assistant prefix].

    The two text parts are tokenised separately, exactly as during training.
    """
    tok, dev = bundle.tokenizer, bundle.device
    prefix_ids = tok(SYSTEM_PROMPT, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
    suffix_ids = tok(
        USER_SUFFIX_TEMPLATE.format(question=question), return_tensors="pt", add_special_tokens=False
    ).input_ids.to(dev)

    embed = bundle.llm.get_input_embeddings()
    if image_embeds is None:
        image_embeds = _encode_image(bundle, image)
    inputs_embeds = torch.cat(
        [embed(prefix_ids).to(DTYPE), image_embeds, embed(suffix_ids).to(DTYPE)], dim=1
    )
    mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=dev)
    return _PromptEmbeds(inputs_embeds, mask, image_start=prefix_ids.shape[1], image_embeds=image_embeds)


def _generation_config(bundle: VLMBundle, s: GenerationSettings):
    cfg = copy.deepcopy(bundle.llm.generation_config)
    cfg.max_new_tokens = int(s.max_new_tokens)
    cfg.repetition_penalty = float(s.repetition_penalty)
    cfg.do_sample = s.do_sample
    if not s.do_sample:  # greedy: sampling params are irrelevant
        cfg.temperature = cfg.top_p = cfg.top_k = None
    elif s.use_top_filtering:
        cfg.temperature, cfg.top_p, cfg.top_k = float(s.temperature), float(s.top_p), int(s.top_k)
    else:  # pure temperature sampling; neutralise Qwen's default top_p=0.8 / top_k=20
        cfg.temperature, cfg.top_p, cfg.top_k = float(s.temperature), 1.0, 0
    cfg.use_cache = True
    cfg.output_attentions = s.capture_attention
    cfg.output_hidden_states = s.capture_attention
    cfg.return_dict_in_generate = True
    cfg.pad_token_id = bundle.tokenizer.pad_token_id or bundle.tokenizer.eos_token_id
    return cfg


def _strip_eos(ids: list[int], eos_ids: set[int]) -> list[int]:
    for i, t in enumerate(ids):
        if t in eos_ids:
            return ids[:i]
    return ids


_ref_ids_cache: dict[int, torch.Tensor] = {}


def _get_reference_ids(bundle: VLMBundle) -> torch.Tensor:
    key = id(bundle.tokenizer)
    if key not in _ref_ids_cache:
        ids = [bundle.tokenizer(" " + w, add_special_tokens=False).input_ids[0] for w in REFERENCE_WORDS]
        _ref_ids_cache[key] = torch.tensor(ids, device=bundle.device)
    return _ref_ids_cache[key]


@torch.inference_mode()
def _compute_baseline_attention(
    bundle: VLMBundle, image: Image.Image, image_embeds: torch.Tensor
) -> np.ndarray | None:
    """Run a single prefill with a generic prompt to measure baseline patch attention."""
    base_prompt = base_out = None
    try:
        base_prompt = build_prompt_embeds(bundle, image, _BASELINE_QUESTION, image_embeds=image_embeds)
        base_out = bundle.llm(
            inputs_embeds=base_prompt.inputs_embeds,
            attention_mask=base_prompt.attention_mask,
            output_attentions=True,
            use_cache=False,
        )
        return collect_baseline_attention(base_out.attentions, base_prompt.image_start)
    except Exception:
        return None
    finally:
        del base_prompt, base_out


def answer_question(
    bundle: VLMBundle, image: Image.Image, question: str, settings: GenerationSettings
) -> InferenceResult:
    """Run the full VQA pipeline and (optionally) capture image attention & logit lens."""
    question = question.strip()
    if not question:
        raise ValueError("Soru boş olamaz.")
    image = image.convert("RGB")
    t0 = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    prompt = out = None
    try:
        with torch.inference_mode():
            prompt = build_prompt_embeds(bundle, image, question)
            gen_cfg = _generation_config(bundle, settings)
            out = bundle.llm.generate(
                inputs_embeds=prompt.inputs_embeds,
                attention_mask=prompt.attention_mask,
                generation_config=gen_cfg,
            )
            # With inputs_embeds only, `sequences` holds just the new tokens.
            eos = gen_cfg.eos_token_id
            eos_ids = set(eos if isinstance(eos, list) else [eos])
            ids = _strip_eos(out.sequences[0].tolist(), eos_ids)

            grounding = None
            if settings.capture_attention and ids and out.attentions:
                tokens = [bundle.tokenizer.decode([t]) for t in ids]
                att_maps = collect_image_attention(out.attentions, prompt.image_start, len(ids))

                lens_maps = None
                if getattr(out, "hidden_states", None):
                    # out.hidden_states[0] is the prefill tuple of (L+1) tensors [1, S, D]
                    prefill_hs = out.hidden_states[0]
                    base_model = bundle.llm.get_base_model()
                    final_norm = base_model.model.norm
                    unembed = bundle.llm.get_output_embeddings().weight
                    ref_ids = _get_reference_ids(bundle)
                    lens_maps = collect_logit_lens(
                        prefill_hs, prompt.image_start, ids, final_norm, unembed, ref_ids
                    )

                img_embeds_ref = prompt.image_embeds
                # Free the heavy generation outputs before running the baseline pass
                del out
                out = None
                free_memory()

                baseline = _compute_baseline_attention(bundle, image, img_embeds_ref)
                grounding = GroundingMaps(
                    attention=att_maps,
                    baseline=baseline,
                    lens=lens_maps,
                    tokens=tokens,
                    meaningful=[is_meaningful_token(t) for t in tokens],
                )

        answer = bundle.tokenizer.decode(ids, skip_special_tokens=True).strip()
        return InferenceResult(answer, grounding, len(ids), time.perf_counter() - t0)
    finally:
        # Drop every GPU tensor (incl. ~0.5 GB of raw attention) before returning.
        del prompt, out
        free_memory()
