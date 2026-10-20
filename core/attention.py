"""Attention capture, 24x24 spatial maps and grayscale / colour focus masking.

Pipeline
--------
1. Capture (during ``generate``)
   * ``collect_image_attention`` : ``output_attentions=True`` tuple -> [T, L, 576]
     (head-averaged, image key columns only, row of the query that emitted token t)
   * ``collect_baseline_attention`` : same for a *generic* query ("Describe the image
     briefly.") -> [L, 576]. Used to cancel question-agnostic attention (sinks,
     positional / recency bias of the last image rows).
   * ``collect_logit_lens`` : ``output_hidden_states=True`` -> contrastive logit-lens
     score of every image patch for every answer token -> [T, L, 576]
2. ``select_focus_tokens``  : first *meaningful* answer tokens
3. ``aggregate_heatmap``    : tokens x layers -> 24x24 in [0, 1] (method dependent)
4. ``upsample_heatmap``     : bilinear interpolation to the original image size
5. ``render_focus_mask``    : alpha blend colour (focus) over grayscale (background)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from core.config import IMAGE_GRID_SIZE, NUM_IMAGE_TOKENS

# Words that carry no visual content (incl. the fixed answer templates used in
# training: "The answer is X." / "There are N in the image.").
_STOPWORDS = frozenset(
    """
    a an the is are was were be been being am it its it's this that these those there
    here i you he she we they me him her us them my your his our their of in on at to
    for with by from as and or but so if then than too very can could would should will
    shall may might must do does did has have had not no yes answer image picture photo
    photograph shows show showing see seen visible appears appear looks look which what
    who whom whose where when why how some any each also just only into onto over under
    """.split()
)
_PUNCT_RE = re.compile(r"^[\W_]+$", re.UNICODE)

# Reference vocabulary for the *contrastive* logit lens: a patch only scores high
# for "cat" if it prefers "cat" over generic visual words, not just because it
# has a large norm or is close to every visual concept.
REFERENCE_WORDS = (
    "person man woman people dog cat car table chair couch bed bottle cup plate food pizza bus "
    "train tree sky grass water road building window door wall floor red blue green white black "
    "brown yellow pink orange gray one two three four street sign phone laptop book clock bag ball"
).split()


class FocusMethod(str, Enum):
    HYBRID = "Hibrit (attention-oranı + logit lens)"
    ATTENTION_RATIO = "Attention / baseline oranı"
    ATTENTION = "Ham cross-attention"
    LOGIT_LENS = "Logit lens (hidden state)"


@dataclass
class GroundingMaps:
    """Everything needed to (re)draw a focus map without re-running the model."""

    attention: np.ndarray  # [T, L, 576] attention distribution over patches
    baseline: np.ndarray | None  # [L, 576]   generic-query attention distribution
    lens: np.ndarray | None  # [T, L, 576] contrastive logit-lens scores
    tokens: list[str]
    meaningful: list[bool]

    @property
    def num_layers(self) -> int:
        return int(self.attention.shape[1])


# --------------------------------------------------------------------------- #
# 1. Capture
# --------------------------------------------------------------------------- #
def _to_distribution(x: torch.Tensor) -> torch.Tensor:
    return x / x.sum(dim=-1, keepdim=True).clamp_min(1e-12)


@torch.inference_mode()
def collect_image_attention(
    step_attentions: Sequence[Sequence[torch.Tensor]],
    image_start: int,
    num_tokens: int,
    num_image_tokens: int = NUM_IMAGE_TOKENS,
) -> np.ndarray:
    """Reduce raw ``generate`` attentions to a compact [T, L, 576] CPU array.

    ``step_attentions[t][l]`` has shape ``[B, H, q, k]``. The *last query row*
    of step ``t`` is the position whose logits produced answer token ``t``.
    Only the image key columns are kept and heads are averaged on the GPU, so
    the big tensors can be freed immediately afterwards.
    """
    img = slice(image_start, image_start + num_image_tokens)
    per_step = [
        torch.stack([layer[0, :, -1, img] for layer in step]).float().mean(dim=1).cpu()
        for step in step_attentions[:num_tokens]
    ]
    # Distribution over patches: removes the large text-vs-image mass variation.
    return _to_distribution(torch.stack(per_step)).numpy()


@torch.inference_mode()
def collect_baseline_attention(
    layer_attentions: Sequence[torch.Tensor], image_start: int, num_image_tokens: int = NUM_IMAGE_TOKENS
) -> np.ndarray:
    """[L, 576] attention of the last prompt position for a generic question."""
    img = slice(image_start, image_start + num_image_tokens)
    maps = torch.stack([layer[0, :, -1, img] for layer in layer_attentions]).float().mean(dim=1)
    return _to_distribution(maps).cpu().numpy()


@torch.inference_mode()
def collect_logit_lens(
    hidden_states: Sequence[torch.Tensor],
    image_start: int,
    token_ids: Sequence[int],
    final_norm: torch.nn.Module,
    unembed: torch.Tensor,
    reference_ids: torch.Tensor,
    num_image_tokens: int = NUM_IMAGE_TOKENS,
) -> np.ndarray:
    """Contrastive logit lens on image patches -> [T, L, 576].

    For every layer the image hidden states are projected through the final
    norm + LM head and the logit of each answer token is compared with the mean
    logit of a reference vocabulary. ``hidden_states[0]`` (embeddings) is skipped.
    Only ``len(token_ids) + |ref|`` rows of the 151k vocab are touched, so this
    costs a few small matmuls.
    """
    img = slice(image_start, image_start + num_image_tokens)
    tgt_w = unembed[torch.as_tensor(list(token_ids), device=unembed.device)]  # [T, D]
    ref_w = unembed[reference_ids]  # [R, D]
    per_layer = []
    for h in hidden_states[1:]:
        hn = final_norm(h[0, img])  # [576, D] bf16
        score = (hn @ tgt_w.T).float() - (hn @ ref_w.T).float().mean(dim=-1, keepdim=True)
        per_layer.append(score.T.cpu())  # [T, 576]
    return torch.stack(per_layer, dim=1).numpy()  # [T, L, 576]


# --------------------------------------------------------------------------- #
# 2. Token selection
# --------------------------------------------------------------------------- #
def is_meaningful_token(token: str) -> bool:
    word = token.strip().lower()
    if not word or _PUNCT_RE.match(word):
        return False
    return word not in _STOPWORDS


def select_focus_tokens(maps: GroundingMaps, max_tokens: int) -> list[int]:
    """Indices of the first ``max_tokens`` meaningful tokens (fallback: all)."""
    idx = [i for i, m in enumerate(maps.meaningful) if m][: max(1, max_tokens)]
    return idx or list(range(len(maps.tokens)))


# --------------------------------------------------------------------------- #
# 3. Aggregation
# --------------------------------------------------------------------------- #
def _zscore(x: np.ndarray) -> np.ndarray:
    """Standardise over the patch axis (last) so layers / tokens are comparable."""
    return (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-8)


def _sink_mask(baseline: np.ndarray, z_thresh: float = 6.0, max_frac: float = 0.02) -> np.ndarray:
    """Token-agnostic "attention sink" patches via a robust z-score."""
    med = np.median(baseline)
    mad = np.median(np.abs(baseline - med)) * 1.4826 + 1e-12
    z = (baseline - med) / mad
    mask = z > z_thresh
    max_sinks = max(1, int(max_frac * baseline.size))
    if mask.sum() > max_sinks:  # keep only the strongest outliers
        mask = z >= np.sort(z.ravel())[-max_sinks]
    return mask


def _inpaint(grid: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace masked patches with the median of their unmasked 3x3 neighbours."""
    out = grid.copy()
    h, w = grid.shape
    for y, x in zip(*np.nonzero(mask)):
        y0, y1, x0, x1 = max(0, y - 1), min(h, y + 2), max(0, x - 1), min(w, x + 2)
        neigh = grid[y0:y1, x0:x1][~mask[y0:y1, x0:x1]]
        out[y, x] = np.median(neigh) if neigh.size else np.median(grid)
    return out


def _smooth3x3(grid: np.ndarray) -> np.ndarray:
    """Light box blur on the 24x24 grid: suppresses single-patch speckle noise."""
    t = torch.from_numpy(grid.astype(np.float32))[None, None]
    return F.avg_pool2d(F.pad(t, (1, 1, 1, 1), mode="replicate"), 3, stride=1)[0, 0].numpy()


def _method_scores(maps: GroundingMaps, method: FocusMethod, tok: list[int], lo: int, hi: int) -> np.ndarray:
    """[576] raw score vector for the chosen explanation method."""
    att = maps.attention[tok, lo : hi + 1]  # [t, l, 576]

    def att_ratio() -> np.ndarray:
        if maps.baseline is None:
            return _zscore(np.log(att + 1e-9)).mean(axis=(0, 1))
        base = maps.baseline[lo : hi + 1][None]  # [1, l, 576]
        return _zscore(np.log(att + 1e-9) - np.log(base + 1e-9)).mean(axis=(0, 1))

    def lens() -> np.ndarray:
        return _zscore(maps.lens[tok, lo : hi + 1]).mean(axis=(0, 1))

    if method is FocusMethod.ATTENTION:
        return att.mean(axis=(0, 1))
    if method is FocusMethod.ATTENTION_RATIO:
        return att_ratio()
    if method is FocusMethod.LOGIT_LENS and maps.lens is not None:
        return lens()
    if method is FocusMethod.HYBRID and maps.lens is not None:
        return _zscore(att_ratio()) + _zscore(lens())
    return att_ratio()  # graceful fallback when hidden states were not captured


def aggregate_heatmap(
    maps: GroundingMaps,
    token_indices: Sequence[int],
    layer_range: tuple[int, int],
    method: FocusMethod = FocusMethod.HYBRID,
    suppress_sinks: bool = True,
) -> np.ndarray:
    """Tokens x layers -> normalised 24x24 heatmap in [0, 1]."""
    lo, hi = sorted(int(v) for v in layer_range)
    lo, hi = max(0, lo), min(maps.num_layers - 1, hi)
    grid = _method_scores(maps, FocusMethod(method), list(token_indices), lo, hi)
    grid = grid.reshape(IMAGE_GRID_SIZE, IMAGE_GRID_SIZE)

    if suppress_sinks:
        ref = maps.baseline[lo : hi + 1] if maps.baseline is not None else maps.attention[:, lo : hi + 1]
        sinks = _sink_mask(ref.reshape(-1, NUM_IMAGE_TOKENS).mean(axis=0).reshape(grid.shape))
        if sinks.any():
            grid = _inpaint(grid, sinks)
        grid = _smooth3x3(grid)

    # Robust min-max: clip the top 0.5 % so a single patch cannot flatten the map.
    lo_v, hi_v = grid.min(), np.quantile(grid, 0.995)
    return np.clip((grid - lo_v) / max(hi_v - lo_v, 1e-12), 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# 4. Spatial upsampling
# --------------------------------------------------------------------------- #
def upsample_heatmap(grid: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Bilinear 24x24 -> (H, W). ``size`` is PIL style ``(width, height)``.

    SigLIP's processor resizes (no crop) to 384x384, so the patch grid maps
    1:1 onto the full original frame regardless of aspect ratio.
    """
    w, h = size
    t = torch.from_numpy(grid)[None, None]  # [1, 1, 24, 24] (tiny, CPU)
    up = F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
    return up[0, 0].clamp_(0.0, 1.0).numpy()


# --------------------------------------------------------------------------- #
# 5. Rendering
# --------------------------------------------------------------------------- #
def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def focus_alpha(heat: np.ndarray, threshold: float, softness: float) -> np.ndarray:
    """Soft mask: 0 below ``threshold``, ramps to 1 over ``softness``."""
    return _smoothstep((heat - threshold) / max(softness, 1e-3))


def render_focus_mask(
    image: Image.Image,
    grid: np.ndarray,
    threshold: float = 0.5,
    softness: float = 0.2,
    background_dim: float = 0.75,
) -> Image.Image:
    """Colour where the model looked, grayscale everywhere else (alpha blending)."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    gray = np.asarray(image.convert("L").convert("RGB"), dtype=np.float32) * background_dim
    alpha = focus_alpha(upsample_heatmap(grid, image.size), threshold, softness)[..., None]
    out = alpha * rgb + (1.0 - alpha) * gray
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


# Compact "inferno"-like colormap (no matplotlib dependency).
_CMAP_POS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
_CMAP_RGB = np.array(
    [[0, 0, 4], [87, 16, 110], [188, 55, 84], [249, 142, 9], [252, 255, 164]], dtype=np.float32
)


def _colorize(heat: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(heat, _CMAP_POS, _CMAP_RGB[:, c]) for c in range(3)], axis=-1)


def render_heatmap_overlay(image: Image.Image, grid: np.ndarray, opacity: float = 0.55) -> Image.Image:
    """Classic heatmap overlay (useful to see the raw, un-thresholded map)."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    heat = upsample_heatmap(grid, image.size)
    a = (opacity * (0.35 + 0.65 * heat))[..., None]
    out = (1.0 - a) * rgb + a * _colorize(heat)
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))
