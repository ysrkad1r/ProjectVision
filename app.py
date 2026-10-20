"""ProjectVision XAI — Minimalist AI Studio UI (components, layout and CSS only).

Run:  python app.py
"""
from __future__ import annotations

import html
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # Anaconda + torch duplicate OpenMP on Windows
if os.name != "nt":  # expandable segments reduce fragmentation; unsupported on Windows
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import logging  # noqa: E402

import gradio as gr  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from core.attention import (  # noqa: E402
    FocusMethod,
    GroundingMaps,
    aggregate_heatmap,
    render_focus_mask,
    render_heatmap_overlay,
    select_focus_tokens,
)
from core.inference import GenerationSettings, answer_question  # noqa: E402
from core.model import free_memory, get_vlm, vram_report  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

AUTO_TOKEN = "Auto (meaningful tokens)"
QUICK_PROMPTS: list[tuple[str, str]] = [
    ("Describe", "Describe the image briefly."),
    ("Count objects", "How many objects are there in the image?"),
    ("Identify colors", "What are the main colors in the image?"),
]
NUM_LAYERS_DEFAULT = 36  # Qwen2.5-3B; real value read from the model at startup

HEAD_FONT_LINKS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
"""

CSS = """
:root, .dark {
  --bg-canvas: #09090b;
  --bg-card: #18181b;
  --bg-elevated: #121215;
  --border-subtle: rgba(255, 255, 255, 0.08);
  --border-hover: rgba(255, 255, 255, 0.16);
  --text-primary: #f4f4f5;
  --text-secondary: #a1a1aa;
  --text-muted: #71717a;
  --radius-xl: 12px;

  --body-background-fill: #09090b !important;
  --background-fill-primary: #18181b !important;
  --background-fill-secondary: #121215 !important;
  --block-background-fill: #18181b !important;
  --block-border-color: rgba(255, 255, 255, 0.08) !important;
  --border-color-primary: rgba(255, 255, 255, 0.08) !important;
  --border-color-accent: rgba(255, 255, 255, 0.18) !important;
  --body-text-color: #f4f4f5 !important;
  --body-text-color-subdued: #a1a1aa !important;
  --block-label-text-color: #a1a1aa !important;
  --block-title-text-color: #a1a1aa !important;
  --input-background-fill: #121215 !important;
  --input-border-color: rgba(255, 255, 255, 0.08) !important;
  --input-border-color-focus: rgba(255, 255, 255, 0.22) !important;
  --button-primary-background-fill: #f4f4f5 !important;
  --button-primary-background-fill-hover: #ffffff !important;
  --button-primary-text-color: #09090b !important;
  --button-secondary-background-fill: #18181b !important;
  --button-secondary-background-fill-hover: #27272a !important;
  --button-secondary-text-color: #d4d4d8 !important;
  --button-secondary-border-color: rgba(255, 255, 255, 0.08) !important;
  --slider-color: #e4e4e7 !important;
}

html, body, .gradio-container {
  background: var(--bg-canvas) !important;
  color: var(--text-primary) !important;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
  -webkit-font-smoothing: antialiased;
}

.gradio-container {
  max-width: 1240px !important;
  margin: 0 auto !important;
  padding: 20px 28px 40px !important;
}

/* Header */
.pv-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  padding: 8px 2px 20px;
  margin-bottom: 8px;
  border-bottom: 1px solid var(--border-subtle);
}
.pv-brand-row {
  display: flex;
  align-items: center;
  gap: 10px;
}
.pv-logo-mark {
  width: 22px;
  height: 22px;
  border-radius: 6px;
  background: linear-gradient(135deg, #f4f4f5 0%, #71717a 100%);
  display: inline-flex;
  align-items: center;
  justify-content: center;
  box-shadow: inset 0 0 0 1px rgba(255,255,255,0.2);
}
.pv-logo-mark::after {
  content: "";
  width: 8px;
  height: 8px;
  border-radius: 2px;
  background: #09090b;
}
.pv-brand-name {
  font-size: 1.05rem;
  font-weight: 600;
  letter-spacing: -0.02em;
  color: var(--text-primary);
}
.pv-version {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem;
  font-weight: 500;
  color: var(--text-muted);
  padding: 2px 7px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid var(--border-subtle);
}
.pv-subtitle {
  margin: 5px 0 0 0;
  font-size: 0.85rem;
  color: var(--text-secondary);
  font-weight: 400;
}

/* Top-right Model Details popover */
.pv-details {
  position: relative;
}
.pv-details > summary {
  list-style: none;
  cursor: pointer;
  user-select: none;
  font-size: 0.78rem;
  font-weight: 500;
  color: var(--text-secondary);
  padding: 6px 12px;
  border-radius: 8px;
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  transition: color 0.15s ease, border-color 0.15s ease, background 0.15s ease;
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.pv-details > summary::-webkit-details-marker { display: none; }
.pv-details > summary::after {
  content: "▾";
  font-size: 0.7rem;
  color: var(--text-muted);
}
.pv-details > summary:hover {
  color: var(--text-primary);
  border-color: var(--border-hover);
}
.pv-details-menu {
  position: absolute;
  right: 0;
  top: calc(100% + 8px);
  width: 290px;
  padding: 12px 14px;
  border-radius: var(--radius-xl);
  background: #141417;
  border: 1px solid rgba(255, 255, 255, 0.1);
  box-shadow: 0 18px 40px rgba(0, 0, 0, 0.55);
  z-index: 50;
}
.pv-spec-row {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  padding: 5px 0;
  font-size: 0.76rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}
.pv-spec-row:last-child { border-bottom: none; }
.pv-spec-k { color: var(--text-muted); }
.pv-spec-v {
  color: var(--text-primary);
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem;
}

/* Blocks & cards */
.block, .gr-box, .gr-panel, .accordion {
  border-radius: var(--radius-xl) !important;
  border: 1px solid var(--border-subtle) !important;
  background: var(--bg-card) !important;
  box-shadow: none !important;
}

/* Upload dropzone */
#pv-upload {
  border-radius: var(--radius-xl) !important;
  overflow: hidden;
}
#pv-upload [data-testid="image"] {
  background: var(--bg-card) !important;
}

/* Prompt bar */
#pv-prompt-row {
  align-items: stretch !important;
  gap: 8px !important;
}
#pv-prompt-input {
  border-radius: var(--radius-xl) !important;
  padding: 0 !important;
  border: none !important;
  background: transparent !important;
}
#pv-prompt-input textarea {
  background: var(--bg-card) !important;
  border: 1px solid var(--border-subtle) !important;
  border-radius: var(--radius-xl) !important;
  color: var(--text-primary) !important;
  font-size: 0.92rem !important;
  padding: 11px 14px !important;
  min-height: 44px !important;
  transition: border-color 0.15s ease !important;
}
#pv-prompt-input textarea:focus {
  border-color: rgba(255, 255, 255, 0.24) !important;
  box-shadow: none !important;
}
#pv-analyze-btn {
  min-width: 108px !important;
  max-width: 124px !important;
  height: 44px !important;
  border-radius: var(--radius-xl) !important;
  font-weight: 500 !important;
  font-size: 0.88rem !important;
  letter-spacing: -0.01em !important;
  background: #f4f4f5 !important;
  color: #09090b !important;
  border: 1px solid transparent !important;
  transition: background 0.15s ease, transform 0.1s ease !important;
}
#pv-analyze-btn:hover {
  background: #ffffff !important;
}

/* Quick pill buttons */
#pv-pills-row {
  gap: 8px !important;
  margin-top: -2px !important;
}
.pv-pill button, button.pv-pill {
  border-radius: 999px !important;
  background: #141417 !important;
  border: 1px solid var(--border-subtle) !important;
  color: var(--text-secondary) !important;
  font-size: 0.78rem !important;
  font-weight: 500 !important;
  padding: 5px 12px !important;
  min-height: 30px !important;
  transition: color 0.15s ease, border-color 0.15s ease, background 0.15s ease !important;
}
.pv-pill button:hover, button.pv-pill:hover {
  color: var(--text-primary) !important;
  border-color: var(--border-hover) !important;
  background: #1f1f23 !important;
}

/* Accordions */
.pv-accordion {
  border-radius: var(--radius-xl) !important;
  background: var(--bg-card) !important;
  border: 1px solid var(--border-subtle) !important;
}
.pv-accordion .label-wrap {
  font-size: 0.83rem !important;
  font-weight: 500 !important;
  color: var(--text-secondary) !important;
}

/* Right column: Tabs & visual canvas */
.pv-tabs {
  border-radius: var(--radius-xl) !important;
  background: var(--bg-card) !important;
  border: 1px solid var(--border-subtle) !important;
  overflow: hidden;
}
.pv-tabs .tab-nav {
  border-bottom: 1px solid var(--border-subtle) !important;
  background: #141417 !important;
  padding: 4px 8px 0 !important;
}
.pv-tabs .tab-nav button {
  font-size: 0.8rem !important;
  font-weight: 500 !important;
  color: var(--text-secondary) !important;
  border: none !important;
  padding: 8px 12px !important;
}
.pv-tabs .tab-nav button.selected {
  color: var(--text-primary) !important;
  border-bottom: 1.5px solid #f4f4f5 !important;
}

/* Response card & inline token distribution */
.pv-output-card {
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-xl);
  padding: 18px 20px;
}
.pv-section-label {
  font-size: 0.7rem;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-muted);
  margin-bottom: 8px;
}
.pv-answer-text {
  font-size: 1.1rem;
  line-height: 1.6;
  font-weight: 400;
  color: #f4f4f5;
  letter-spacing: -0.01em;
}
.pv-answer-placeholder {
  font-size: 0.92rem;
  color: var(--text-muted);
}
.pv-token-section {
  margin-top: 16px;
  padding-top: 14px;
  border-top: 1px solid var(--border-subtle);
}
.pv-token-stream {
  font-size: 0.92rem;
  line-height: 1.9;
  color: var(--text-secondary);
}
.pv-tok {
  display: inline;
  padding: 2px 3px;
  margin: 0 1px;
  border-radius: 4px;
  border-bottom: 1px solid transparent;
  transition: color 0.15s ease, background 0.15s ease, border-color 0.15s ease;
  cursor: default;
}
.pv-tok.is-focus {
  color: #e4e4e7;
  border-bottom-color: rgba(244, 244, 245, 0.36);
}
.pv-tok:hover {
  color: #ffffff;
  background: rgba(255, 255, 255, 0.07);
  border-bottom-color: rgba(255, 255, 255, 0.75);
}
.pv-meta-bar {
  margin-top: 12px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem;
  color: var(--text-muted);
}

/* Minimal loading indicator: hide verbose Gradio timer text, keep clean spinner */
.eta-bar, .progress-text, .meta-text, .meta-text-center {
  display: none !important;
}
.wrap.default.generating, .wrap.translucent {
  background: rgba(9, 9, 11, 0.45) !important;
  backdrop-filter: blur(2px);
}

footer { display: none !important; }
"""

HEADER_HTML = """
<header class="pv-header">
  <div>
    <div class="pv-brand-row">
      <span class="pv-logo-mark"></span>
      <span class="pv-brand-name">ProjectVision</span>
      <span class="pv-version">v2.1</span>
    </div>
    <p class="pv-subtitle">Görsel soru-cevap ve dikkat odak analizi.</p>
  </div>
  <details class="pv-details">
    <summary>Model Details</summary>
    <div class="pv-details-menu">
      <div class="pv-spec-row"><span class="pv-spec-k">Vision Encoder</span><span class="pv-spec-v">SigLIP-B/16 (384px)</span></div>
      <div class="pv-spec-row"><span class="pv-spec-k">Patch Grid</span><span class="pv-spec-v">24×24 · 576 tokens</span></div>
      <div class="pv-spec-row"><span class="pv-spec-k">Projector</span><span class="pv-spec-v">2-Layer MLP (768→2048)</span></div>
      <div class="pv-spec-row"><span class="pv-spec-k">Language Model</span><span class="pv-spec-v">Qwen2.5-3B-Instruct</span></div>
      <div class="pv-spec-row"><span class="pv-spec-k">Adaptation</span><span class="pv-spec-v">QLoRA (r=16) · NF4</span></div>
      <div class="pv-spec-row"><span class="pv-spec-k">Grounding</span><span class="pv-spec-v">Cross-Attn + Logit Lens</span></div>
    </div>
  </details>
</header>
"""


# --------------------------------------------------------------------------- #
# Rendering & formatting helpers
# --------------------------------------------------------------------------- #
def _token_labels(maps: GroundingMaps) -> list[str]:
    return [f"#{i} · {t.strip() or '␣'}" for i, t in enumerate(maps.tokens)]


def _format_response_card(
    answer: str | None,
    maps: GroundingMaps | None = None,
    token_choice: str = AUTO_TOKEN,
    k: int = 3,
    stats: str = "",
) -> str:
    """Render the response card and subtle hover/underlined inline token stream."""
    if not answer:
        return (
            '<div class="pv-output-card">'
            '<div class="pv-section-label">Response</div>'
            '<div class="pv-answer-placeholder">Yanıt ve dikkat dağılımını görmek için bir görsel yükleyip soru gönderin.</div>'
            "</div>"
        )

    safe_answer = html.escape(answer)
    tokens_html = ""
    if maps and maps.tokens:
        labels = _token_labels(maps)
        if token_choice and token_choice != AUTO_TOKEN and token_choice in labels:
            used = {labels.index(token_choice)}
        else:
            used = set(select_focus_tokens(maps, int(k)))

        spans: list[str] = []
        for idx, tok in enumerate(maps.tokens):
            cls = "pv-tok is-focus" if idx in used else "pv-tok"
            role = "odak token" if idx in used else "token"
            title = html.escape(f"#{idx} · { role }")
            spans.append(f'<span class="{cls}" title="{title}">{html.escape(tok)}</span>')
        tokens_html = (
            '<div class="pv-token-section">'
            '<div class="pv-section-label">Token Attention Distribution</div>'
            f'<div class="pv-token-stream">{"".join(spans)}</div>'
            "</div>"
        )

    meta_html = f'<div class="pv-meta-bar">{html.escape(stats)}</div>' if stats else ""
    return (
        '<div class="pv-output-card">'
        '<div class="pv-section-label">Response</div>'
        f'<div class="pv-answer-text">{safe_answer}</div>'
        f"{tokens_html}"
        f"{meta_html}"
        "</div>"
    )


def _render(
    image: Image.Image | None,
    maps: GroundingMaps | None,
    token_choice: str,
    k: int,
    method: str,
    layer_lo: int,
    layer_hi: int,
    thr: float,
    soft: float,
    sinks: bool,
):
    """Build mask + heatmap from cached attention (no model call)."""
    if image is None or maps is None:
        return None, None
    labels = _token_labels(maps)
    if token_choice and token_choice != AUTO_TOKEN and token_choice in labels:
        indices = [labels.index(token_choice)]
    else:
        indices = select_focus_tokens(maps, int(k))
    grid = aggregate_heatmap(
        maps,
        indices,
        (int(layer_lo), int(layer_hi)),
        method=FocusMethod(method),
        suppress_sinks=bool(sinks),
    )
    return render_focus_mask(image, grid, float(thr), float(soft)), render_heatmap_overlay(image, grid)


def _highlight(maps: GroundingMaps | None, token_choice: str, k: int):
    if maps is None:
        return []
    labels = _token_labels(maps)
    if token_choice and token_choice != AUTO_TOKEN and token_choice in labels:
        used = {labels.index(token_choice)}
    else:
        used = set(select_focus_tokens(maps, int(k)))
    return [(t, "odak" if i in used else None) for i, t in enumerate(maps.tokens)]


# --------------------------------------------------------------------------- #
# Event handlers
# --------------------------------------------------------------------------- #
def run_vqa(
    image,
    question,
    temperature,
    max_tokens,
    rep_penalty,
    use_top,
    top_p,
    top_k,
    capture,
    method,
    k,
    layer_lo,
    layer_hi,
    thr,
    soft,
    sinks,
):
    if image is None:
        gr.Warning("Lütfen önce bir görsel yükleyin.")
        return (gr.skip(),) * 9
    if not question or not question.strip():
        gr.Warning("Lütfen bir soru girin veya örnek sorulardan birini seçin.")
        return (gr.skip(),) * 9

    settings = GenerationSettings(
        temperature=float(temperature),
        max_new_tokens=int(max_tokens),
        repetition_penalty=float(rep_penalty),
        use_top_filtering=bool(use_top),
        top_p=float(top_p),
        top_k=int(top_k),
        capture_attention=bool(capture),
    )
    try:
        result = answer_question(get_vlm(), image, question, settings)
    except torch.cuda.OutOfMemoryError:
        free_memory()
        raise gr.Error(
            "GPU belleği yetersiz (OOM). Max Tokens değerini düşürün."
        )
    except Exception as exc:
        free_memory()
        logging.exception("Inference failed")
        raise gr.Error(f"Çıkarım hatası: {exc}")

    maps = result.attention
    mask, heat = _render(image, maps, AUTO_TOKEN, k, method, layer_lo, layer_hi, thr, soft, sinks)
    choices = [AUTO_TOKEN] + (_token_labels(maps) if maps else [])
    mode = "sampling" if settings.do_sample else "greedy"
    stats = f"{result.elapsed_s:.2f}s · {result.num_tokens} tokens · {mode} · {vram_report()}"
    answer_text = result.answer or "(no response)"
    card_html = _format_response_card(answer_text, maps, AUTO_TOKEN, k, stats)

    return (
        card_html,
        answer_text,
        mask,
        image,
        heat,
        _highlight(maps, AUTO_TOKEN, k),
        maps,
        gr.update(choices=choices, value=AUTO_TOKEN),
        stats,
    )


def rerender(
    image,
    maps,
    answer_text,
    stats,
    token_choice,
    method,
    k,
    layer_lo,
    layer_hi,
    thr,
    soft,
    sinks,
):
    mask, heat = _render(image, maps, token_choice, k, method, layer_lo, layer_hi, thr, soft, sinks)
    card_html = _format_response_card(answer_text, maps, token_choice, k, stats or "")
    return mask, heat, card_html, _highlight(maps, token_choice, k)


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
def build_ui(num_layers: int = NUM_LAYERS_DEFAULT) -> gr.Blocks:
    method_choices = [m.value for m in FocusMethod]
    with gr.Blocks(title="ProjectVision v2.1") as demo:
        attn_state = gr.State(None)
        answer_state = gr.State("")
        stats_state = gr.State("")

        gr.HTML(HEADER_HTML)

        with gr.Row(equal_height=False):
            # -------------------- LEFT: Input & Controls ------------------- #
            with gr.Column(scale=1):
                image_in = gr.Image(
                    type="pil",
                    show_label=False,
                    height=360,
                    sources=["upload", "clipboard", "webcam"],
                    elem_id="pv-upload",
                )

                with gr.Row(elem_id="pv-prompt-row"):
                    question_in = gr.Textbox(
                        show_label=False,
                        placeholder="Görsel hakkında bir soru sorun... (Örn: What is on the table?)",
                        lines=1,
                        max_lines=3,
                        scale=5,
                        container=False,
                        elem_id="pv-prompt-input",
                    )
                    run_btn = gr.Button(
                        "Analyze",
                        variant="primary",
                        scale=1,
                        elem_id="pv-analyze-btn",
                    )

                with gr.Row(elem_id="pv-pills-row"):
                    quick_btns = [
                        gr.Button(label, size="sm", variant="secondary", elem_classes="pv-pill")
                        for label, _ in QUICK_PROMPTS
                    ]

                with gr.Accordion("Advanced Settings", open=False, elem_classes="pv-accordion"):
                    temperature = gr.Slider(
                        0.0,
                        1.0,
                        value=0.0,
                        step=0.05,
                        label="Temperature",
                    )
                    max_tokens = gr.Slider(
                        10,
                        150,
                        value=35,
                        step=1,
                        label="Max Tokens",
                    )
                    # Preserved parametric controls kept hidden for full pipeline compatibility
                    rep_penalty = gr.Slider(1.0, 1.5, value=1.2, step=0.01, visible=False)
                    use_top = gr.Checkbox(False, visible=False)
                    top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.01, visible=False)
                    top_k = gr.Slider(1, 100, value=50, step=1, visible=False)
                    capture = gr.Checkbox(True, visible=False)
                    method = gr.Dropdown(
                        method_choices, value=FocusMethod.HYBRID.value, visible=False
                    )
                    focus_k = gr.Slider(1, 10, value=3, step=1, visible=False)
                    layer_lo = gr.Slider(0, num_layers - 1, value=14, step=1, visible=False)
                    layer_hi = gr.Slider(
                        0, num_layers - 1, value=min(28, num_layers - 1), step=1, visible=False
                    )
                    threshold = gr.Slider(0.0, 0.95, value=0.50, step=0.01, visible=False)
                    softness = gr.Slider(0.02, 0.6, value=0.20, step=0.01, visible=False)
                    sinks = gr.Checkbox(True, visible=False)

            # ----------------- RIGHT: Intelligence & Output ---------------- #
            with gr.Column(scale=1):
                with gr.Tabs(elem_classes="pv-tabs"):
                    with gr.Tab("Focus Mask"):
                        mask_out = gr.Image(
                            show_label=False,
                            interactive=False,
                            height=360,
                            buttons=["download", "fullscreen"],
                        )
                    with gr.Tab("Original"):
                        orig_out = gr.Image(
                            show_label=False,
                            interactive=False,
                            height=360,
                            buttons=["fullscreen"],
                        )
                    with gr.Tab("Heatmap"):
                        heat_out = gr.Image(
                            show_label=False,
                            interactive=False,
                            height=360,
                            buttons=["download", "fullscreen"],
                        )

                response_card = gr.HTML(_format_response_card(None))

                with gr.Accordion("Focus Inspector", open=False, elem_classes="pv-accordion"):
                    token_select = gr.Dropdown(
                        [AUTO_TOKEN],
                        value=AUTO_TOKEN,
                        label="Target Token",
                    )
                    tokens_out = gr.HighlightedText(visible=False)

        # ------------------------------ EVENTS -------------------------------- #
        for btn, (_, prompt_text) in zip(quick_btns, QUICK_PROMPTS):
            btn.click(lambda q=prompt_text: q, outputs=question_in, show_progress="hidden")

        gen_inputs = [
            image_in,
            question_in,
            temperature,
            max_tokens,
            rep_penalty,
            use_top,
            top_p,
            top_k,
            capture,
            method,
            focus_k,
            layer_lo,
            layer_hi,
            threshold,
            softness,
            sinks,
        ]
        gen_outputs = [
            response_card,
            answer_state,
            mask_out,
            orig_out,
            heat_out,
            tokens_out,
            attn_state,
            token_select,
            stats_state,
        ]
        run_btn.click(run_vqa, gen_inputs, gen_outputs, show_progress="minimal")
        question_in.submit(run_vqa, gen_inputs, gen_outputs, show_progress="minimal")

        view_inputs = [
            image_in,
            attn_state,
            answer_state,
            stats_state,
            token_select,
            method,
            focus_k,
            layer_lo,
            layer_hi,
            threshold,
            softness,
            sinks,
        ]
        view_outputs = [mask_out, heat_out, response_card, tokens_out]
        for comp in (token_select, method, focus_k, layer_lo, layer_hi, threshold, softness, sinks):
            evt = comp.release if isinstance(comp, gr.Slider) else comp.change
            evt(rerender, view_inputs, view_outputs, show_progress="hidden")

        # When a new image is uploaded or cleared, sync the Original tab and reset stale outputs.
        image_in.change(
            lambda img: (
                None,
                "",
                "",
                None,
                img,
                None,
                _format_response_card(None),
                [],
                gr.update(choices=[AUTO_TOKEN], value=AUTO_TOKEN),
            ),
            inputs=image_in,
            outputs=[
                attn_state,
                answer_state,
                stats_state,
                mask_out,
                orig_out,
                heat_out,
                response_card,
                tokens_out,
                token_select,
            ],
            show_progress="hidden",
        )
    return demo


def main() -> None:
    bundle = get_vlm()  # load once at startup so the first request is fast
    demo = build_ui(bundle.num_llm_layers)
    demo.queue(default_concurrency_limit=1)  # single GPU worker -> no parallel OOM
    demo.launch(
        theme=gr.themes.Base(
            primary_hue="zinc",
            secondary_hue="zinc",
            neutral_hue="zinc",
            radius_size=gr.themes.sizes.radius_lg,
            font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
            font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
        ),
        css=CSS,
        head=HEAD_FONT_LINKS,
        js="() => { document.body.classList.add('dark'); }",
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
    )


if __name__ == "__main__":
    main()
