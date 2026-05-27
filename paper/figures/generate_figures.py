#!/usr/bin/env python3
"""
Generate the three figures for the MedSafe-Dx v3 paper.

Outputs (next to this script):
  fig1_safety_vs_recall.pdf   — Safety Pass Rate vs Top-3 Recall scatter
  fig2_failure_modes.pdf      — Hard Safety Failure Modes bar chart
  fig3_tsr_tradeoff.pdf       — SPR vs over-escalation with iso-TSR contours
"""
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
LEADERBOARD = HERE.parent.parent / "leaderboard"
OUT_DIR = HERE

# Eight most narratively-important models for the scatter plots (fig1, fig3),
# chosen to keep the visual readable while preserving every headline finding:
#   leader (GPT-5 Chat), highest SPR (GPT-5.2), open-weight frontier
#   (Llama 4 Maverick), calibration extreme (Grok 4.20), reasoning anchor
#   (o3-pro), size-paradox endpoints (Haiku 4.5 vs Opus 4.7), outlier
#   (Gemini 3 Pro Preview). Sonnet 4.6, GPT-5 Mini, GPT-OSS 120B, and
#   DeepSeek R1 are omitted from scatters but appear on fig2 and in text.
SCATTER_MODELS = {
    "openai-gpt-5-chat",
    "openai-gpt-5.2",
    "meta-llama-llama-4-maverick",
    "x-ai-grok-4.20",
    "openai-o3-pro",
    "anthropic-claude-haiku-4.5",
    "anthropic-claude-opus-4.7",
    "google-gemini-3-pro-preview",
}

PROVIDER_COLOR = {
    "openai":    "#10a37f",
    "anthropic": "#d97757",
    "google":    "#4285f4",
    "deepseek":  "#6366f1",
    "meta":      "#0866ff",
    "xai":       "#212529",
}
PROVIDER_LABEL = {
    "openai": "OpenAI", "anthropic": "Anthropic", "google": "Google",
    "deepseek": "DeepSeek", "meta": "Meta", "xai": "xAI",
}

def provider_of(model_id: str) -> str:
    m = model_id.lower()
    if m.startswith("openai"): return "openai"
    if m.startswith("anthropic"): return "anthropic"
    if m.startswith("google"): return "google"
    if m.startswith("deepseek"): return "deepseek"
    if m.startswith("meta-llama"): return "meta"
    if m.startswith("x-ai"): return "xai"
    return "openai"

def short_name(model_id: str) -> str:
    s = model_id
    pairs = [
        ("anthropic-claude-haiku-",  "Haiku "),
        ("anthropic-claude-sonnet-", "Sonnet "),
        ("anthropic-claude-opus-",   "Opus "),
        ("anthropic-claude-",        "Claude "),
        ("openai-gpt-oss-",          "GPT-OSS "),
        ("openai-gpt-",              "GPT-"),
        ("openai-o",                 "o"),
        ("openai-",                  ""),
        ("google-gemini-",           "Gemini "),
        ("deepseek-deepseek-r1",     "DeepSeek R1"),
        ("deepseek-deepseek-",       "DeepSeek "),
        ("meta-llama-llama-4-maverick", "Llama 4 Maverick"),
        ("meta-llama-llama-",        "Llama "),
        ("x-ai-grok-",               "Grok "),
    ]
    for k, v in pairs:
        if s.startswith(k):
            s = v + s[len(k):]
            break
    # Strip suffixes
    suffix_map = {
        "-pro-preview": " Pro",
        "-pro": " Pro",
        "-flash-lite": " Flash Lite",
        "-flash": " Flash",
        "-chat": " chat",
        "-mini": " mini",
    }
    for k, v in suffix_map.items():
        if s.endswith(k):
            s = s[:-len(k)] + v
    return s

def load_models():
    """Return list of dicts with model, spr, missed, oc, unsafe, top3, over, total, nonurgent."""
    models = []
    for path in sorted(LEADERBOARD.glob("*-250cases-eval.json")):
        with open(path) as f:
            d = json.load(f)
        eff = d.get("effectiveness", {}) or {}
        safety = d.get("safety", {}) or {}
        denoms = d.get("denominators", {}) or {}
        nonurgent = denoms.get("nonurgent_cases", 94)
        total = d.get("cases_expected") or d.get("cases") or 250
        over = eff.get("over_escalation") or (d.get("informational") or {}).get("overdiagnosis") or 0
        spr = d.get("safety_pass_rate", 0.0)
        top3 = eff.get("top3_recall")
        models.append({
            "model": d["model"],
            "short": short_name(d["model"]),
            "provider": provider_of(d["model"]),
            "spr": spr,
            "missed": safety.get("missed_escalations", 0),
            "oc": safety.get("overconfident_wrong", 0),
            "unsafe": safety.get("unsafe_reassurance", 0),
            "top3": top3,
            "over": over,
            "total": total,
            "nonurgent": nonurgent,
            "tsr": spr - over / total,
        })
    return models

def wilson_ci(k, n, z=1.96):
    """Wilson score 95% CI for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))

# ─────────────────────────────────────────────────────────────
# Figure 1: Safety Pass Rate vs Top-3 Diagnostic Recall
# ─────────────────────────────────────────────────────────────

def fig1(models):
    # Same 8-model highlighted set as fig3 for visual consistency.
    highlighted = [m for m in models if m["model"] in SCATTER_MODELS]
    # Per-model label positions to avoid collisions in the SPR ~95% cluster.
    label_offsets = {
        "openai-gpt-5.2":              dict(xytext=(10, -8),  ha="left"),    # right-below
        "meta-llama-llama-4-maverick": dict(xytext=(10, 6),   ha="left"),    # right-above
        "anthropic-claude-haiku-4.5":  dict(xytext=(-10, 6),  ha="right"),   # left-above
        "openai-gpt-5-chat":           dict(xytext=(10, -10), ha="left"),    # right-below
        "openai-o3-pro":               dict(xytext=(-10, 6),  ha="right"),   # left-above
        "x-ai-grok-4.20":              dict(xytext=(0, -14),  ha="center"),  # below
        "anthropic-claude-opus-4.7":   dict(xytext=(-10, 6),  ha="right"),   # left-above
        "google-gemini-3-pro-preview": dict(xytext=(-10, 6),  ha="right"),   # left-above
    }
    fig, ax = plt.subplots(figsize=(11, 6.5))
    seen_providers = set()
    for m in highlighted:
        if m["top3"] is None:
            continue
        color = PROVIDER_COLOR[m["provider"]]
        label = PROVIDER_LABEL[m["provider"]] if m["provider"] not in seen_providers else None
        seen_providers.add(m["provider"])
        # 95% CI on SPR
        n_total = m["total"]
        n_safe = round(m["spr"] * n_total)
        lo, hi = wilson_ci(n_safe, n_total)
        spr_pct = m["spr"] * 100
        top3_pct = m["top3"] * 100
        ax.errorbar(top3_pct, spr_pct,
                    yerr=[[(spr_pct - lo * 100)], [(hi * 100 - spr_pct)]],
                    fmt='o', color=color, ecolor=color, elinewidth=1.2,
                    capsize=3, markersize=9, alpha=0.9, label=label)
        off = label_offsets.get(m["model"], dict(xytext=(8, 7), ha="left"))
        ax.annotate(m["short"], (top3_pct, spr_pct), textcoords="offset points",
                    fontsize=10, color="#212529", weight="medium",
                    bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.85),
                    **off)
    # Safety risk threshold
    ax.axhline(85, linestyle="--", color="#999", linewidth=0.8)
    ax.axhspan(50, 85, color="#fee2e2", alpha=0.4, zorder=0)
    ax.text(58, 85.3, "← higher safety risk", fontsize=8.5, color="#c0392b")
    ax.set_xlabel("Top-3 Diagnostic Recall (%)", fontsize=11)
    ax.set_ylabel("Safety Pass Rate (%)", fontsize=11)
    ax.set_title("Safety Pass Rate vs. Top-3 Diagnostic Recall ($N$=250, May 2026 refresh)\n"
                 "8 of 12 models shown; error bars: 95% Wilson CI", fontsize=11)
    ax.set_xlim(55, 95)
    ax.set_ylim(55, 100)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
    ax.legend(loc="lower left", fontsize=9.5, frameon=True)
    fig.tight_layout()
    out = OUT_DIR / "fig1_safety_vs_recall.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")

# ─────────────────────────────────────────────────────────────
# Figure 2: Hard Safety Failure Modes by Model
# ─────────────────────────────────────────────────────────────

def fig2(models):
    # Sort by TSR descending (matches table order)
    rows = sorted(models, key=lambda m: -m["tsr"])
    labels = [m["short"] for m in rows]
    missed = [m["missed"] for m in rows]
    oc = [m["oc"] for m in rows]
    unsafe = [m["unsafe"] for m in rows]
    x = np.arange(len(labels))
    width = 0.28
    fig, ax = plt.subplots(figsize=(11, 5.5))
    b1 = ax.bar(x - width, missed, width, label="Missed Escalation", color="#dc3545")
    b2 = ax.bar(x,         oc,     width, label="Overconfident Wrong", color="#f59e0b")
    b3 = ax.bar(x + width, unsafe, width, label="Unsafe Reassurance", color="#a78bfa")
    for bars in (b1, b2, b3):
        for b in bars:
            h = b.get_height()
            if h > 0:
                ax.text(b.get_x() + b.get_width() / 2, h + 0.4, str(int(h)),
                        ha="center", fontsize=8, color="#212529")
    ax.set_ylabel("Number of Cases (out of 250)", fontsize=11)
    ax.set_title("Hard Safety Failure Modes by Model ($N$=250, May 2026 refresh)\nSorted by Triage Success Rate (descending)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.legend(fontsize=9, frameon=True)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    out = OUT_DIR / "fig2_failure_modes.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")

# ─────────────────────────────────────────────────────────────
# Figure 3: SPR vs over-escalation with iso-TSR contours
# ─────────────────────────────────────────────────────────────

def fig3(models):
    highlighted = [m for m in models if m["model"] in SCATTER_MODELS]
    fig, ax = plt.subplots(figsize=(12, 6))
    Y_MIN, Y_MAX = 50, 100
    # Iso-TSR contours: y = TSR + k*x, where x is over-esc rate (%) of non-urgent.
    # Drop TSR=25 since its line falls below the new y range (50–100).
    ks = [m["nonurgent"] / m["total"] for m in models]
    k = sum(ks) / len(ks) if ks else 0.376
    for tsr in [50, 75, 90]:
        # Clip x so both x in [0, 100] AND y = tsr + k*x in [Y_MIN, Y_MAX]
        x_lo = max(0, (Y_MIN - tsr) / k)
        x_hi = min(100, (Y_MAX - tsr) / k)
        if x_hi <= x_lo:
            continue
        xs = np.linspace(x_lo, x_hi, 50)
        ys = tsr + k * xs
        ax.plot(xs, ys, "--", color="#adb5bd", linewidth=0.8, alpha=0.7)
        # Label at the line's right endpoint (after x-axis inversion this is the
        # low-x / high-quality side, so the label sits near the ideal corner)
        label_x = x_lo if tsr + k * x_lo <= Y_MAX else x_hi
        label_y = tsr + k * label_x
        ax.text(label_x - 2, label_y + 0.5, f"TSR {tsr}%", fontsize=9, color="#868e96", style="italic",
                horizontalalignment="right")
    # ACS-COT tolerance zone: SPR >= 90 AND over_esc_rate (of non-urgent) <= 25
    ax.add_patch(plt.Rectangle((0, 90), 25, 10, facecolor="#198754", alpha=0.10, zorder=0))
    ax.text(2, 95, "ACS-COT tolerance\n(SPR≥90, over-esc≤25%)", fontsize=8.5,
            color="#198754", style="italic", verticalalignment="center")

    # Plot points — only the 8 most-narratively-interesting models.
    # Per-model label positions mirror the live web chart (avoid label overlap
    # in the top-left cluster where GPT-5.2 / Llama 4 / Haiku 4.5 sit close).
    label_offsets = {
        "openai-gpt-5.2":              dict(xytext=(-10, -14), ha="right"),   # below-left
        "meta-llama-llama-4-maverick": dict(xytext=(0,   10),  ha="center"),  # above
        "anthropic-claude-haiku-4.5":  dict(xytext=(10,  4),   ha="left"),    # right
        "openai-gpt-5-chat":           dict(xytext=(10,  4),   ha="left"),    # right
        "openai-o3-pro":               dict(xytext=(-10, -12), ha="right"),   # below-left
        "x-ai-grok-4.20":              dict(xytext=(0,   10),  ha="center"),  # above
        "anthropic-claude-opus-4.7":   dict(xytext=(-10, 8),   ha="right"),   # above-left
        "google-gemini-3-pro-preview": dict(xytext=(0,   10),  ha="center"),  # above
    }
    seen_providers = set()
    for m in highlighted:
        x = m["over"] / m["nonurgent"] * 100
        y = m["spr"] * 100
        color = PROVIDER_COLOR[m["provider"]]
        label = PROVIDER_LABEL[m["provider"]] if m["provider"] not in seen_providers else None
        seen_providers.add(m["provider"])
        ax.scatter(x, y, color=color, s=110, alpha=0.9, edgecolor=color, linewidth=1.4,
                   label=label, zorder=3)
        off = label_offsets.get(m["model"], dict(xytext=(8, 7), ha="left"))
        ax.annotate(m["short"], (x, y), textcoords="offset points",
                    fontsize=10, color="#212529", weight="medium", zorder=4,
                    bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.85),
                    **off)

    ax.set_xlim(0, 100)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlabel("Over-escalation rate (% of non-urgent cases) — lower is better", fontsize=11)
    ax.set_ylabel("Safety Pass Rate (%) — higher is better", fontsize=11)
    ax.set_title("Triage tradeoff: Safety Pass Rate vs over-escalation, with iso-TSR contours\n"
                 "($N$=250; 8 of 12 models shown; ideal corner = 100% SPR with 0% over-escalation)",
                 fontsize=11)
    ax.invert_xaxis()  # so lower over-esc (better) is on the right → top-right is ideal
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
    ax.legend(loc="lower left", fontsize=9.5, frameon=True)
    fig.tight_layout()
    out = OUT_DIR / "fig3_tsr_tradeoff.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")

# ─────────────────────────────────────────────────────────────

def main():
    models = load_models()
    print(f"Loaded {len(models)} models")
    fig1(models)
    fig2(models)
    fig3(models)

if __name__ == "__main__":
    main()
