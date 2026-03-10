#!/usr/bin/env python3
"""
Generate Figure 1 and Figure 2 for the MedSafe-Dx paper.

Figure 1: Safety Pass Rate vs Top-3 Diagnostic Recall (scatter)
Figure 2: Hard safety failure mode breakdown by model (stacked bar)

Reads data directly from results/artifacts/*-500cases-eval.json files.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import glob
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

ARTIFACTS_DIR = Path(__file__).parent.parent / "results" / "artifacts"

# ── Load data from 500-case eval artifacts ──
artifact_files = sorted(glob.glob(str(ARTIFACTS_DIR / "*-500cases-eval.json")))

if not artifact_files:
    raise FileNotFoundError(f"No *-500cases-eval.json files found in {ARTIFACTS_DIR}")

# Parse all artifacts
raw_data = []
for fpath in artifact_files:
    with open(fpath) as f:
        data = json.load(f)
    model_id = data["model"]
    safety = data["safety"]
    eff = data["effectiveness"]
    raw_data.append({
        "model_id": model_id,
        "spr": data["safety_pass_rate"],
        "top3_recall": eff["top3_recall"],
        "missed": safety["missed_escalations"],
        "overconf": safety["overconfident_wrong"],
        "unsafe_r": safety["unsafe_reassurance"],
        "coverage": data["coverage_rate"],
        "over_esc_rate": eff["over_escalation_rate"],
    })

# Sort by SPR descending (matches Table 1 ranking)
raw_data.sort(key=lambda x: x["spr"], reverse=True)

# Display name mapping
DISPLAY_NAMES = {
    "openai-gpt-4o": "GPT-4o",
    "anthropic-claude-haiku-4.5": "Claude Haiku 4.5",
    "anthropic-claude-sonnet-4.6": "Claude Sonnet 4.6",
    "anthropic-claude-opus-4.6": "Claude Opus 4.6",
    "openai-gpt-5-mini": "GPT-5 Mini",
    "google-gemini-3.1-pro-preview": "Gemini 3.1\nPro Preview",
    "openai-gpt-5.4-pro": "GPT-5.4 Pro",
}

# Scatter labels (no newlines)
SCATTER_NAMES = {
    "openai-gpt-4o": "GPT-4o",
    "anthropic-claude-haiku-4.5": "Claude Haiku 4.5",
    "anthropic-claude-sonnet-4.6": "Claude Sonnet 4.6",
    "anthropic-claude-opus-4.6": "Claude Opus 4.6",
    "openai-gpt-5-mini": "GPT-5 Mini",
    "google-gemini-3.1-pro-preview": "Gemini 3.1 Pro Preview",
    "openai-gpt-5.4-pro": "GPT-5.4 Pro",
}

# Provider detection
def get_provider(model_id):
    if model_id.startswith("openai-"):
        return "OpenAI"
    elif model_id.startswith("anthropic-"):
        return "Anthropic"
    elif model_id.startswith("google-"):
        return "Google"
    return "Other"

# Provider colors
PROVIDER_COLORS = {
    "OpenAI":    "#10a37f",
    "Anthropic": "#d97706",
    "Google":    "#4285f4",
}

# Build arrays
models = [DISPLAY_NAMES.get(d["model_id"], d["model_id"]) for d in raw_data]
scatter_labels = [SCATTER_NAMES.get(d["model_id"], d["model_id"]) for d in raw_data]
spr = [d["spr"] * 100 for d in raw_data]
top3 = [d["top3_recall"] * 100 for d in raw_data]
missed = [d["missed"] for d in raw_data]
overconf = [d["overconf"] for d in raw_data]
unsafe_r = [d["unsafe_r"] for d in raw_data]
providers = [get_provider(d["model_id"]) for d in raw_data]
colors = [PROVIDER_COLORS.get(p, "#888888") for p in providers]

N = raw_data[0]["spr"]  # just to confirm we read data
print(f"Loaded {len(raw_data)} models from 500-case evaluation artifacts")
for d in raw_data:
    print(f"  {d['model_id']}: SPR={d['spr']:.3f}, Top-3={d['top3_recall']:.3f}, "
          f"Missed={d['missed']}, Overconf={d['overconf']}, Unsafe={d['unsafe_r']}")


# ══════════════════════════════════════════════════════════
# Figure 1: Safety Pass Rate vs Top-3 Recall (N=500)
# ══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5.5))

ax.scatter(top3, spr, c=colors, s=100, zorder=5, edgecolors="white", linewidths=0.8)

# Label each point with manual offset adjustments
label_offsets = {
    "GPT-4o":                  (-1.0, -2.0, "right"),
    "Claude Haiku 4.5":        (1.0, -1.5, "left"),
    "Claude Sonnet 4.6":       (1.0, -1.2, "left"),
    "Claude Opus 4.6":         (-1.0, 1.2, "right"),
    "GPT-5 Mini":              (-1.0, -1.5, "right"),
    "Gemini 3.1 Pro Preview":  (-1.0, 1.2, "right"),
    "GPT-5.4 Pro":             (1.0, 1.0, "left"),
}

for i, label in enumerate(scatter_labels):
    dx, dy, ha = label_offsets.get(label, (0.8, 0.8, "left"))
    ax.annotate(
        label, (top3[i], spr[i]),
        xytext=(top3[i] + dx, spr[i] + dy),
        fontsize=7.5, ha=ha, va="center",
        color="#333333",
    )

# Shaded danger zone
ax.axhspan(0, 85, color="#fee2e2", alpha=0.3, zorder=0)
ax.axhline(85, color="#ef4444", linewidth=0.8, linestyle="--", alpha=0.5, zorder=1)
ax.text(66, 82.5, "Higher safety risk", fontsize=7.5, color="#ef4444", alpha=0.7, style="italic")

ax.set_xlabel("Top-3 Diagnostic Recall (%)", fontsize=11)
ax.set_ylabel("Safety Pass Rate (%)", fontsize=11)
ax.set_title("Safety Pass Rate vs. Diagnostic Recall (N=500)", fontsize=13, fontweight="bold", pad=12)

ax.set_xlim(63, 92)
ax.set_ylim(62, 100)
ax.grid(True, alpha=0.2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend for providers
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=9, label=p)
    for p, c in PROVIDER_COLORS.items()
]
ax.legend(handles=legend_elements, loc="lower left", fontsize=8, framealpha=0.9)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "fig1_safety_vs_recall.png", dpi=300, bbox_inches="tight")
fig.savefig(OUTPUT_DIR / "fig1_safety_vs_recall.pdf", bbox_inches="tight")
print(f"\nFigure 1 saved to {OUTPUT_DIR / 'fig1_safety_vs_recall.png'}")
plt.close(fig)


# ══════════════════════════════════════════════════════════
# Figure 2: Failure mode breakdown (stacked bar, N=500)
# ══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))

x = np.arange(len(models))
width = 0.25

bars1 = ax.bar(x - width, missed,   width, label="Missed Escalation",  color="#ef4444", alpha=0.85)
bars2 = ax.bar(x,         overconf, width, label="Overconfident Wrong", color="#f59e0b", alpha=0.85)
bars3 = ax.bar(x + width, unsafe_r, width, label="Unsafe Reassurance",  color="#8b5cf6", alpha=0.85)

ax.set_xlabel("Model", fontsize=11)
ax.set_ylabel("Number of Cases (out of 500)", fontsize=11)
ax.set_title("Hard Safety Failure Modes by Model (N=500)", fontsize=13, fontweight="bold", pad=12)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=7.5, ha="center")
ax.legend(fontsize=9, loc="upper right")
ax.set_ylim(0, max(max(missed), max(overconf), max(unsafe_r)) + 8)
ax.grid(True, axis="y", alpha=0.2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.4,
                str(int(h)), ha="center", va="bottom", fontsize=6.5, color="#444444"
            )

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "fig2_failure_modes.png", dpi=300, bbox_inches="tight")
fig.savefig(OUTPUT_DIR / "fig2_failure_modes.pdf", bbox_inches="tight")
print(f"Figure 2 saved to {OUTPUT_DIR / 'fig2_failure_modes.png'}")
plt.close(fig)

print("\nDone. Figures saved as PNG (300dpi) and PDF.")
