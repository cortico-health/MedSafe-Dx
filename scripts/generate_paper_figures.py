#!/usr/bin/env python3
"""
Generate Figure 1 and Figure 2 for the MedSafe-Dx paper.

Figure 1: Safety Pass Rate vs Top-3 Diagnostic Recall (scatter)
Figure 2: Hard safety failure mode breakdown by model (stacked bar)

Uses data from BENCHMARK_REPORT.md Table (Section 4).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Data from benchmark results (N=250 eval set) ──
models = [
    "GPT-5.2",
    "Claude 3.5\nHaiku",
    "GPT-5\nChat",
    "GPT-4o\nMini",
    "GPT-4.1",
    "Claude 3.5\nSonnet",
    "DeepSeek\nChat v3",
    "GPT OSS\n120B",
    "GPT-5\nMini",
    "Gemini 2.0\nFlash",
    "Gemini 3\nPro Prev.",
]

# For scatter labels (no newlines)
model_labels = [
    "GPT-5.2",
    "Claude 3.5 Haiku",
    "GPT-5 Chat",
    "GPT-4o Mini",
    "GPT-4.1",
    "Claude 3.5 Sonnet",
    "DeepSeek Chat v3",
    "GPT OSS 120B",
    "GPT-5 Mini",
    "Gemini 2.0 Flash",
    "Gemini 3 Pro Prev.",
]

spr       = [97.6, 95.6, 94.0, 90.4, 87.6, 87.2, 85.2, 85.2, 84.8, 80.0, 62.4]
top3      = [71.3, 69.9, 79.6, 59.3, 81.3, 84.4, 70.4, 78.9, 77.8, 67.5, 87.2]
missed    = [5,    11,   8,    3,    13,   18,   18,   17,   9,    26,   9]
overconf  = [1,    0,    6,    1,    12,   7,    10,   16,   0,    0,    10]
unsafe_r  = [0,    0,    1,    3,    5,    8,    10,   4,    0,    0,    10]
coverage  = [100,  100,  100,  93,   100,  100,  100,  100,  88,   90,   74]

# Provider colors
provider_colors = {
    "OpenAI":    "#10a37f",
    "Anthropic": "#d97706",
    "Google":    "#4285f4",
    "DeepSeek":  "#7c3aed",
}

providers = [
    "OpenAI", "Anthropic", "OpenAI", "OpenAI", "OpenAI",
    "Anthropic", "DeepSeek", "OpenAI", "OpenAI", "Google", "Google",
]

colors = [provider_colors[p] for p in providers]


# ══════════════════════════════════════════════════════════
# Figure 1: Safety Pass Rate vs Top-3 Recall
# ══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5.5))

ax.scatter(top3, spr, c=colors, s=100, zorder=5, edgecolors="white", linewidths=0.8)

# Label each point
for i, label in enumerate(model_labels):
    # Offset labels to avoid overlap
    dx, dy = 0.8, 0.8
    ha = "left"
    if label == "GPT-5.2":
        dx, dy = -1.0, -1.8
        ha = "right"
    elif label == "Claude 3.5 Haiku":
        dx, dy = 0.8, -1.5
        ha = "left"
    elif label == "GPT OSS 120B":
        dx, dy = -1.0, 0.5
        ha = "right"
    elif label == "DeepSeek Chat v3":
        dx, dy = -1.0, -1.5
        ha = "right"
    elif label == "Claude 3.5 Sonnet":
        dx, dy = 0.8, -1.2
        ha = "left"
    elif label == "GPT-4.1":
        dx, dy = 0.8, 0.5
        ha = "left"
    elif label == "GPT-4o Mini":
        dx, dy = -1.0, 0.5
        ha = "right"
    elif label == "Gemini 3 Pro Prev.":
        dx, dy = -1.0, 1.0
        ha = "right"

    ax.annotate(
        label, (top3[i], spr[i]),
        xytext=(top3[i] + dx, spr[i] + dy),
        fontsize=7.5, ha=ha, va="center",
        color="#333333",
    )

# Shaded danger zone
ax.axhspan(0, 85, color="#fee2e2", alpha=0.3, zorder=0)
ax.axhline(85, color="#ef4444", linewidth=0.8, linestyle="--", alpha=0.5, zorder=1)
ax.text(58, 82.5, "Higher safety risk", fontsize=7.5, color="#ef4444", alpha=0.7, style="italic")

ax.set_xlabel("Top-3 Diagnostic Recall (%)", fontsize=11)
ax.set_ylabel("Safety Pass Rate (%)", fontsize=11)
ax.set_title("Safety Pass Rate vs. Diagnostic Recall", fontsize=13, fontweight="bold", pad=12)

ax.set_xlim(55, 92)
ax.set_ylim(58, 100)
ax.grid(True, alpha=0.2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend for providers
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=9, label=p)
    for p, c in provider_colors.items()
]
ax.legend(handles=legend_elements, loc="lower left", fontsize=8, framealpha=0.9)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "fig1_safety_vs_recall.png", dpi=300, bbox_inches="tight")
fig.savefig(OUTPUT_DIR / "fig1_safety_vs_recall.pdf", bbox_inches="tight")
print(f"Figure 1 saved to {OUTPUT_DIR / 'fig1_safety_vs_recall.png'}")
plt.close(fig)


# ══════════════════════════════════════════════════════════
# Figure 2: Failure mode breakdown (grouped bar)
# ══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))

x = np.arange(len(models))
width = 0.25

bars1 = ax.bar(x - width, missed,   width, label="Missed Escalation",  color="#ef4444", alpha=0.85)
bars2 = ax.bar(x,         overconf, width, label="Overconfident Wrong", color="#f59e0b", alpha=0.85)
bars3 = ax.bar(x + width, unsafe_r, width, label="Unsafe Reassurance",  color="#8b5cf6", alpha=0.85)

ax.set_xlabel("Model", fontsize=11)
ax.set_ylabel("Number of Cases (out of 250)", fontsize=11)
ax.set_title("Hard Safety Failure Modes by Model", fontsize=13, fontweight="bold", pad=12)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=7.5, ha="center")
ax.legend(fontsize=9, loc="upper left")
ax.set_ylim(0, 32)
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
