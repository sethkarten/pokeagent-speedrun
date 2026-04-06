import matplotlib.pyplot as plt
import numpy as np

# Data: (label, tier, GXE, CI_95)
# Tiers: Base, PokeChamp, Heuristic
# CI_95 = half-width of 95% confidence interval on GXE
models = [
    ("Gemini 3.1 Pro",       "Base",       90.68, 2.2),
    ("GPT-5",                "Base",       89.79, 2.4),
    ("Gemini 3 Flash",       "Base",       82.17, 3.9),
    ("GLM-5",                "Base",       80.40, 4.2),
    ("Gemini 3 Pro",         "Base",       75.50, 4.9),
    ("Claude Opus 4.6",      "Base",       69.34, 5.6),
    ("Grok-3 Mini",          "Base",       60.06, 6.4),
    ("Gemini 2.5 Flash",     "Base",       55.35, 6.5),
    ("Claude Sonnet 4.6",    "Base",       54.83, 6.6),
    ("Grok-3",               "Base",       53.63, 6.6),
    ("PC-Gemma3-4B",         "PokeChamp",  46.71, 6.6),
    ("Hermes 4 405B",        "Base",       46.15, 6.6),
    ("DeepSeek V3",          "Base",       42.67, 6.5),
    ("Qwen3-14B",            "Base",       42.50, 6.5),
    ("GPT-4o",               "Base",       42.26, 6.5),
    ("PC-Gemma3-1B",         "PokeChamp",  41.75, 6.4),
    ("PC-Llama3.1-8B",       "PokeChamp",  41.02, 6.4),
    ("Qwen3.5 Plus",         "Base",       40.11, 6.4),
    ("Gemini 2.5 Flash Lite","Base",       37.14, 6.2),
    ("Qwen3-4B",             "Base",       36.52, 6.1),
    ("Gemma3-12B",           "Base",       36.50, 6.1),
    ("Llama3.1-8B",          "Base",       32.71, 5.8),
    ("Qwen3-8B",             "Base",       29.03, 5.5),
    ("Gemma3-4B",            "Base",       25.09, 5.0),
    ("Gemma3-1B",            "Base",       12.88, 3.0),
]

# Sort descending by GXE
models = sorted(models, key=lambda x: x[2], reverse=True)

labels = [m[0] for m in models]
tiers  = [m[1] for m in models]
gxes   = np.array([m[2] for m in models])
ci95   = np.array([m[3] for m in models])

tier_colors = {
    "Base":      "#4A90D9",  # blue
    "PokeChamp": "#E8833A",  # orange
}
colors = [tier_colors[t] for t in tiers]

n = len(models)
fig, ax = plt.subplots(figsize=(6, n * 0.32 + 1.0))

y_pos = np.arange(n)[::-1]
ax.barh(y_pos, gxes, color=colors, edgecolor="white", height=0.7)

# Error bars for 95% CI
ax.errorbar(gxes, y_pos, xerr=ci95, fmt="none",
            ecolor="black", elinewidth=0.8, capsize=2.5, capthick=0.8, alpha=0.7)

# GXE value labels
for yp, gxe, ci in zip(y_pos, gxes, ci95):
    ax.text(gxe + ci + 1.0, yp, f"{gxe:.1f}%",
            va="center", fontsize=7.5, fontweight="bold")

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=8.5)
ax.set_xlabel("GXE — expected win rate vs. pool-average opponent (%)", fontsize=10)
ax.set_xlim(0, 105)

# 50% reference line
ax.axvline(x=50, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
ax.text(50.5, y_pos[0] + 0.8, "50%", color="gray", fontsize=8)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=tier_colors["Base"],      label="Base (direct prompt)"),
    Patch(facecolor=tier_colors["PokeChamp"], label="PokeChamp scaffold"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=8, framealpha=0.9)

ax.set_title("Gen 9 OU LLM Baselines (95% CI)", fontsize=11, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("figures/gen9_baselines_chart.pdf", bbox_inches="tight", dpi=300)
plt.savefig("figures/gen9_baselines_chart.png", bbox_inches="tight", dpi=300)
print("Saved to figures/gen9_baselines_chart.pdf and .png")
