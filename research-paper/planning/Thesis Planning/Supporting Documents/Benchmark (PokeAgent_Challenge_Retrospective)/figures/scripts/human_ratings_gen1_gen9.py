import matplotlib.pyplot as plt
import numpy as np

formats = ["Gen 1 OU", "Gen 9 OU"]
gxes = [82, 71]
threshold = 65

colors = ["#E91E7E" if g >= threshold else "#8B9FD9" for g in gxes]

fig, ax = plt.subplots(figsize=(5, 5.5))

bars = ax.bar(formats, gxes, color=colors, width=0.5, edgecolor="white", linewidth=1.5)

# Add GXE values on top of bars
for bar, gxe in zip(bars, gxes):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{gxe}%", ha="center", va="bottom", fontsize=14, fontweight="bold")

# Leaderboard threshold line
ax.axhline(y=threshold, color="gray", linestyle="--", linewidth=1.5)
ax.text(1.35, threshold + 0.5, f"Leaderboard\n({threshold}% GXE)",
        fontsize=8, color="gray", va="bottom", ha="center")

ax.set_ylabel("GXE (%)", fontsize=12)
ax.set_xlabel("Format", fontsize=12)
ax.set_ylim(0, 92)
ax.set_title("Kakuna (142M) vs. Human Players", fontsize=12, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#E91E7E", label="Above threshold"),
    Patch(facecolor="#8B9FD9", label="Below threshold"),
]
ax.legend(handles=legend_elements, loc="upper right", fontsize=9, framealpha=0.9)

plt.tight_layout()
plt.savefig("figures/human_ratings_gen1_gen9.pdf", bbox_inches="tight", dpi=300)
plt.savefig("figures/human_ratings_gen1_gen9.png", bbox_inches="tight", dpi=300)
print("Saved to figures/human_ratings_gen1_gen9.pdf and .png")
