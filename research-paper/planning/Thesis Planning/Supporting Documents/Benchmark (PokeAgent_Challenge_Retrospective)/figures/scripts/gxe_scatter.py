"""
GXE scatter plot for Gen1OU and Gen9OU.

Three sections:
  Left   – "Baselines vs. Humans"  (public ranked ladder GXE, with leaderboard bands)
  Middle – "RL Baselines"          (self-play internal eval GXE)
  Right  – "LLM Baselines"         (Gen9OU self-play ladder, horizontal bar chart)

Colour scheme:
  Gold  = PokéAgent Challenge models
  Pink  = Metamon models
  Blue  = LLM (PokéChamp / Base)
  Grey  = Heuristic
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np

from ladder_gxe_stats import parse_gxe_from_ladder

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Lato", "Helvetica", "Arial", "DejaVu Sans"],
    "axes.unicode_minus": False,
})

# --------------------------------------------------------------------------- #
#  PUBLIC LADDER DATA
# --------------------------------------------------------------------------- #

gen1ou_data = [
    ("PokeEnv Heuristic",  26.7, "old", "heuristic"),
    ("Small-IL",           46.9, "old", "metamon"),
    ("Large-IL",           47.0, "old", "metamon"),
    ("Large-RL",           58.5, "old", "metamon"),
    ("SynRL-V0",           69.4, "old", "metamon"),
    ("SynRL-V1+SP",        70.4, "old", "metamon"),
    ("SynRL-V1",           74.2, "old", "metamon"),
    ("SyntheticRLV2",      77.0, "old", "metamon"),
    ("Kadabra3",           80.0, "new", "metamon"),
    ("Kakuna",             82.0, "new", "metamon"),
]

gen9ou_data = [
    ("PokeEnv Heuristic",  39.7, "old", "heuristic"),
    ("SmallRLGen9Beta",    46.7, "new", "metamon"),
    ("Abra",               48.9, "new", "metamon"),
    ("Kadabra",            50.0, "new", "metamon"),
    ("Kadabra2",           61.0, "new", "metamon"),
    ("Kadabra3",           64.0, "new", "metamon"),
    ("Kakuna",             71.0, "new", "metamon"),
]

# --------------------------------------------------------------------------- #
#  INTERNAL EVAL DATA  (Competitive TeamSet, G1 column for early gens)
#  era="old" → gold (Metamon), era="new" → pink (PokéAgent Challenge)
# --------------------------------------------------------------------------- #

gen1ou_internal = [
    ("Kakuna",          75, "new", "metamon"),
    ("Superkazam",      67, "new", "metamon"),
    ("Kadabra4",        66, "new", "metamon"),
    ("Kadabra3",        68, "new", "metamon"),
    ("Kadabra2",        67, "new", "metamon"),
    ("Alakazam",        66, "new", "metamon"),
    ("Kadabra",         56, "new", "metamon"),
    ("Abra",            39, "new", "metamon"),
    ("Minikazam",       39, "new", "metamon"),
    ("SynRLV2",         50, "old", "metamon"),
    ("SynRLV1++",       43, "old", "metamon"),
    ("SynRLV1",         43, "old", "metamon"),
    ("SynRLV0",         41, "old", "metamon"),
    ("LargeRL",         25, "old", "metamon"),
    ("SmallILFA",       24, "old", "metamon"),
]

gen9ou_internal = [
    ("Kakuna",          76, "new", "metamon"),
    ("Superkazam",      75, "new", "metamon"),
    ("Kadabra4",        75, "new", "metamon"),
    ("Kadabra3",        73, "new", "metamon"),
    ("Kadabra2",        73, "new", "metamon"),
    ("Alakazam",        73, "new", "metamon"),
    ("Abra",            61, "new", "metamon"),
    ("Kadabra",         58, "new", "metamon"),
    ("Minikazam",       50, "new", "metamon"),
    ("SmallRLGen9Beta", 56, "new", "metamon"),
    ("SynRLV0",         32, "old", "metamon"),
    ("SynRLV2",         32, "old", "metamon"),
    ("SynRLV1++",       32, "old", "metamon"),
    ("LargeRL",         29, "old", "metamon"),
    ("SynRLV1",         31, "old", "metamon"),
    ("SmallILFA",       23, "old", "metamon"),
]

# --------------------------------------------------------------------------- #
#  LLM BASELINES DATA  (Gen9OU self-play ladder)
# --------------------------------------------------------------------------- #

LLM_FRONTIER_COLOR = "#4A90D9"   # blue  — Frontier (not on public ladder)
LLM_LADDER_COLOR   = "#7BB369"   # green — LLM-Ladder (direct prompt, on ladder)
LLM_PC_COLOR       = "#F57C3A"   # orange — PokeChamp scaffold
LLM_HEU_COLOR      = "#9B59B6"   # purple — Heuristic baselines

# (label, tier, GXE, CI_95)
models = [
    ("Gemini 3.1 Pro",       "Base",       90.76, 2.2),
    ("GPT-5.2",              "Base",       89.86, 2.4),
    ("Gemini 3 Flash",       "Base",       82.29, 3.9),
    ("GLM-5",                "Base",       80.54, 4.2),
    ("Gemini 3 Pro",         "Base",       75.66, 4.9),
    ("Claude Opus 4.6",      "Base",       69.58, 5.6),
    ("Grok-3 Mini",          "Base",       60.26, 6.4),
    ("Gemini 2.5 Flash",     "Base",       55.52, 6.5),
    ("Claude Sonnet 4.6",    "Base",       55.11, 6.6),
    ("Grok-3",               "Base",       53.91, 6.6),
    ("MiniMax M2.5",         "Base",       52.43, 6.6),
    ("PC-Gemma3-4B",         "PokeChamp",  46.86, 6.6),
    ("Hermes 4 405B",        "Base",       46.20, 6.6),
    ("DeepSeek V3",          "Base",       42.98, 6.5),
    ("Qwen3-14B",            "Base",       42.61, 6.5),
    ("GPT-oss",              "Base",       42.43, 6.5),
    ("Kimi K2.5",            "Base",       42.37, 6.5),
    ("PC-Gemma3-1B",         "PokeChamp",  41.87, 6.4),
    ("PC-Llama3.1-8B",       "PokeChamp",  41.19, 6.4),
    ("Qwen3.5 Plus",         "Base",       40.39, 6.4),
    ("Gemini 2.5 Flash Lite","Base",       37.28, 6.2),
    ("Qwen3-4B",             "Base",       36.69, 6.1),
    ("Gemma3-12B",           "Base",       36.65, 6.1),
    ("Llama3.1-8B",          "Base",       32.86, 5.8),
    ("Qwen3-8B",             "Base",       29.18, 5.5),
    ("Gemma3-4B",            "Base",       25.21, 5.0),
    ("Gemma3-1B",            "Base",       12.95, 3.0),
]
llm_baselines = models

# --------------------------------------------------------------------------- #
#  LADDER DATA
# --------------------------------------------------------------------------- #

SCRIPT_DIR = Path(__file__).resolve().parent
gen1ou_ladder_gxe = parse_gxe_from_ladder(SCRIPT_DIR / "gen1ou_ladder.tsv")
gen9ou_ladder_gxe = parse_gxe_from_ladder(SCRIPT_DIR / "gen9ou_ladder.tsv")

# --------------------------------------------------------------------------- #
#  STYLE
# --------------------------------------------------------------------------- #

METAMON_COLOR = "#F5B800"
METAMON_GLOW  = "#FFD54F"
POKECHAMP_COLOR = "#2E9FE8"
POKECHAMP_GLOW  = "#7ECBFF"
HEURISTIC_COLOR = "#666666"
PREV_COLOR = "#E8649A"
PREV_GLOW  = "#FFB3D1"

ALPHA_OLD = 1.0
ALPHA_NEW = 1.0
SIZE_OLD = 72
SIZE_NEW = 100

BAND_COLOR_90 = "#BBCFE3"
BAND_COLOR_IQR = "#7AA3CC"

FAMILY_STYLE = {
    "metamon":   {"color": METAMON_COLOR,   "glow": METAMON_GLOW,   "marker": "o"},
    "pokechamp": {"color": POKECHAMP_COLOR, "glow": POKECHAMP_GLOW, "marker": "o"},
    "heuristic": {"color": HEURISTIC_COLOR, "glow": None,           "marker": "D"},
}


def _style(era, family="metamon"):
    if family == "heuristic":
        return 1.0, 80
    if era == "old":
        return ALPHA_OLD, SIZE_OLD
    return ALPHA_NEW, SIZE_NEW


def _color_glow(era, family):
    """Gold for Metamon (era=old), pink for PokéAgent Challenge (era=new)."""
    if family == "heuristic":
        return HEURISTIC_COLOR, None
    if family == "metamon" and era == "new":
        return PREV_COLOR, PREV_GLOW
    fs = FAMILY_STYLE[family]
    return fs["color"], fs.get("glow")


def plot_strip(ax, data, title, ladder_gxe=None, label_bands=False,
               band_border=False, band_xmin=0.0, band_xmax=1.0, ymin=20, ymax=90):
    """Draw a single clean vertical strip with all dots on one axis."""

    dot_cx = 0.0

    if ladder_gxe is not None and len(ladder_gxe) > 0:
        p10, p90 = np.percentile(ladder_gxe, 10), np.percentile(ladder_gxe, 90)
        q1, q3 = np.percentile(ladder_gxe, 25), np.percentile(ladder_gxe, 75)
        ax.axhspan(p10, p90, xmin=band_xmin, xmax=band_xmax,
                   color=BAND_COLOR_90, alpha=0.25, zorder=1, linewidth=0)
        ax.axhspan(q1, q3, xmin=band_xmin, xmax=band_xmax,
                   color=BAND_COLOR_IQR, alpha=0.28, zorder=2, linewidth=0)
        for y in (p10, p90):
            ax.axhline(y, xmin=band_xmin, xmax=band_xmax,
                       color=BAND_COLOR_90, alpha=0.35, lw=0.5, ls="--", zorder=2)
        for y in (q1, q3):
            ax.axhline(y, xmin=band_xmin, xmax=band_xmax,
                       color=BAND_COLOR_IQR, alpha=0.4, lw=0.5, ls="--", zorder=2)

        if label_bands or band_border:
            stroke = []
            BAND_BORDER = "#1a3a5c"

            import matplotlib.transforms as mtransforms
            from matplotlib.patches import FancyBboxPatch

            trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
            w = band_xmax - band_xmin
            h = p90 - p10
            pad = 0.015  # rounding radius in axes-x units

            def _rbox(x, y, w, h, lw, alpha, color, z):
                p = FancyBboxPatch(
                    (x + pad, y), w - 2*pad, h,
                    boxstyle=f"round,pad={pad}",
                    fill=False, edgecolor=color, linewidth=lw, alpha=alpha,
                    transform=trans, zorder=z, clip_on=False,
                    mutation_aspect=1/8,   # compress rounding vertically
                )
                ax.add_patch(p)

            # crisp rounded border
            _rbox(band_xmin, p10, w, h, 1.8, 1.0, "#3a92cc", 2.5)

            ax.text(
                0.5, p90 + 0.5, "Top 500 Leaderboard",
                transform=ax.get_yaxis_transform(),
                fontsize=8.5, color="#2d7db3", ha="center", va="bottom",
                fontweight="bold",
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
                clip_on=False, zorder=10,
            )

            lx = band_xmin + 0.02
            ax.text(
                lx, (q1 + q3) / 2, "IQR",
                transform=ax.get_yaxis_transform(),
                fontsize=6.5, color="#1a3a5c", ha="left", va="center",
                fontweight="bold", path_effects=stroke, clip_on=False,
            )

            ax.text(
                lx, p90 - 1.2, "90th Percentile",
                transform=ax.get_yaxis_transform(),
                fontsize=5.5, color="#1a3a5c", ha="left", va="top",
                fontstyle="italic", path_effects=stroke, clip_on=False,
            )

            ax.text(
                lx, p10 + 0.3, "10th Percentile",
                transform=ax.get_yaxis_transform(),
                fontsize=5.5, color="#1a3a5c", ha="left", va="bottom",
                fontstyle="italic", path_effects=stroke, clip_on=False,
            )

            if label_bands:
                ax.text(
                    lx, (q1 + q3) / 2 - 1.5, "(of the top 500 players)",
                    transform=ax.get_yaxis_transform(),
                    fontsize=4.4, color="#1a3a5c", ha="left", va="top",
                    fontstyle="italic", path_effects=stroke, clip_on=False,
                )

    rng = np.random.default_rng(42)
    for _label, gxe, era, family in data:
        alpha, size = _style(era, family)
        fs = FAMILY_STYLE[family]
        color, glow_color = _color_glow(era, family)
        jitter = dot_cx + rng.uniform(-0.005, 0.005)

        # subtle drop shadow
        ax.scatter(
            jitter + 0.001, gxe - 0.4, c="#000000",
            s=size * 1.2, marker=fs["marker"],
            edgecolors="none", alpha=0.12, zorder=2,
        )

        if glow_color:
            for scale, ga in [(3.5, 0.06), (2.5, 0.11), (1.7, 0.17)]:
                ax.scatter(
                    jitter, gxe, c=glow_color,
                    s=size * scale, marker=fs["marker"],
                    edgecolors="none", alpha=ga, zorder=3,
                )

        ax.scatter(
            jitter, gxe,
            c=color, s=size, alpha=alpha,
            marker=fs["marker"],
            edgecolors="white", linewidths=1.0, zorder=5,
        )
        # ax.annotate(
        #     _label, (jitter, gxe), textcoords="offset points",
        #     xytext=(8, 0), fontsize=3.5, color="#333333", va="center",
        #     zorder=10,
        # )

    ax.set_ylim(ymin, ymax)
    ax.set_xlim(-0.04, 0.04)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])

    ax.grid(axis="y", color="#aaaaaa", alpha=0.45, lw=0.7, zorder=0, linestyle="--")

    arrow_kw = dict(arrowstyle="<|-|>", color="#222222", lw=1.5, mutation_scale=11)
    pad = 1.5
    ax.annotate(
        "", xy=(dot_cx, ymax + pad + 2), xytext=(dot_cx, ymin - pad),
        arrowprops=arrow_kw, zorder=0, annotation_clip=False,
    )

    ax.yaxis.set_ticks_position("left")
    ax.tick_params(axis="y", length=3, width=0.6, colors="black", labelsize=8, pad=2)

    ax.set_xlabel(title, fontsize=11, labelpad=18, color="black")
    ax.set_ylabel("GXE (%)", fontsize=10.5, labelpad=3, color="black")


# --------------------------------------------------------------------------- #
#  FIGURE  — 4 scatter panels + bar chart, with 2 spacer columns
# --------------------------------------------------------------------------- #

fig = plt.figure(figsize=(8.6, 3.8), facecolor="white")

gs = gridspec.GridSpec(1, 7,
    width_ratios=[0.18, 0.17, 0.09, 0.11, 0.11, 0.19, 0.22],
    wspace=0.0)

ax1 = fig.add_subplot(gs[0, 0])   # Gen1OU public
ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)   # Gen9OU public
ax_gap1 = fig.add_subplot(gs[0, 2])            # spacer 1
ax3 = fig.add_subplot(gs[0, 3], sharey=ax1)   # Gen1OU internal
ax4 = fig.add_subplot(gs[0, 4], sharey=ax1)   # Gen9OU internal
ax_gap2 = fig.add_subplot(gs[0, 5])            # spacer 2
ax5 = fig.add_subplot(gs[0, 6])               # LLM bar chart

ax_gap1.set_visible(False)
ax_gap2.set_visible(False)

# --- left pair: public ladder ---
plot_strip(ax1, gen1ou_data, "Gen 1 OU", ladder_gxe=gen1ou_ladder_gxe,
           label_bands=True, band_xmax=0.93)
plot_strip(ax2, gen9ou_data, "Gen 9 OU", ladder_gxe=gen9ou_ladder_gxe,
           band_border=True, band_xmin=0.07)

ax2.set_ylabel("")
ax2.tick_params(labelleft=False, left=False)

# --- middle pair: internal evals ---
plot_strip(ax3, gen1ou_internal, "Gen 1 OU")
plot_strip(ax4, gen9ou_internal, "Gen 9 OU")

ax3.set_ylabel("GXE (%)", fontsize=10.5, labelpad=3, color="black")
ax4.set_ylabel("")
ax4.tick_params(labelleft=False, left=False)

# hide inter-panel spines on scatter panels
for ax in (ax1, ax2, ax3, ax4):
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

# --- right panel: LLM baselines horizontal bar chart ---
llm_labels    = [m[0] for m in llm_baselines]
llm_tiers     = [m[1] for m in llm_baselines]
llm_gxes      = np.array([m[2] for m in llm_baselines])
llm_ci95      = np.array([m[3] for m in llm_baselines])
_tier_color_map = {
    "Base":       LLM_FRONTIER_COLOR,
    "PokeChamp":  LLM_PC_COLOR,
}
llm_colors = [_tier_color_map[t] for t in llm_tiers]

y_pos = np.arange(len(llm_labels))[::-1]

from matplotlib.patches import Rectangle as _Rect
bh = 0.65

# downward drop shadow
shadow_offset = 0.8
shadow_drop   = 0.14
for yp, gxe in zip(y_pos, llm_gxes):
    ax5.add_patch(_Rect(
        (shadow_offset, yp - bh/2 + shadow_drop * bh), gxe, bh,
        color="#000000", alpha=0.13, zorder=2, clip_on=True,
    ))

# glow layers — progressively larger, more transparent bars behind each bar
for glow_pad, glow_alpha in [(0.18, 0.09), (0.10, 0.16), (0.04, 0.22)]:
    for yp, gxe, col in zip(y_pos, llm_gxes, llm_colors):
        ax5.add_patch(_Rect(
            (0, yp - bh/2 - glow_pad), gxe, bh + glow_pad * 2,
            color=col, alpha=glow_alpha, zorder=3, clip_on=True,
            linewidth=0,
        ))

bars = ax5.barh(y_pos, llm_gxes, color=llm_colors,
                edgecolor="#222222", linewidth=0.5,
                height=bh, alpha=1.0, zorder=4)

# 95% CI error bars
ax5.errorbar(llm_gxes, y_pos, xerr=llm_ci95, fmt="none",
             ecolor="black", elinewidth=0.8, capsize=2.0, capthick=0.8,
             alpha=0.65, zorder=5)

for yp, gxe, ci in zip(y_pos, llm_gxes, llm_ci95):
    ax5.text(gxe + ci + 1.5, yp, f"{gxe:.0f}%", va="center", fontsize=6,
             fontweight="bold", color="black")

ax5.set_yticks(y_pos)
ax5.set_yticklabels(llm_labels, fontsize=7.5, color="black")
ax5.set_xlabel("GXE (%)", fontsize=10, color="black")
ax5.set_xlim(0, 115)
ax5.set_ylim(-0.8, y_pos.max() + 0.5)
ax5.axvline(x=50, color="#999999", linestyle="--", linewidth=0.8, alpha=0.6, zorder=2)
ax5.spines["top"].set_visible(False)
ax5.spines["right"].set_visible(False)
ax5.spines["left"].set_color("#dddddd")
ax5.spines["bottom"].set_color("#dddddd")
ax5.tick_params(axis="x", labelsize=7, colors="black")
ax5.tick_params(axis="y", length=0, pad=3)
ax5.grid(False)

# --------------------------------------------------------------------------- #
#  SECTION TITLES  — centered above each group using axes bbox midpoints
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
#  LEGEND
# --------------------------------------------------------------------------- #

_mk = dict(color="none", markeredgecolor="white", markeredgewidth=1.0)
handles = [
    mlines.Line2D([], [], marker="o", markersize=10, markerfacecolor=PREV_COLOR,
                  alpha=1.0, label="RL (PokéAgent)", **_mk),
    mlines.Line2D([], [], marker="o", markersize=10, markerfacecolor=METAMON_COLOR,
                  alpha=1.0, label="RL (Metamon)", **_mk),
    mlines.Line2D([], [], marker="D", markersize=9, markerfacecolor=HEURISTIC_COLOR,
                  alpha=1.0, label="Heuristic", **_mk),
    mpatches.Patch(facecolor=LLM_FRONTIER_COLOR,   label="LLM (PokéAgent scaffold)", linewidth=0),
    mpatches.Patch(facecolor=LLM_PC_COLOR,        label="LLM (PokéChamp scaffold)", linewidth=0),
]

plt.subplots_adjust(left=0.03, right=0.97, bottom=0.18, top=0.87)

# --------------------------------------------------------------------------- #
#  SECTION TITLES  — placed after subplots_adjust so bbox coords are correct
# --------------------------------------------------------------------------- #

def _section_center(ax_left, ax_right):
    bb_l = ax_left.get_position()
    bb_r = ax_right.get_position()
    return (bb_l.x0 + bb_r.x1) / 2

lc = _section_center(ax1, ax2)
fig.text(lc, 0.985, "Agents vs. Humans",
         ha="center", va="top", fontsize=13.5, fontweight="bold", color="#111111")
fig.text(lc, 0.935, "GXE on Public Ranked Ladder (if Known)",
         ha="center", va="top", fontsize=7.5, color="#555555", fontstyle="italic")

mc = _section_center(ax3, ax4)
fig.text(mc, 0.985, "RL vs. RL",
         ha="center", va="top", fontsize=13.5, fontweight="bold", color="#111111")
fig.text(mc, 0.935, "GXE on Self-Play Ladder (Selected Methods)",
         ha="center", va="top", fontsize=7.5, color="#555555", fontstyle="italic")

bb5 = ax5.get_position()
# Use tight bbox to include model name tick labels in centering
fig.canvas.draw()
_renderer = fig.canvas.get_renderer()
_bb5_tight = ax5.get_tightbbox(_renderer).transformed(fig.transFigure.inverted())
rc = (_bb5_tight.x0 + _bb5_tight.x1) / 2
fig.text(rc, 0.985, "LLM vs. LLM",
         ha="center", va="top", fontsize=13.5, fontweight="bold", color="#111111")
fig.text(rc, 0.935, "GXE on Self-Play Ladder (Gen 9 OU)",
         ha="center", va="top", fontsize=7.5, color="#555555", fontstyle="italic")


leg = fig.legend(
    handles=handles, loc="lower center", ncol=5, fontsize=9.5,
    frameon=False,
    bbox_to_anchor=(0.5, -0.03), bbox_transform=fig.transFigure,
    handletextpad=0.6, columnspacing=1.2,
    handlelength=0.8, borderpad=0.4, labelspacing=0.4,
)

# --------------------------------------------------------------------------- #
#  SAVE
# --------------------------------------------------------------------------- #

out_dir = Path(__file__).resolve().parents[1]
plt.savefig(out_dir / "gxe_scatter.pdf", dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.savefig(out_dir / "gxe_scatter.png", dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
print(f"Saved to {out_dir}/gxe_scatter.{{pdf,png}}")
