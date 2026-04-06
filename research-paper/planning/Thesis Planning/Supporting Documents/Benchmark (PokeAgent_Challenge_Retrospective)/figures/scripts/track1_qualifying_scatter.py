"""
Track 1 Qualifying Results: WHR vs Total Games scatter plot.

Two panels (Gen1OU, Gen9OU). Each entry is plotted with:
  x = Total games (wins + losses)
  y = WHR (Whole-History Rating)
  error bars = WHR ± std

Categories:
  Organizer:
    RL (public)       – publicly released Metamon RL models
    RL (held-out)     – held-out Metamon RL models
    LLM (PokéChamp)   – PokéChamp scaffolding LLM baselines
    LLM (Standard)    – standard LLM scaffolding baselines
    Heuristic         – rule-based baselines
  Participant         – external competitors
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.transforms as mtransforms
import numpy as np

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Lato", "Helvetica", "Arial", "DejaVu Sans"],
    "axes.unicode_minus": False,
})

SCRIPT_DIR = Path(__file__).resolve().parent
JSON_PATH = SCRIPT_DIR / "track1_qualifying.json"
OUT_DIR = SCRIPT_DIR.parent

# ── Colors ────────────────────────────────────────────────────────────────── #

RL_PUBLIC_COLOR = "#F9D84A"
RL_PUBLIC_GLOW = "#FCE88A"
RL_HELDOUT_COLOR = "#F472B6"
RL_HELDOUT_GLOW = "#F9A8D4"
LLM_PC_COLOR = "#FB923C"
LLM_PC_GLOW = "#FDBA74"
LLM_STD_COLOR = "#38BDF8"
LLM_STD_GLOW = "#7DD3FC"
HEURISTIC_COLOR = "#94A3B8"
HEURISTIC_GLOW = "#CBD5E1"
PARTICIPANT_QUAL_COLOR = "#22C55E"
PARTICIPANT_QUAL_GLOW = "#86EFAC"
PARTICIPANT_COLOR = "#EF4444"
PARTICIPANT_GLOW = "#FCA5A5"

# ── Categorization ────────────────────────────────────────────────────────── #

HEURISTIC_ORIGINALS = {"PAC-PC-MAX-POWER", "PAC-PC-ABYSSAL"}

RL_PUBLIC_SUFFIXES = {
    "SynRLV0", "SynRLV1", "SynRLV1-SP", "SynRLV2",
    "SmallILFA", "SmallRLG9", "SmallIL", "Minikazam",
    "Abra", "SmallG9v2",
}


def _strip_metamon_prefix(orig):
    """Strip PAC-MM- prefix and optional generation suffix like -9."""
    name = orig.removeprefix("PAC-MM-")
    return name


def classify(entry):
    if not entry["username"]["is_starter_kit"]:
        return "participant"
    orig = entry["username"]["original"]
    if orig.startswith("PAC-BH-") or orig in HEURISTIC_ORIGINALS:
        return "heuristic"
    if orig.startswith("PAC-LLM-"):
        return "llm_std"
    if orig.startswith("PAC-PC-"):
        return "llm_pc"
    if orig.startswith("PAC-MM-"):
        stripped = _strip_metamon_prefix(orig)
        for suffix in RL_PUBLIC_SUFFIXES:
            if stripped == suffix or stripped.startswith(suffix + "-"):
                return "rl_public"
        return "rl_heldout"
    return "rl_public"


CATEGORY_STYLE = {
    "rl_public":   {"color": RL_PUBLIC_COLOR,   "glow": RL_PUBLIC_GLOW,   "marker": "o", "label": "RL (Public)"},
    "rl_heldout":  {"color": RL_HELDOUT_COLOR,  "glow": RL_HELDOUT_GLOW,  "marker": "o", "label": "RL (Held-out)"},
    "llm_pc":      {"color": LLM_PC_COLOR,      "glow": LLM_PC_GLOW,      "marker": "s", "label": "LLM (PokéChamp)"},
    "llm_std":     {"color": LLM_STD_COLOR,     "glow": LLM_STD_GLOW,     "marker": "s", "label": "LLM (Standard)"},
    "heuristic":   {"color": HEURISTIC_COLOR,    "glow": HEURISTIC_GLOW,   "marker": "^", "label": "Heuristic"},
    "participant_qual": {"color": PARTICIPANT_QUAL_COLOR, "glow": PARTICIPANT_QUAL_GLOW, "marker": "*", "label": "Winning Participant"},
    "participant":      {"color": PARTICIPANT_COLOR,      "glow": PARTICIPANT_GLOW,      "marker": "D", "label": "Participant"},
}

# ── Data loading ──────────────────────────────────────────────────────────── #

with open(JSON_PATH) as f:
    data = json.load(f)


def extract_entries(format_key):
    """Return list of dicts with elo, whr, whr_lo, whr_hi, name, category."""
    entries = []
    for e in data["formats"][format_key]:
        if "whr" not in e:
            continue
        wins = int(e["wins"])
        losses = int(e["losses"])
        total = wins + losses
        entries.append({
            "elo": int(e["elo"]),
            "whr": e["whr"]["whr_elo"],
            "whr_std": e["whr"]["whr_std"],
            "games": total,
            "winrate": wins / total * 100 if total > 0 else 0,
            "gxe": float(e["gxe"].rstrip("%")),
            "name": e["username"]["display"],
            "category": classify(e),
        })
    return entries


def mark_qualified_participants(entries, top_n=8):
    """Mark top N participants by WHR as 'participant_qual', rest as 'participant'."""
    participants = sorted(
        [e for e in entries if e["category"] == "participant"],
        key=lambda e: e["whr"], reverse=True,
    )
    qual_names = {e["name"] for e in participants[:top_n]}
    for e in entries:
        if e["category"] == "participant" and e["name"] in qual_names:
            e["category"] = "participant_qual"
    return entries


gen1ou = mark_qualified_participants(extract_entries("gen1ou"))
gen9ou = mark_qualified_participants(extract_entries("gen9ou"))


def add_gxe_residuals(entries):
    """Add whr_resid field: WHR minus the linear fit of WHR on GXE."""
    xs = np.array([e["gxe"] for e in entries])
    ys = np.array([e["whr"] for e in entries])
    m, b = np.polyfit(xs, ys, 1)
    for e in entries:
        e["whr_resid"] = e["whr"] - (m * e["gxe"] + b)
    return entries, m, b


gen1ou, _m1, _b1 = add_gxe_residuals(gen1ou)
gen9ou, _m9, _b9 = add_gxe_residuals(gen9ou)

# ── Plotting ──────────────────────────────────────────────────────────────── #

def _scatter_on_ax(ax, entries, x_field, y_field="whr"):
    """Plot all categories on a single axes. Returns (xs, ys, names) for participants."""
    from adjustText import adjust_text

    label_points = []  # (x, y, name) for participant labels

    for cat_key in ["heuristic", "rl_public", "rl_heldout", "llm_pc", "llm_std", "participant", "participant_qual"]:
        style = CATEGORY_STYLE[cat_key]
        subset = [e for e in entries if e["category"] == cat_key]
        if not subset:
            continue

        xs = np.array([e[x_field] for e in subset])
        ys = np.array([e[y_field] for e in subset])
        stds = np.array([e["whr_std"] for e in subset])

        # Error bars only when raw WHR is on y-axis
        if y_field == "whr" and x_field != "whr_std":
            ax.errorbar(
                xs, ys, yerr=stds,
                fmt="none", ecolor=style["color"], elinewidth=1.2,
                capsize=3, capthick=0.8, alpha=0.45, zorder=3,
            )

        organizer_cats = {"rl_public", "rl_heldout", "llm_pc", "llm_std", "heuristic"}
        pt_size = 500 if cat_key == "participant_qual" else (160 if cat_key == "heuristic" else (110 if cat_key in organizer_cats else 200))

        if style["glow"]:
            for scale, ga in [(2.0, 0.05), (1.5, 0.09)]:
                ax.scatter(
                    xs, ys, c=style["glow"],
                    s=pt_size * scale, marker=style["marker"],
                    edgecolors="none", alpha=ga, zorder=4,
                )

        is_participant = cat_key in ("participant", "participant_qual")

        if is_participant:
            shadow_tf = ax.transData + mtransforms.Affine2D().translate(3, -3)
            ax.scatter(
                xs, ys, c="#000000", s=pt_size * 1.1, marker=style["marker"],
                edgecolors="none", alpha=0.22, zorder=5, transform=shadow_tf,
            )

        ax.scatter(
            xs, ys,
            c=style["color"], s=pt_size, marker=style["marker"],
            edgecolors="#222222" if is_participant else "white",
            linewidths=0.6 if is_participant else 0.8,
            alpha=1.0, zorder=6,
            label=style["label"],
        )

        if is_participant:
            for e in subset:
                label_points.append((e[x_field], e[y_field], e["name"]))

    # Add labels with collision avoidance
    texts = []
    for x, y, name in label_points:
        t = ax.text(
            x, y, name,
            fontsize=11, color="#111111", va="bottom", zorder=10,
            fontweight="semibold",
            path_effects=[pe.withStroke(linewidth=3.5, foreground="white")],
        )
        texts.append(t)

    if texts:
        all_xs = [x for x, y, _ in label_points]
        all_ys = [y for x, y, _ in label_points]
        adjust_text(
            texts, x=all_xs, y=all_ys, ax=ax,
            arrowprops=dict(
                arrowstyle="-", color="#888888", lw=0.7,
                shrinkA=3, shrinkB=6,
            ),
            expand=(1.2, 1.4),
            force_text=(0.15, 0.2),
            force_static=(0.2, 0.3),
            force_pull=(0.06, 0.08),
            max_move=(25, 25),
            min_arrow_len=6,
        )


def _style_ax(ax):
    ax.grid(True, alpha=0.35, lw=0.7, linestyle="--", color="#bbbbbb", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.tick_params(labelsize=13, colors="black", length=4, width=0.9)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("black")
        ax.spines[spine].set_linewidth(0.9)



FIELD_LABELS = {
    "elo": "Online ELO ↑",
    "games": "Battles Played ↑",
    "winrate": "Win Rate (%) ↑",
    "gxe": "GXE (%) ↑",
    "whr_std": "Uncertainty ↓",
    "whr": "FH-BT ↑",
    "whr_resid": "FH-BT Deviation from GXE Trend",
}


def plot_panel_simple(ax, entries, title, x_field="elo", y_field="whr"):
    """Single-axis panel."""
    _scatter_on_ax(ax, entries, x_field, y_field)
    ax.set_xlabel(FIELD_LABELS.get(x_field, x_field), fontsize=18, color="black", labelpad=10)
    ax.set_ylabel(FIELD_LABELS.get(y_field, y_field), fontsize=18, color="black")
    ax.set_title(title, fontsize=22, fontweight="bold", color="#111111", pad=10)
    _style_ax(ax)
    ax.spines["right"].set_visible(False)




# ── Figure generation ─────────────────────────────────────────────────────── #

_mk = dict(markeredgecolor="white", markeredgewidth=0.8, color="none")

all_handles = [
    mlines.Line2D([], [], marker="o", markersize=16,
                  markerfacecolor=RL_PUBLIC_COLOR, label="Organizer RL (Public)", **_mk),
    mlines.Line2D([], [], marker="o", markersize=16,
                  markerfacecolor=RL_HELDOUT_COLOR, label="Organizer RL (Held-out)", **_mk),
    mlines.Line2D([], [], marker="s", markersize=16,
                  markerfacecolor=LLM_PC_COLOR, label="Organizer LLM (PokéChamp)", **_mk),
    mlines.Line2D([], [], marker="s", markersize=16,
                  markerfacecolor=LLM_STD_COLOR, label="Organizer LLM (Standard)", **_mk),
    mlines.Line2D([], [], marker="^", markersize=18,
                  markerfacecolor=HEURISTIC_COLOR, label="Organizer Heuristic", **_mk),
    mlines.Line2D([], [], marker="D", markersize=14,
                  markerfacecolor=PARTICIPANT_COLOR, label="Participant", **_mk),
    mlines.Line2D([], [], marker="*", markersize=24,
                  markerfacecolor=PARTICIPANT_QUAL_COLOR, label="Winning Participant", **_mk),
]




def make_simple_figure(x_field, suffix, suptitle, y_field="whr", figsize=(16, 5)):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, facecolor="white")

    g1, g9 = gen1ou, gen9ou
    if "whr_std" in (x_field, y_field):
        g1 = [e for e in g1 if e["name"] != "Metamon-Kadabra-9"]
        g9 = [e for e in g9 if e["name"] != "Metamon-Kadabra-9"]

    plot_panel_simple(ax1, g1, "Gen 1 OU", x_field=x_field, y_field=y_field)
    plot_panel_simple(ax2, g9, "Gen 9 OU", x_field=x_field, y_field=y_field)

    n_cols = min(len(all_handles), 4) if figsize[0] < 14 else len(all_handles)
    leg_fontsize = 10 if figsize[0] < 14 else 12
    fig.subplots_adjust(bottom=0.14, top=0.87, left=0.08, right=0.98, wspace=0.25)

    fig.legend(
        handles=all_handles, loc="lower center", ncol=n_cols, fontsize=leg_fontsize,
        frameon=False, bbox_to_anchor=(0.5, 0.01),
        handletextpad=0.4, columnspacing=1.0,
    )

    fig.suptitle(suptitle, fontsize=15, fontweight="bold",
                 color="#111111", y=0.94)

    plt.savefig(f"{OUT_DIR}/track1_qualifying_{suffix}.pdf", dpi=300,
                bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.savefig(f"{OUT_DIR}/track1_qualifying_{suffix}.png", dpi=300,
                bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"Saved to {OUT_DIR}/track1_qualifying_{suffix}.{{pdf,png}}")
    plt.close(fig)


TITLE = "Battling Track Qualifying Stage"

# ── Main text figure: WHR vs. GXE ────────────────────────────────────────── #
make_simple_figure("gxe", "scatter_main", TITLE, figsize=(11, 7))

# ── Appendix figure: remaining 4 x-axes stacked ──────────────────────────── #
APPENDIX_PANELS = [
    ("elo",     "Online ELO"),
    ("games",   "Battles Played"),
    ("winrate", "Win Rate (%)"),
    ("gxe",     "GXE (%)"),
]

def make_appendix_figure():
    n = len(APPENDIX_PANELS)
    fig, axes = plt.subplots(n, 2, figsize=(16, 5.5 * n), facecolor="white",
                             gridspec_kw={"hspace": 0.40})

    for row, panel in enumerate(APPENDIX_PANELS):
        x_field = panel[0]
        y_field = panel[2] if len(panel) > 2 else "whr"
        ax1, ax2 = axes[row]
        col1_title = "Gen 1 OU" if row == 0 else ""
        col2_title = "Gen 9 OU" if row == 0 else ""
        g1, g9 = gen1ou, gen9ou
        if "whr_std" in (x_field, y_field):
            g1 = [e for e in g1 if e["name"] != "Metamon-Kadabra-9"]
            g9 = [e for e in g9 if e["name"] != "Metamon-Kadabra-9"]
        plot_panel_simple(ax1, g1, col1_title, x_field=x_field, y_field=y_field)
        plot_panel_simple(ax2, g9, col2_title, x_field=x_field, y_field=y_field)
        ax2.set_ylabel("")

    fig.legend(
        handles=all_handles, loc="lower center", ncol=4, fontsize=18,
        frameon=False, bbox_to_anchor=(0.5, 0.01),
        handletextpad=0.5, columnspacing=1.2,
    )

    fig.suptitle(TITLE, fontsize=26, fontweight="bold", color="#111111", y=0.995)
    fig.subplots_adjust(top=0.93, bottom=0.10, left=0.07, right=0.98, hspace=0.40)


    plt.savefig(f"{OUT_DIR}/track1_qualifying_scatter_appendix.pdf", dpi=300,
                bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.savefig(f"{OUT_DIR}/track1_qualifying_scatter_appendix.png", dpi=300,
                bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"Saved to {OUT_DIR}/track1_qualifying_scatter_appendix.{{pdf,png}}")
    plt.close(fig)

make_appendix_figure()
