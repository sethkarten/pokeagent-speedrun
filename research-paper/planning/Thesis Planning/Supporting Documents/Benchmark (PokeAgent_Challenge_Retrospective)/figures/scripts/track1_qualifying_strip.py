"""
Track 1 Qualifying – scatter styled like gxe_scatter.py.

Layout: two sections side-by-side (Gen1OU | Gen9OU), each with two panels:
  Left panel   – Organizer baselines
  Right panel  – Participants

Within each section the two panels share a y-axis.
Y-axis: Bradley-Terry Elo (absolute rating)
X-axis: WHR deviation from GXE linear trend (positive = outperforms GXE prediction)
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.patheffects as pe
import matplotlib.transforms as mtransforms
import numpy as np

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Lato", "Helvetica", "Arial", "DejaVu Sans"],
    "axes.unicode_minus": False,
})

SCRIPT_DIR = Path(__file__).resolve().parent
JSON_PATH  = SCRIPT_DIR / "track1_qualifying.json"
OUT_DIR    = SCRIPT_DIR.parent

def _lighten(hex_color, amount=0.55):
    """Blend hex_color toward white by `amount` (0=original, 1=white)."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f"#{r:02X}{g:02X}{b:02X}"

# ── Colors ────────────────────────────────────────────────────────────────── #
RL_PUBLIC_COLOR        = "#F0A500"
RL_PUBLIC_GLOW         = "#FFC933"
RL_HELDOUT_COLOR       = "#E0357A"
RL_HELDOUT_GLOW        = "#FF6EA8"
LLM_PC_COLOR           = "#F06020"
LLM_PC_GLOW            = "#FF8C55"
LLM_STD_COLOR          = "#2196F3"
LLM_STD_GLOW           = "#64B5F6"
HEURISTIC_COLOR        = "#999999"
HEURISTIC_GLOW         = None
PARTICIPANT_QUAL_COLOR = "#52C882"
PARTICIPANT_QUAL_GLOW  = "#96E0B8"
PARTICIPANT_COLOR      = "#E04040"
PARTICIPANT_GLOW       = "#F08080"

CATEGORY_STYLE = {
    "rl_public":        {"color": RL_PUBLIC_COLOR,        "glow": RL_PUBLIC_GLOW,        "marker": "o"},
    "rl_heldout":       {"color": RL_HELDOUT_COLOR,       "glow": RL_HELDOUT_GLOW,       "marker": "o"},
    "llm_pc":           {"color": LLM_PC_COLOR,           "glow": LLM_PC_GLOW,           "marker": "s"},
    "llm_std":          {"color": LLM_STD_COLOR,          "glow": LLM_STD_GLOW,          "marker": "s"},
    "heuristic":        {"color": HEURISTIC_COLOR,        "glow": HEURISTIC_GLOW,        "marker": "^"},
    "participant":      {"color": PARTICIPANT_COLOR,      "glow": PARTICIPANT_GLOW,      "marker": "D"},
    "participant_qual": {"color": PARTICIPANT_QUAL_COLOR, "glow": PARTICIPANT_QUAL_GLOW, "marker": "*"},
}

ORGANIZER_CATS   = {"rl_public", "rl_heldout", "llm_pc", "llm_std", "heuristic"}
PARTICIPANT_CATS = {"participant", "participant_qual"}

CATEGORY_ZORDER = {
    "heuristic":        5,
    "llm_std":          6,
    "llm_pc":           7,
    "rl_public":        8,
    "rl_heldout":       9,
    "participant":      10,
    "participant_qual": 11,
}

# ── Data loading ──────────────────────────────────────────────────────────── #

HEURISTIC_ORIGINALS = {"PAC-PC-MAX-POWER", "PAC-PC-ABYSSAL"}
RL_PUBLIC_SUFFIXES  = {
    "SynRLV0", "SynRLV1", "SynRLV1-SP", "SynRLV2",
    "SmallILFA", "SmallRLG9", "SmallIL", "Minikazam",
    "Abra", "SmallG9v2",
}

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
        name = orig.removeprefix("PAC-MM-")
        for suf in RL_PUBLIC_SUFFIXES:
            if name == suf or name.startswith(suf + "-"):
                return "rl_public"
        return "rl_heldout"
    return "rl_public"

with open(JSON_PATH) as f:
    raw = json.load(f)

def extract_entries(fmt):
    entries = []
    for e in raw["formats"][fmt]:
        if "whr" not in e:
            continue
        entries.append({
            "whr":      e["whr"]["whr_elo"],
            "whr_std":  e["whr"]["whr_std"],
            "gxe":      float(e["gxe"].rstrip("%")),
            "name":     e["username"]["display"],
            "category": classify(e),
        })
    return entries

def mark_qual(entries, top_n=8):
    parts = sorted([e for e in entries if e["category"] == "participant"],
                   key=lambda e: e["whr"], reverse=True)
    qual = {e["name"] for e in parts[:top_n]}
    for e in entries:
        if e["category"] == "participant" and e["name"] in qual:
            e["category"] = "participant_qual"
    return entries

def add_residuals(entries):
    """Add `resid` = WHR minus linear prediction from GXE."""
    xs = np.array([e["gxe"] for e in entries])
    ys = np.array([e["whr"] for e in entries])
    m, b = np.polyfit(xs, ys, 1)
    for e in entries:
        e["resid"] = e["whr"] - (m * e["gxe"] + b)
    return entries

gen1ou = add_residuals(mark_qual(extract_entries("gen1ou")))
gen1ou = [e for e in gen1ou if "G1Boss" not in e["name"]]
gen9ou = add_residuals(mark_qual(extract_entries("gen9ou")))

# ── Panel plot function ───────────────────────────────────────────────────── #

def _pt_size(cat):
    if cat == "participant_qual": return 260
    if cat == "participant":      return 85
    if cat == "heuristic":       return 90
    if cat in ORGANIZER_CATS:    return 75
    return 120

def plot_panel(ax, entries, cat_filter,
               show_ylabel=False, show_yticks=True, show_xticks=True,
               panel_label="", title="", xpad=None):
    subset = [e for e in entries if e["category"] in cat_filter]
    subset = sorted(subset, key=lambda e: CATEGORY_ZORDER[e["category"]])

    # grid & spines — y-grid prominent, x-grid lighter
    ax.grid(axis="y", color="#999999", alpha=0.30, lw=0.8, zorder=0, linestyle="--")
    ax.grid(axis="x", color="#cccccc", alpha=0.25, lw=0.5, zorder=0, linestyle=":")

    # vertical reference at x=0
    ax.axvline(0, color="black", lw=1.0, ls="-", alpha=0.8, zorder=1)

    for e in subset:
        cat   = e["category"]
        style = CATEGORY_STYLE[cat]
        x, y  = e["resid"], e["whr"]
        pt    = _pt_size(cat)
        z     = CATEGORY_ZORDER[cat]

        # subtle drop shadow
        shadow_tf = ax.transData + mtransforms.Affine2D().translate(1.2, -1.2)
        ax.scatter(x, y, c="#000000", s=pt * 1.05,
                   marker=style["marker"], edgecolors="none",
                   alpha=0.08, zorder=z - 0.3, transform=shadow_tf)

        ec = _lighten(style["color"], 0.42)
        lw = 0.8
        ax.scatter(x, y, c=style["color"], s=pt,
                   marker=style["marker"], edgecolors=ec,
                   linewidths=lw, alpha=1.0, zorder=z)


    # axes limits with padding
    xvals = [e["resid"] for e in subset]
    yvals = [e["whr"]   for e in subset]
    if xvals:
        pad  = xpad if xpad else max(abs(v) for v in xvals) * 0.12
        half = max(abs(v) for v in xvals) + pad
        ax.set_xlim(-half, half)   # always centred at zero

    # all spines hidden — x-axis line is drawn as a double-headed arrow in _add_axis_arrows
    for sp in ax.spines.values():
        sp.set_visible(False)

    import matplotlib.ticker as mticker
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=7, integer=True))
    ax.tick_params(axis="y", labelsize=8, colors="#333333", length=3, width=0.6)
    ax.tick_params(axis="x", labelsize=8, colors="#333333", length=3, width=0.6)

    if not show_yticks:
        ax.tick_params(labelleft=False, left=False)
    if not show_xticks:
        ax.tick_params(labelbottom=False, bottom=False)

    if show_ylabel:
        ax.set_ylabel("Skill Rating (FH-BT) ↑", fontsize=10, labelpad=4, color="black")

    ax.set_xlabel(panel_label, fontsize=8.5, labelpad=6, color="#555555")

    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", color="#111111", pad=5)

    return subset

# ── Figure layout ─────────────────────────────────────────────────────────── #

fig = plt.figure(figsize=(10.0, 3.4), facecolor="white")

# 5 columns: [G1-Org | G1-Part | spacer | G9-Org | G9-Part]
# 7 columns: [G1-Org | inner-gap | G1-Part | section-gap | G9-Org | inner-gap | G9-Part]
gs = gridspec.GridSpec(1, 7,
    width_ratios=[0.13, 0.03, 0.28, 0.08, 0.13, 0.03, 0.28],
    wspace=0.0)

ax1 = fig.add_subplot(gs[0, 0])                  # Gen1OU – Organizers
ax_g1_inner = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[0, 2], sharey=ax1)      # Gen1OU – Participants
ax_gap = fig.add_subplot(gs[0, 3])
ax3 = fig.add_subplot(gs[0, 4], sharey=ax1)      # Gen9OU – Organizers
ax_g9_inner = fig.add_subplot(gs[0, 5])
ax4 = fig.add_subplot(gs[0, 6], sharey=ax1)      # Gen9OU – Participants

ax_gap.set_visible(False)

# Inner gaps: share y-axis and show only horizontal gridlines to bridge the gap
for _ax_inner in (ax_g1_inner, ax_g9_inner):
    _ax_inner.sharey(ax1)
    _ax_inner.set_facecolor("none")
    for sp in _ax_inner.spines.values():
        sp.set_visible(False)
    _ax_inner.tick_params(left=False, right=False, bottom=False,
                          labelleft=False, labelright=False, labelbottom=False)
    _ax_inner.grid(axis="y", color="#999999", alpha=0.35, lw=0.8, zorder=0, linestyle="--")
    _ax_inner.set_xticks([])

# Compute per-section y bounds so both panels in a section get the same range
def _section_ybounds(entries):
    vals = [e["whr"] for e in entries]
    pad  = max(40, (max(vals) - min(vals)) * 0.05)
    return min(vals) - pad, max(vals) + pad * 1.6

g1_y0, g1_y1 = _section_ybounds(gen1ou)
g9_y0, g9_y1 = _section_ybounds(gen9ou)
# Unify y range across both sections
g1_y0 = g9_y0 = min(g1_y0, g9_y0)
g1_y1 = g9_y1 = max(g1_y1, g9_y1)

GXE_XLABEL = "Deviation from GXE"

plot_panel(ax1, gen1ou, ORGANIZER_CATS,
           show_ylabel=True, show_yticks=True,
           title="Organizers", panel_label=GXE_XLABEL)
plot_panel(ax2, gen1ou, PARTICIPANT_CATS,
           show_ylabel=False, show_yticks=False,
           title="Participants", panel_label=GXE_XLABEL)
plot_panel(ax3, gen9ou, ORGANIZER_CATS,
           show_ylabel=False, show_yticks=True,
           title="Organizers", panel_label=GXE_XLABEL)
plot_panel(ax4, gen9ou, PARTICIPANT_CATS,
           show_ylabel=False, show_yticks=False,
           title="Participants", panel_label=GXE_XLABEL)

ax1.set_ylim(g1_y0, g1_y1)
ax3.set_ylim(g9_y0, g9_y1)
import matplotlib.ticker as mticker
ax1.yaxis.set_major_locator(mticker.MultipleLocator(100))


import math

def _nice_step(raw):
    """Round *raw* DOWN to the nearest 'nice' number (1,2,2.5,5 × 10^n)."""
    if raw <= 0:
        return 1
    exp = math.floor(math.log10(raw))
    base = 10 ** exp
    candidates = sorted([base * m for m in (1, 2, 2.5, 5, 10)])
    best = candidates[0]
    for c in candidates:
        if c <= raw:
            best = c
        else:
            break
    return int(best) if best == int(best) else best

ax1.set_xticks([-25, 0, 25])
ax2.set_xticks([-30, -15, 0, 15, 30])
ax3.set_xticks([-25, 0, 25])
ax4.set_xticks([-30, -15, 0, 15, 30])

def _add_zero_arrows():
    """Draw an upward triangle marker at the top of the x=0 line in each panel."""
    for ax in (ax1, ax2, ax3, ax4):
        lo, hi = ax.get_xlim()
        x_frac = (0 - lo) / (hi - lo)
        ax.plot(x_frac, 1.0, transform=ax.transAxes,
                marker='^', color='black', markersize=5,
                clip_on=False, zorder=10, linestyle='none')

def _add_axis_arrows():
    """Replace each panel's x-axis spine with a double-headed arrow spanning
    the full panel width.  Hiding the spine prevents any line from extending
    past the arrowheads.
    """
    AW = dict(arrowstyle="<|-|>", color="black", lw=0.8, mutation_scale=8)
    for ax in (ax1, ax2, ax3, ax4):
        # Tips stop 3% short of each edge so opposing arrowheads have a gap
        ax.annotate("",
            xy=(0.97, 0), xycoords="axes fraction",
            xytext=(0.03, 0), textcoords="axes fraction",
            arrowprops=AW, annotation_clip=False)


# Annotate the x=0 reference line

# ── Participant name labels ───────────────────────────────────────────────── #
# Per-label angle directions (0=right, 90=up, 180=left, 270=down)
G1_DIRS = {
    "PA-Agent": 90, "4thLesson": 110, "srsk-1729": 90,
    "ED-Testing": 0, "MetaHorns": 50, "Exp-05": 200,
    "Gradient": 290, "FoulPlay": 310, "SnTeam": 210,
    "GCOGS": 300, "Porygon2AI": 270, "hida": 180,
    "Puffer": 90, "VibePoking": 120,
}
G9_DIRS = {
    "FoulPlay": 190, "ED-Testing": 300, "Q": 15,
    "piploop": 95, "PA-Agent": 120, "MetaHorns": 320,
    "srsk-1729": 180, "Porygon2AI": 240, "GCOGS": 250,
    "hida": 0, "Hypercursed": 210, "PCL": 45,
    "August": 0, "VibePoking": 0, "INI": 105,
}

# All offset constants are in points (1 pt = 1/72 in, DPI-independent).
# Per-label (dx, dy) nudges in points applied on top of the angle offset.
G1_NUDGE = {
    "PA-Agent":  (0, -1.9),
    "MetaHorns": (0, -6.2),
    "srsk-1729": (0, -4.3),
}
G9_NUDGE = {
    "ED-Testing": (0, 1.9),
    "FoulPlay":   (1.9, 0),
}

STAR_DIST = 22.0   # pt: marker center → text anchor
DIAM_DIST = 17.5

STAR_MARKER_R = 3.4  # visual radius in pt
DIAM_MARKER_R = 2.4

ARROW_LEN = 2.9  # fixed arrow length in pt

def _label_participants(ax, entries, directions, nudges=None):
    subset = [e for e in entries if e["category"] in PARTICIPANT_CATS]
    if not subset:
        return

    texts = []
    for e in subset:
        angle = directions.get(e["name"], 270)
        is_star = e["category"] == "participant_qual"
        dist = STAR_DIST if is_star else DIAM_DIST
        rad = np.radians(angle)
        dx = dist * np.cos(rad)
        dy = dist * np.sin(rad)
        if nudges:
            ndx, ndy = nudges.get(e["name"], (0, 0))
            dx += ndx
            dy += ndy
        if 45 < angle < 135:
            ha, va = "center", "bottom"
        elif 225 < angle < 315:
            ha, va = "center", "top"
        elif 90 <= angle <= 270:
            ha, va = "right", "center"
        else:
            ha, va = "left", "center"
        marker_r = STAR_MARKER_R if is_star else DIAM_MARKER_R

        # Text label — no arrowprops; arrow drawn separately below.
        t = ax.annotate(
            e["name"], (e["resid"], e["whr"]),
            xytext=(dx, dy), textcoords="offset points",
            fontsize=7.75, color="#111111", va=va, ha=ha,
            fontweight="semibold", zorder=10,
            path_effects=[pe.withStroke(linewidth=3.0, foreground="white")],
        )
        texts.append((e, t, dx, dy))

        _arrow_len = 9.0   # visible shaft in pt
        _arrow_gap = 4.5   # gap: tail distance past arrowhead (pt)
        _vert_extra = 3.4  # extra gap for vertical arrows (pt)
        ux, uy = dx / dist, dy / dist
        gap = _arrow_gap + _vert_extra * abs(uy)
        tail_dx = ux * (marker_r + gap + _arrow_len)
        tail_dy = uy * (marker_r + gap + _arrow_len)
        ax.annotate(
            "", (e["resid"], e["whr"]),
            xytext=(tail_dx, tail_dy), textcoords="offset points",
            arrowprops=dict(
                arrowstyle="-|>",
                color="#666666",
                lw=0.7,
                shrinkA=0,
                shrinkB=marker_r + gap,
                mutation_scale=6,
            ),
            zorder=9,
        )

_label_participants(ax2, gen1ou, G1_DIRS, G1_NUDGE)
_label_participants(ax4, gen9ou, G9_DIRS, G9_NUDGE)

# ── Layout & section titles ───────────────────────────────────────────────── #
plt.subplots_adjust(left=0.09, right=0.97, bottom=0.22, top=0.85)

def _section_center(ax_l, ax_r):
    """Center accounting for y-axis labels that extend left of ax_l's data area."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_w = fig.get_window_extent(renderer).width
    # tight bbox includes tick labels and ylabel to the left of the axes
    x0_frac = ax_l.get_tightbbox(renderer).x0 / fig_w
    x1_frac = ax_r.get_position().x1
    return (x0_frac + x1_frac) / 2


# Centred under Gen1OU participants (ax2) x-axis label
_ax2_cx = (ax2.get_position().x0 + ax2.get_position().x1) / 2
_ax2_bot = ax2.get_position().y0
fig.text(_ax2_cx, _ax2_bot - 0.115,
         "\u2190 FH-BT lower than GXE predicts  |  FH-BT higher than GXE predicts \u2192",
         ha="center", va="top", fontsize=7.5, color="#666666", fontstyle="italic")

lc = _section_center(ax1, ax2)
fig.text(lc, 0.977, "Gen 1 OU", ha="center", va="top",
         fontsize=14, fontweight="bold", color="#111111")

rc = _section_center(ax3, ax4)
fig.text(rc, 0.977, "Gen 9 OU", ha="center", va="top",
         fontsize=14, fontweight="bold", color="#111111")



# ── Legend ────────────────────────────────────────────────────────────────── #
_mk = dict(markeredgecolor="white", markeredgewidth=0.7, color="none")
handles = [
    mlines.Line2D([], [], marker="o",  markersize=9,  markerfacecolor=RL_PUBLIC_COLOR,        label="Org. RL (Public)",     **_mk),
    mlines.Line2D([], [], marker="o",  markersize=9,  markerfacecolor=RL_HELDOUT_COLOR,       label="Org. RL (Held-Out)",   **_mk),
    mlines.Line2D([], [], marker="s",  markersize=9,  markerfacecolor=LLM_PC_COLOR,           label="Org. LLM (PokéChamp)", **_mk),
    mlines.Line2D([], [], marker="s",  markersize=9,  markerfacecolor=LLM_STD_COLOR,          label="Org. LLM (PokéAgent)",  **_mk),
    mlines.Line2D([], [], marker="^",  markersize=10, markerfacecolor=HEURISTIC_COLOR,        label="Org. Heuristic",       **_mk),
    mlines.Line2D([], [], marker="D",  markersize=8,  markerfacecolor=PARTICIPANT_COLOR,      label="Participants",         **_mk),
    mlines.Line2D([], [], marker="*",  markersize=13, markerfacecolor=PARTICIPANT_QUAL_COLOR, label="Winning Participants", **_mk),
]
fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=9,
           frameon=False, bbox_to_anchor=(0.5, -0.04),
           bbox_transform=fig.transFigure,
           handletextpad=0.4, columnspacing=1.2,
           handlelength=1.0, borderpad=0.2, labelspacing=0.2)

# ── Save ──────────────────────────────────────────────────────────────────── #
_add_axis_arrows()
_add_zero_arrows()
for ext in ("pdf", "png"):
    plt.savefig(f"{OUT_DIR}/track1_qualifying_strip.{ext}", dpi=300,
                bbox_inches="tight", facecolor="white", edgecolor="none")
print(f"Saved to {OUT_DIR}/track1_qualifying_strip.{{pdf,png}}")
plt.close(fig)
