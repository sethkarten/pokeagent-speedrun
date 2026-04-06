"""
RL vs LLM Agent Performance Across Game Benchmarks

This script generates Figure 1 for the PokéAgent Challenge paper.
Uses split axes: bounded (log scale) and unbounded (∞).

All data points have citations below.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse, Polygon
from matplotlib.lines import Line2D
import numpy as np
from pathlib import Path
from scipy.spatial import ConvexHull

# =============================================================================
# NEURIPS STYLE CONFIGURATION
# =============================================================================
# NeurIPS typically uses 5.5in single-column, ~8.5in double-column
# Using full-width figure for this complex visualization

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'patch.linewidth': 1.2,
    'text.usetex': False,  # Set True if LaTeX is available
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# =============================================================================
# PROFESSIONAL COLOR PALETTE (Colorblind-friendly)
# =============================================================================
# Using a palette based on Okabe-Ito / ColorBrewer recommendations
COLORS = {
    'rl': '#6BAED6',       # Blue - soft pastel
    'llm': '#FDAE6B',      # Orange - soft pastel
    'rl_text': '#084594',  # Blue - dark for text
    'llm_text': '#8C2900', # Orange - dark for text
    'rl_text_muted': '#5A7FA8',   # Lighter blue for non-Pokemon labels
    'llm_text_muted': '#A06743',  # Lighter orange for non-Pokemon labels
    'heuristic': '#7A7A7A', # Gray for heuristic markers
    'heuristic_text': '#8E8E8E',  # Lighter gray for heuristic labels
    'human': '#888888',     # Gray - neutral reference
    'region_single': '#44AA99',    # Teal-green
    'region_zerosum': '#332288',   # Indigo/dark blue
    'region_partial': '#E754C6',   # Hot pink/magenta for partially observable
    'region_rlenv': '#DDCC77',     # Yellow-tan
    'region_openended': '#117733', # Forest green (was wine)
    'pokemon_rl_text': '#2B73BC',   # Brighter blue for Pokémon labels
    'pokemon_llm_text': '#C85A1A',  # Brighter orange for Pokémon labels
}

# =============================================================================
# DATA WITH CITATIONS
# =============================================================================
# Format: (benchmark, method, agent_type, state_space_exponent, performance)
# state_space_exponent: the power of 10 (e.g., 8 means 10^8)
# performance: % of human expert (100% = human level)

# BOUNDED STATE SPACE
bounded_data = [
    # Zero-Sum (Perfect Information)
    ("Backgammon", "TD-Gammon", "RL", 20, 100, "Tesauro, 1995"),
    ("Chess", "AlphaZero", "RL", 44, 125, "Silver et al., 2018"),
    ("Chess", "GPT-4o", "LLM", 44, 62, "Acher, 2024"),
    ("Go", "AlphaGo Zero", "RL", 170, 110, "Silver et al., 2017"),
    ("Go", "LLMs", "LLM", 170, 5, "Meta, 2024"),

    # Partially Observable Zero-Sum
    ("Poker", "Pluribus", "RL", 160, 115, "Brown & Sandholm, 2019"),
    ("Poker", "GPT-4", "LLM", 160, 50, "Wu et al., 2025"),
    ("Pokémon\nBattling", "Metamon", "RL", 400, 90, "Grigsby et al., 2025"),
    ("Pokémon\nBattling", "PokéChamp", "LLM", 400, 68, "Karten et al., 2025"),

]

# UNBOUNDED STATE SPACE (time-series, open-ended)
unbounded_data = [
    ("Dota 2", "OpenAI Five", "RL", 105, "OpenAI, 2019"),
    ("StarCraft II", "AlphaStar", "RL", 100, "Vinyals et al., 2019"),
    ("StarCraft II", "LLM Agent", "LLM", 50, "Ma et al., 2024"),
    ("Minecraft", "VPT", "RL", 35, "Baker et al., 2022"),
    ("Minecraft", "Voyager", "LLM", 70, "Wang et al., 2023"),
    ("Pokémon RPG", "PokeAgent", "LLM", 15, "Karten et al., 2025"),
]

# HEURISTIC CEILINGS per game
heuristic_bounded = [
    (20, 80, "Backgammon", "Tesauro, 1995"),
    (44, 105, "Chess", "Stockfish"),
    (170, 35, "Go", "Crazy Stone"),
    (160, 40, "Poker", "Billings et al., 2002"),
]

heuristic_unbounded = [
    (20, "Dota 2", "OpenAI, 2017"),
    (25, "StarCraft II", "Blizzard AI"),
    (30, "Minecraft", "Scripted Bot"),
]

# =============================================================================
# LABEL POSITION OFFSETS - (X_offset, Y_offset, alignment)
# X: positive=right, negative=left  |  Y: positive=up, negative=down
# =============================================================================

# Bounded RL/LLM agent labels
LABEL_OFFSETS_BOUNDED = {
    # (Game, Method):                   (X,   Y,   align)
    ("Backgammon", "TD-Gammon"):        (8,  -7, 'left'),   # RL
    ("Chess", "AlphaZero"):             (8,  2,   'left'),   # RL
    ("Chess", "GPT-4o"):                (10, -2,  'left'),   # LLM
    ("Go", "AlphaGo Zero"):             (10,  0,   'left'),   # RL
    ("Go", "LLMs"):                     (-10,  0, 'right'),   # LLM
    ("Poker", "Pluribus"):              (10,  0,   'left'),   # RL
    ("Poker", "GPT-4"):                 (10,  0, 'left'),   # LLM
    ("Pokémon\nBattling", "Metamon"):   (58, -16,   'right'),  # RL
    ("Pokémon\nBattling", "PokéChamp"): (-10, 2, 'right'),  # LLM
    ("TextArena", "LLMs"):              (10,  0,   'left'),   # LLM
    ("Prime Intellect\nEnv Hub", "LLMs"): (8, 0,  'left'),   # LLM
}

# Bounded heuristic labels
LABEL_OFFSETS_HEURISTIC_BOUNDED = {
    # Game:           (X,   Y,   align)
    "Backgammon":     (-10,  9,  'left'),  # Tesauro, 1995
    "Chess":          (0,  6,   'center'),  # Stockfish
    "Go":             (-8,  -6,  'right'),  # Crazy Stone
    "Poker":          (-8,  0,  'right'),  # Billings, 2002
}

# Unbounded RL/LLM agent labels
LABEL_OFFSETS_UNBOUNDED = {
    # (Game, Method):                (X,   Y,   align)
    ("Dota 2", "OpenAI Five"):       (10,  0,   'left'),   # RL
    ("StarCraft II", "AlphaStar"):   (10,  -8, 'left'),   # RL
    ("StarCraft II", "LLM Agent"):   (10,  0,   'left'),   # LLM
    ("Minecraft", "VPT"):            (10,  0,  'left'),   # RL
    ("Minecraft", "Voyager"):        (10,  0, 'left'),   # LLM
    ("Pokémon RPG", "PokeAgent"):    (10,  0, 'left'),   # LLM
}

# Unbounded heuristic labels
LABEL_OFFSETS_HEURISTIC_UNBOUNDED = {
    # Game:           (X,   Y,   align)
    "Dota 2":         (-8,  0,   'right'),  # OpenAI, 2017
    "StarCraft II":   (-8,  0,   'right'),  # Blizzard AI
    "Minecraft":      (-8,  0,   'right'),  # Scripted Bot
}

# Region label positions (X, Y)
REGION_LABEL_POSITIONS = {
    "Single-Agent":                 (4.5,  118),
    "Zero-Sum":                     (18,   128),
    "RL Env":                       (8,    58),
    "Partially Observable Zero-Sum": (620, 94),
    "Open-Ended":                   (1.7,  60),
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def format_citation(citation):
    """Format citation for figure labels: 'Author et al., Year' -> 'Author+ Year' or 'Year'"""
    import re
    # Extract year (4-digit number)
    year_match = re.search(r'(\d{4})', citation)
    year = year_match.group(1) if year_match else ""

    # Extract first author
    if ',' in citation:
        first_part = citation.split(',')[0].strip()
        # Get first author name (before "et al." or "&")
        if ' et al.' in first_part:
            author = first_part.replace(' et al.', '').strip()
        elif ' & ' in first_part:
            author = first_part.split(' & ')[0].strip()
        else:
            author = first_part
        return f"{author}, {year}"
    else:
        # Single word citations like "OpenAI, 2019" or "Stockfish"
        return citation

# =============================================================================
# PLOTTING
# =============================================================================

def create_figure():
    import matplotlib.patheffects as pe
    # NeurIPS full-width figure: approximately 7 inches wide
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(5.5, 3.5),
        sharey=True,
        gridspec_kw={'width_ratios': [3.0, 1.2], 'wspace': 0.05}
    )

    # Extract colors
    rl_color = COLORS['rl']
    llm_color = COLORS['llm']
    heuristic_color = COLORS['heuristic']

    # Marker sizes - slightly smaller for cleaner look
    marker_size = 80
    marker_edge = 1.0

    # ============ LEFT PLOT: Bounded State Space ============
    bounded_rl = [(d[0], d[1], d[3], d[4], d[5]) for d in bounded_data if d[2] == "RL"]
    bounded_llm = [(d[0], d[1], d[3], d[4], d[5]) for d in bounded_data if d[2] == "LLM"]
    connector_style = dict(color='#7F7F7F', linestyle='--', linewidth=0.8, alpha=0.28, zorder=-10)

    partial_obs_games = {'Pokémon\nBattling', 'Poker'}
    partial_edge_lw = 1.58
    partial_edge_alpha = 0.75
    partial_edge_color = (*plt.matplotlib.colors.to_rgb(COLORS['region_partial']), partial_edge_alpha)
    partial_glow_alpha = 0.18
    partial_glow_color = (*plt.matplotlib.colors.to_rgb(COLORS['region_partial']), partial_glow_alpha)
    pokemon_label_fs = 6.5
    feat_size = marker_size * 2.2

    def scatter_points(ax, data, color, marker):
        for benchmark, method, x, y, citation in data:
            is_partial = benchmark in partial_obs_games
            ec = partial_edge_color if is_partial else 'white'
            lw = partial_edge_lw if is_partial else marker_edge
            is_featured = benchmark == 'Pokémon\nBattling'
            sz = feat_size if is_featured else marker_size
            if is_partial:
                # Neon-like halo behind the crisp partially-observable border.
                ax.scatter([x], [y], facecolors='none', edgecolors=partial_glow_color,
                           s=sz, zorder=4.6, linewidths=3.2, marker=marker)
            sc = ax.scatter([x], [y], c=color, s=sz, zorder=5, edgecolors=ec, linewidths=lw, marker=marker)
            if is_featured:
                sc.set_path_effects([
                    pe.withStroke(linewidth=3.2, foreground=(1, 1, 1, 0.35)),
                    pe.withSimplePatchShadow(offset=(1.65, -1.65), alpha=0.3, shadow_rgbFace=(0, 0, 0)),
                    pe.Normal(),
                ])

    scatter_points(ax1, bounded_rl, rl_color, 'o')
    scatter_points(ax1, bounded_llm, llm_color, 's')

    # Dashed vertical connectors between RL/LLM results for same bounded game.
    bounded_rl_map = {game: (x, y) for game, _method, x, y, _citation in bounded_rl}
    bounded_llm_map = {game: (x, y) for game, _method, x, y, _citation in bounded_llm}
    for game in sorted(set(bounded_rl_map) & set(bounded_llm_map)):
        x_rl, y_rl = bounded_rl_map[game]
        x_llm, y_llm = bounded_llm_map[game]
        if x_rl == x_llm:
            ax1.plot([x_rl, x_rl], [min(y_rl, y_llm), max(y_rl, y_llm)], **connector_style)

    # Human expert line
    ax1.axhline(y=100, color=COLORS['human'], linestyle='--', linewidth=1.5, alpha=0.7, zorder=2).set_clip_on(False)
    ax2.axhline(y=100, color=COLORS['human'], linestyle='--', linewidth=1.5, alpha=0.7, zorder=2)
    ax1.set_zorder(1)
    ax2.set_zorder(2)


    # Labels for bounded points - uses LABEL_OFFSETS_BOUNDED from top of file
    for benchmark, method, x, y, citation in bounded_rl + bounded_llm:
        key = (benchmark, method)
        ox, oy, ha = LABEL_OFFSETS_BOUNDED.get(key, (10, 4, 'left'))
        if ha in ('left', 'right'):
            oy -= 2

        game_short = benchmark.replace('\n', ' ')
        short_cite = format_citation(citation)
        label = f"{game_short} ({short_cite})"
        is_rl = (benchmark, method, x, y, citation) in bounded_rl
        is_pokemon = benchmark == 'Pokémon\nBattling'
        if is_pokemon:
            color = COLORS['pokemon_rl_text'] if is_rl else COLORS['pokemon_llm_text']
        else:
            color = COLORS['rl_text_muted'] if is_rl else COLORS['llm_text_muted']
        is_poke_showdown = benchmark == 'Pokémon\nBattling'
        fs = pokemon_label_fs if is_poke_showdown else 6
        alpha = 1.0
        fw = 'bold' if is_poke_showdown else 'normal'
        # Pokémon Battling (Metamon) label is added as a figure-level artist
        # after layout is frozen to avoid ax2's background covering it.
        if is_poke_showdown and method == 'Metamon':
            continue
        ann = ax1.annotate(label, (x, y), textcoords="offset points", xytext=(ox, oy),
                    fontsize=fs, color=color, ha=ha, alpha=alpha, fontweight=fw)
        ann.set_zorder(20)
        if is_poke_showdown:
            ann.set_clip_on(False)
            ann.set_zorder(9999)

    # Heuristic markers - uses LABEL_OFFSETS_HEURISTIC_BOUNDED from top of file
    partial_obs_heur_bounded = {'Poker'}
    for x, y, game, citation in heuristic_bounded:
        is_partial = game in partial_obs_heur_bounded
        if is_partial:
            ax1.scatter([x], [y], facecolors='none', edgecolors=partial_glow_color,
                        s=34, zorder=3.8, linewidths=3.2, marker='D')
        hec = partial_edge_color if is_partial else heuristic_color
        hlw = partial_edge_lw if is_partial else 1.0
        ax1.scatter([x], [y], marker='D', facecolors=heuristic_color, edgecolors=hec,
                    s=34, linewidths=hlw, zorder=4)
        ox, oy, ha = LABEL_OFFSETS_HEURISTIC_BOUNDED.get(game, (-8, 0, 'right'))
        if ha in ('left', 'right'):
            oy -= 2
        short_cite = format_citation(citation) if ',' in citation else citation
        label = f"{game} ({short_cite})"
        ann = ax1.annotate(label, (x, y), textcoords="offset points", xytext=(ox, oy),
                    fontsize=6, color=COLORS['heuristic_text'], ha=ha, alpha=1.0)
        ann.set_zorder(20)

    # Axis settings
    ax1.set_xscale('log')
    ax1.set_xlabel('', fontsize=9)
    ax1.set_ylabel('Human-Expert-Normalized Performance', fontsize=9, fontweight='medium')
    ax1.set_xlim(15, 800)
    ax1.set_ylim(-5, 135)

    # Clean x-ticks
    ax1.set_xticks([100])
    ax1.set_xticklabels([r'$10^{100}$'], fontsize=9)

    # Grid
    ax1.grid(True, alpha=0.25, linestyle='-', linewidth=0.4, zorder=0, axis='y')
    ax1.set_title('', fontsize=12, fontstyle='italic', pad=2)

    # ============ RIGHT PLOT: Unbounded State Space ============
    unbounded_rl = [(d[0], d[1], d[3], d[4]) for d in unbounded_data if d[2] == "RL"]
    unbounded_llm = [(d[0], d[1], d[3], d[4]) for d in unbounded_data if d[2] == "LLM"]

    x_pos = 1.0

    # Plot points — partial obs games (Dota 2, StarCraft II) get red edges
    partial_obs_unb = {'Dota 2', 'StarCraft II', 'Minecraft', 'Pokémon RPG'}
    for benchmark, method, y, citation in unbounded_rl:
        is_partial = benchmark in partial_obs_unb
        ec = partial_edge_color if is_partial else 'white'
        lw = partial_edge_lw if is_partial else marker_edge
        if is_partial:
            ax2.scatter([x_pos], [y], facecolors='none', edgecolors=partial_glow_color,
                        s=marker_size, zorder=4.6, linewidths=3.2, marker='o')
        ax2.scatter([x_pos], [y], c=rl_color, s=marker_size, zorder=5,
                    edgecolors=ec, linewidths=lw, marker='o')
    for benchmark, method, y, citation in unbounded_llm:
        is_partial = benchmark in partial_obs_unb
        ec = partial_edge_color if is_partial else 'white'
        lw = partial_edge_lw if is_partial else marker_edge
        is_featured = benchmark == 'Pokémon RPG'
        sz = feat_size if is_featured else marker_size
        if is_partial:
            ax2.scatter([x_pos], [y], facecolors='none', edgecolors=partial_glow_color,
                        s=sz, zorder=4.6, linewidths=3.2, marker='s')
        sc = ax2.scatter([x_pos], [y], c=llm_color, s=sz, zorder=5,
                         edgecolors=ec, linewidths=lw, marker='s')
        if is_featured:
            sc.set_path_effects([
                pe.withSimplePatchShadow(offset=(1.65, -1.65), alpha=0.3, shadow_rgbFace=(0, 0, 0)),
                pe.Normal(),
            ])

    # Dashed vertical connectors between RL/LLM results for same unbounded game.
    unbounded_rl_map = {game: y for game, _method, y, _citation in unbounded_rl}
    unbounded_llm_map = {game: y for game, _method, y, _citation in unbounded_llm}
    for game in sorted(set(unbounded_rl_map) & set(unbounded_llm_map)):
        y_rl = unbounded_rl_map[game]
        y_llm = unbounded_llm_map[game]
        ax2.plot([x_pos, x_pos], [min(y_rl, y_llm), max(y_rl, y_llm)], **connector_style)

    # Human expert line
    # human expert line drawn figure-level for continuity

    # Labels for unbounded - uses LABEL_OFFSETS_UNBOUNDED from top of file
    for benchmark, method, y, citation in unbounded_rl + unbounded_llm:
        ox, oy, ha = LABEL_OFFSETS_UNBOUNDED.get((benchmark, method), (12, 3, 'left'))
        if ha in ('left', 'right'):
            oy -= 2
        short_cite = format_citation(citation)
        label = f"{benchmark} ({short_cite})"
        is_rl = (benchmark, method, y, citation) in unbounded_rl
        is_poke_rpg = benchmark == 'Pokémon RPG'
        if is_poke_rpg:
            color = COLORS['pokemon_rl_text'] if is_rl else COLORS['pokemon_llm_text']
        else:
            color = COLORS['rl_text_muted'] if is_rl else COLORS['llm_text_muted']
        alpha = 1.0
        fs = pokemon_label_fs if is_poke_rpg else 6
        fw = 'bold' if is_poke_rpg else 'normal'
        # Pokémon RPG label added as figure-level artist after layout freeze (below)
        if is_poke_rpg:
            continue
        ann = ax2.annotate(label, (x_pos, y), textcoords="offset points", xytext=(ox, oy),
                    fontsize=fs, color=color, ha=ha, alpha=alpha, fontweight=fw)
        ann.set_zorder(20)
        ann.set_clip_on(False)

    # Heuristic markers for unbounded - uses LABEL_OFFSETS_HEURISTIC_UNBOUNDED from top of file
    for y, game, citation in heuristic_unbounded:
        is_partial = game in partial_obs_unb
        if is_partial:
            ax2.scatter([x_pos], [y], facecolors='none', edgecolors=partial_glow_color,
                        s=34, zorder=3.8, linewidths=3.2, marker='D')
        hec = partial_edge_color if is_partial else heuristic_color
        hlw = partial_edge_lw if is_partial else 1.0
        ax2.scatter([x_pos], [y], marker='D', facecolors=heuristic_color, edgecolors=hec,
                    s=34, linewidths=hlw, zorder=4)
        ox, oy, ha = LABEL_OFFSETS_HEURISTIC_UNBOUNDED.get(game, (-8, 0, 'right'))
        if ha in ('left', 'right'):
            oy -= 2
        short_cite = format_citation(citation) if ',' in citation else citation
        label = f"{game} ({short_cite})"
        ann = ax2.annotate(label, (x_pos, y), textcoords="offset points", xytext=(ox, oy),
                    fontsize=6, color=COLORS['heuristic_text'], ha=ha, alpha=1.0)
        ann.set_zorder(20)
        ann.set_clip_on(False)

    # Axis settings
    ax2.set_xlim(0.2, 5.5)
    ax2.set_xticks([1])
    ax2.set_xticklabels([r'$\infty$'], fontsize=14)
    ax2.grid(True, alpha=0.25, linestyle='-', linewidth=0.4, axis='y', zorder=0)
    ax2.spines['left'].set_visible(False)
    ax2.tick_params(left=False)
    ax2.set_title('', fontsize=12, fontstyle='italic', pad=2)


    # ============ LEGEND ============
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=rl_color,
               markersize=8, label='RL', markeredgecolor='white', markeredgewidth=0.5),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=llm_color,
               markersize=8, label='LLM', markeredgecolor='white', markeredgewidth=0.5),
        Line2D([0], [0], marker='D', color=heuristic_color, linestyle='None',
               markersize=7, markeredgewidth=1.0, label='Heuristic'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
               markersize=10, markeredgecolor=partial_edge_color,
               markeredgewidth=partial_edge_lw, linestyle='None',
               label='Partially Observable'),
    ]

    fig.legend(handles=legend_elements, loc='lower center',
               bbox_to_anchor=(0.50, 0.03), frameon=False, fontsize=8, ncol=4,
               borderaxespad=0.2, handletextpad=0.4, columnspacing=1.1, labelspacing=0.3)

    # Main title
    # Title removed

    plt.subplots_adjust(left=0.12, right=0.91, bottom=0.25, top=0.95, wspace=0.05)

    # Pokémon Battling (Metamon/Grigsby) and Pokémon RPG (Karten) labels —
    # added at figure level so neither ax2's background nor the canvas right
    # edge can clip them.
    _poke_xy = ax1.transData.transform((400, 92))
    _poke_xf, _poke_yf = fig.transFigure.inverted().transform(_poke_xy)
    fig.text(_poke_xf, _poke_yf - 0.056,
             'Pokémon Battling (Grigsby et al., 2025)',
             ha='center', va='top', fontsize=pokemon_label_fs, fontweight='bold',
             color=COLORS['pokemon_rl_text'], alpha=1.0, zorder=9999)

    _rpg_xy = ax2.transData.transform((x_pos, 15))
    _rpg_xf, _rpg_yf = fig.transFigure.inverted().transform(_rpg_xy)
    fig.text(_rpg_xf - 0.039, _rpg_yf - 0.010,
             'Pokémon RPG (Karten et al., 2025)',
             ha='right', va='center', fontsize=pokemon_label_fs, fontweight='bold',
             color=COLORS['pokemon_llm_text'], alpha=1.0, zorder=9999)

    # Bridge the tiny seam gap between axes for the y=100 dashed line.
    from matplotlib.lines import Line2D as _Line2D
    ax1_pos = ax1.get_position()
    ax2_pos = ax2.get_position()
    y100_px = ax1.transData.transform((1, 100))[1]
    y100_fig = fig.transFigure.inverted().transform((0, y100_px))[1]
    fig.add_artist(_Line2D(
        [ax1_pos.x1, ax2_pos.x0], [y100_fig, y100_fig],
        transform=fig.transFigure, color=COLORS['human'],
        linestyle='--', linewidth=1.5, alpha=0.7, zorder=3, clip_on=False))

    # Break marks: two diagonal slashes touching the x-axis spines at the gap
    # ax1 spine runs to its right edge; ax2 spine starts at its left edge.
    # Slashes are drawn right at those edges.
    spine_y = ax1_pos.y0
    dx, dy = 0.004, 0.013
    for x_center in [ax1_pos.x1 + 0.008, ax2_pos.x0]:
        fig.add_artist(_Line2D(
            [x_center - dx, x_center + dx],
            [spine_y - dy, spine_y + dy],
            transform=fig.transFigure, color='k', linewidth=1.1, clip_on=False, zorder=10))

    fig.text(0.55, 0.14, 'State Space Complexity', ha='center', fontsize=9, fontweight='medium')

    return fig


def print_citations():
    """Print all data with citations."""
    print("\n" + "=" * 80)
    print("BOUNDED STATE SPACE DATA")
    print("=" * 80)
    for benchmark, method, agent_type, exp, perf, citation in bounded_data:
        print(f"\n{benchmark} ({method}): {perf}% | State: 10^{exp} | Type: {agent_type}")
        print(f"  Citation: {citation}")

    print("\n" + "=" * 80)
    print("UNBOUNDED STATE SPACE DATA")
    print("=" * 80)
    for benchmark, method, agent_type, perf, citation in unbounded_data:
        print(f"\n{benchmark} ({method}): {perf}% | Type: {agent_type}")
        print(f"  Citation: {citation}")


if __name__ == "__main__":
    print_citations()

    fig = create_figure()

    output_path = Path(__file__).parent.parent / "rl_vs_llm_benchmarks_v3.png"
    fig.savefig(output_path, dpi=300, facecolor='white')
    print(f"\n\nFigure saved to: {output_path}")

    # Also save PDF for LaTeX
    pdf_path = output_path.with_suffix('.pdf')
    fig.savefig(pdf_path, facecolor='white')
    print(f"PDF saved to: {pdf_path}")

    plt.close(fig)
