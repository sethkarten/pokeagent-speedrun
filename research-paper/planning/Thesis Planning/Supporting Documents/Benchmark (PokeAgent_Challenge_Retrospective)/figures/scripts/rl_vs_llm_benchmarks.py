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
    'rl': '#0077BB',       # Blue - strong, professional
    'llm': '#EE7733',      # Orange - complementary, distinct
    'heuristic': '#009988', # Teal - distinct from both
    'human': '#888888',     # Gray - neutral reference
    'region_single': '#44AA99',    # Teal-green
    'region_zerosum': '#332288',   # Indigo/dark blue
    'region_partial': '#CC3311',   # Red-orange (was purple)
    'region_rlenv': '#DDCC77',     # Yellow-tan
    'region_openended': '#117733', # Forest green (was wine)
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
    ("Chess", "GPT-4o", "LLM", 44, 62, "Acher, 2023"),
    ("Go", "AlphaGo Zero", "RL", 170, 110, "Silver et al., 2017"),
    ("Go", "LLMs", "LLM", 170, 5, "Ma et al., 2026"),

    # Partially Observable Zero-Sum
    ("Poker", "Pluribus", "RL", 160, 115, "Brown & Sandholm, 2019"),
    ("Poker", "LLMs", "LLM", 160, 50, "Zhuang et al., 2025"),
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
    ("Backgammon", "TD-Gammon"):        (10,  -8, 'left'),   # RL
    ("Chess", "AlphaZero"):             (8,  2,   'left'),   # RL
    ("Chess", "GPT-4o"):                (10,  0,  'left'),   # LLM
    ("Go", "AlphaGo Zero"):             (10,  0,   'left'),   # RL
    ("Go", "LLMs"):                     (10,  0, 'left'),   # LLM
    ("Poker", "Pluribus"):              (10,  0,   'left'),   # RL
    ("Poker", "LLMs"):                   (10,  0, 'left'),   # LLM
    ("Pokémon\nBattling", "Metamon"):   (-10, 0,   'right'),  # RL
    ("Pokémon\nBattling", "PokéChamp"): (-10, 0, 'right'),  # LLM
}

# Bounded heuristic labels
LABEL_OFFSETS_HEURISTIC_BOUNDED = {
    # Game:           (X,   Y,   align)
    "Backgammon":     (8,  0,  'left'),  # Tesauro, 1995
    "Chess":          (-8,  0,   'right'),  # Stockfish
    "Go":             (-8,  0,  'right'),  # Crazy Stone
    "Poker":          (-8,  0,  'right'),  # Billings, 2002
    "Pokémon Battling": (-8,  0,   'right'),  # Foul Play
}

# Unbounded RL/LLM agent labels
LABEL_OFFSETS_UNBOUNDED = {
    # (Game, Method):                (X,   Y,   align)
    ("Dota 2", "OpenAI Five"):       (-10,  0,   'right'),   # RL
    ("StarCraft II", "AlphaStar"):   (0,  -10, 'center'),   # RL
    ("StarCraft II", "LLM Agent"):   (0,  -10,   'center'),   # LLM
    ("Minecraft", "VPT"):            (-10,  0,  'right'),   # RL
    ("Minecraft", "Voyager"):        (0,  -10, 'center'),   # LLM
    ("Pokémon RPG", "PokeAgent"):    (-10,  0, 'right'),   # LLM
}

# Unbounded heuristic labels
LABEL_OFFSETS_HEURISTIC_UNBOUNDED = {
    # Game:           (X,   Y,   align)
    "Dota 2":         (-10,  0,   'right'),  # OpenAI, 2017
    "StarCraft II":   (-10,  0,   'right'),  # Blizzard AI
    "Minecraft":      (-10,  0,   'right'),  # Scripted Bot
}

# Region label positions (X, Y)
REGION_LABEL_POSITIONS = {
    "Single-Agent":                 (4.5,  118),
    "Zero-Sum":                     (18,   128),
    "RL Env":                       (8,    58),
    "Partially Observable Zero-Sum": (450, 94),
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
    # NeurIPS full-width figure: approximately 7 inches wide
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(4.0, 4.0),
        sharey=True,
        gridspec_kw={'width_ratios': [3.0, 1.2], 'wspace': 0.05}
    )

    # Extract colors
    rl_color = COLORS['rl']
    llm_color = COLORS['llm']
    heuristic_color = COLORS['heuristic']

    # Marker sizes - slightly smaller for cleaner look
    marker_size = 80
    marker_size_pokemon = 160  # 2x for emphasis
    marker_edge = 1.0
    marker_edge_pokemon = 1.5

    # Helper to detect Pokémon data points
    def is_pokemon(benchmark):
        return 'poké' in benchmark.lower() or 'pokémon' in benchmark.lower()

    # ============ LEFT PLOT: Bounded State Space ============
    bounded_rl = [(d[0], d[1], d[3], d[4], d[5]) for d in bounded_data if d[2] == "RL"]
    bounded_llm = [(d[0], d[1], d[3], d[4], d[5]) for d in bounded_data if d[2] == "LLM"]

    # Plot RL points (non-Pokémon)
    rl_x = [d[2] for d in bounded_rl if not is_pokemon(d[0])]
    rl_y = [d[3] for d in bounded_rl if not is_pokemon(d[0])]
    ax1.scatter(rl_x, rl_y, c=rl_color, s=marker_size, zorder=5,
                edgecolors='white', linewidths=marker_edge, marker='o')

    # Plot LLM points (non-Pokémon)
    llm_x = [d[2] for d in bounded_llm if not is_pokemon(d[0])]
    llm_y = [d[3] for d in bounded_llm if not is_pokemon(d[0])]
    ax1.scatter(llm_x, llm_y, c=llm_color, s=marker_size, marker='s', zorder=5,
                edgecolors='white', linewidths=marker_edge)

    # Plot Pokémon RL points (emphasized: halo + larger marker + dark edge)
    poke_rl_x = [d[2] for d in bounded_rl if is_pokemon(d[0])]
    poke_rl_y = [d[3] for d in bounded_rl if is_pokemon(d[0])]
    ax1.scatter(poke_rl_x, poke_rl_y, c=rl_color, s=marker_size_pokemon * 3,
                marker='o', alpha=0.15, zorder=4, edgecolors='none')  # halo
    ax1.scatter(poke_rl_x, poke_rl_y, c=rl_color, s=marker_size_pokemon, zorder=6,
                edgecolors='black', linewidths=marker_edge_pokemon, marker='o')

    # Plot Pokémon LLM points (emphasized)
    poke_llm_x = [d[2] for d in bounded_llm if is_pokemon(d[0])]
    poke_llm_y = [d[3] for d in bounded_llm if is_pokemon(d[0])]
    ax1.scatter(poke_llm_x, poke_llm_y, c=llm_color, s=marker_size_pokemon * 3,
                marker='s', alpha=0.15, zorder=4, edgecolors='none')  # halo
    ax1.scatter(poke_llm_x, poke_llm_y, c=llm_color, s=marker_size_pokemon, marker='s', zorder=6,
                edgecolors='black', linewidths=marker_edge_pokemon)

    # Human expert line
    ax1.axhline(y=100, color=COLORS['human'], linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)
    ax1.text(100, 102, 'Human Expert', fontsize=8, color=COLORS['human'], style='italic')

    # Labels for bounded points - uses LABEL_OFFSETS_BOUNDED from top of file
    for benchmark, method, x, y, citation in bounded_rl + bounded_llm:
        key = (benchmark, method)
        ox, oy, ha = LABEL_OFFSETS_BOUNDED.get(key, (10, 4, 'left'))

        game_short = benchmark.replace('\n', ' ')
        short_cite = format_citation(citation)
        label = f"{game_short} ({short_cite})"
        color = rl_color if (benchmark, method, x, y, citation) in bounded_rl else llm_color
        poke = is_pokemon(benchmark)
        ax1.annotate(label, (x, y), textcoords="offset points", xytext=(ox, oy),
                    fontsize=8 if poke else 7, color=color, ha=ha,
                    fontweight='bold' if poke else 'normal')

    # Heuristic markers - uses LABEL_OFFSETS_HEURISTIC_BOUNDED from top of file
    for x, y, game, citation in heuristic_bounded:
        poke = is_pokemon(game)
        ax1.scatter(x, y, marker='+', c=heuristic_color,
                    s=120 if poke else 80,
                    linewidths=2.5 if poke else 1.8, zorder=4)
        ox, oy, ha = LABEL_OFFSETS_HEURISTIC_BOUNDED.get(game, (-8, 0, 'right'))
        short_cite = format_citation(citation) if ',' in citation else citation
        label = f"{game} ({short_cite})"
        ax1.annotate(label, (x, y), textcoords="offset points", xytext=(ox, oy),
                    fontsize=7 if poke else 6, color=heuristic_color, ha=ha,
                    alpha=1.0 if poke else 0.9,
                    fontweight='bold' if poke else 'normal')

    # Axis settings
    ax1.set_xscale('log')
    ax1.set_xlabel('State Space Complexity', fontsize=11, fontweight='medium')
    ax1.set_ylabel('Performance (% of Human Expert)', fontsize=11, fontweight='medium')
    ax1.set_xlim(10, 1000)
    ax1.set_ylim(-5, 135)

    # Clean x-ticks
    ax1.set_xticks([10, 100, 1000])
    ax1.set_xticklabels([r'$10^{10}$', r'$10^{100}$', r'$10^{1000}$'], fontsize=9)

    # Grid
    ax1.grid(True, alpha=0.25, linestyle='-', linewidth=0.4, zorder=0)
    # Title removed; provided by LaTeX caption

    # ============ RIGHT PLOT: Unbounded State Space ============
    unbounded_rl = [(d[0], d[1], d[3], d[4]) for d in unbounded_data if d[2] == "RL"]
    unbounded_llm = [(d[0], d[1], d[3], d[4]) for d in unbounded_data if d[2] == "LLM"]

    x_pos = 1.0

    # Plot points (non-Pokémon)
    rl_y_unb = [d[2] for d in unbounded_rl if not is_pokemon(d[0])]
    ax2.scatter([x_pos] * len(rl_y_unb), rl_y_unb, c=rl_color, s=marker_size, zorder=5,
                edgecolors='white', linewidths=marker_edge, marker='o')

    llm_y_unb = [d[2] for d in unbounded_llm if not is_pokemon(d[0])]
    ax2.scatter([x_pos] * len(llm_y_unb), llm_y_unb, c=llm_color, s=marker_size, marker='s', zorder=5,
                edgecolors='white', linewidths=marker_edge)

    # Plot Pokémon RPG points (emphasized)
    poke_rl_y_unb = [d[2] for d in unbounded_rl if is_pokemon(d[0])]
    if poke_rl_y_unb:
        ax2.scatter([x_pos] * len(poke_rl_y_unb), poke_rl_y_unb, c=rl_color,
                    s=marker_size_pokemon * 3, marker='o', alpha=0.15, zorder=4, edgecolors='none')
        ax2.scatter([x_pos] * len(poke_rl_y_unb), poke_rl_y_unb, c=rl_color,
                    s=marker_size_pokemon, zorder=6, edgecolors='black',
                    linewidths=marker_edge_pokemon, marker='o')

    poke_llm_y_unb = [d[2] for d in unbounded_llm if is_pokemon(d[0])]
    if poke_llm_y_unb:
        ax2.scatter([x_pos] * len(poke_llm_y_unb), poke_llm_y_unb, c=llm_color,
                    s=marker_size_pokemon * 3, marker='s', alpha=0.15, zorder=4, edgecolors='none')
        ax2.scatter([x_pos] * len(poke_llm_y_unb), poke_llm_y_unb, c=llm_color,
                    s=marker_size_pokemon, marker='s', zorder=6, edgecolors='black',
                    linewidths=marker_edge_pokemon)

    # Human expert line
    ax2.axhline(y=100, color=COLORS['human'], linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)

    # Labels for unbounded - uses LABEL_OFFSETS_UNBOUNDED from top of file
    for benchmark, method, y, citation in unbounded_rl + unbounded_llm:
        ox, oy, ha = LABEL_OFFSETS_UNBOUNDED.get((benchmark, method), (12, 3, 'left'))
        short_cite = format_citation(citation)
        label = f"{benchmark} ({short_cite})"
        color = rl_color if (benchmark, method, y, citation) in unbounded_rl else llm_color
        poke = is_pokemon(benchmark)
        ax2.annotate(label, (x_pos, y), textcoords="offset points", xytext=(ox, oy),
                    fontsize=8 if poke else 7, color=color, ha=ha,
                    fontweight='bold' if poke else 'normal')

    # Heuristic markers for unbounded - uses LABEL_OFFSETS_HEURISTIC_UNBOUNDED from top of file
    for y, game, citation in heuristic_unbounded:
        ax2.scatter(x_pos, y, marker='+', c=heuristic_color, s=80, linewidths=1.8, zorder=4)
        ox, oy, ha = LABEL_OFFSETS_HEURISTIC_UNBOUNDED.get(game, (-8, 0, 'right'))
        short_cite = format_citation(citation) if ',' in citation else citation
        label = f"{game} ({short_cite})"
        ax2.annotate(label, (x_pos, y), textcoords="offset points", xytext=(ox, oy),
                    fontsize=6, color=heuristic_color, ha=ha, alpha=0.9)

    # Axis settings
    ax2.set_xlim(0.2, 1.8)
    ax2.set_xticks([1])
    ax2.set_xticklabels(['Unbounded'], fontsize=9)
    ax2.grid(True, alpha=0.25, linestyle='-', linewidth=0.4, axis='y', zorder=0)
    ax2.spines['left'].set_visible(False)
    ax2.tick_params(left=False)
    # Title removed

    # ============ CATEGORY REGIONS ============
    region_alpha = 0.12
    region_lw = 1.5

    # 1. Zero-Sum region (Backgammon, Chess, Go) - x~20-170
    zerosum_pts = np.array([[16, -3], [220, -3], [220, 132], [16, 132]])
    poly_zerosum = Polygon(zerosum_pts, fill=True, facecolor=COLORS['region_zerosum'],
                           alpha=region_alpha, edgecolor=COLORS['region_zerosum'],
                           linestyle='--', linewidth=region_lw, zorder=0)
    ax1.add_patch(poly_zerosum)
    x, y = REGION_LABEL_POSITIONS["Zero-Sum"]
    ax1.text(x, y, 'Zero-Sum', fontsize=7, color=COLORS['region_zerosum'],
             fontweight='bold', ha='left')

    # 2. Partially Observable Zero-Sum (Poker, Pokemon Battling + Dota, StarCraft)
    # Bounded portion - x~160-400
    partial_pts = np.array([[130, 32], [550, 32], [550, 122], [130, 122]])
    poly_partial = Polygon(partial_pts, fill=True, facecolor=COLORS['region_partial'],
                           alpha=region_alpha, edgecolor=COLORS['region_partial'],
                           linestyle='--', linewidth=region_lw, zorder=0)
    ax1.add_patch(poly_partial)
    # "Partially Observable Zero-Sum" label added at figure level below (to span between panels)

    # Unbounded portion - Dota 2 (105), StarCraft II (100, 50)
    partial_unb_pts = np.array([[0.5, 42], [1.1, 42], [1.1, 112], [0.5, 112]])
    poly_partial_unb = Polygon(partial_unb_pts, fill=True, facecolor=COLORS['region_partial'],
                               alpha=region_alpha, edgecolor=COLORS['region_partial'],
                               linestyle='--', linewidth=region_lw, zorder=0)
    ax2.add_patch(poly_partial_unb)

    # 3. Open-Ended region (Minecraft, Pokemon RPG)
    open_pts = np.array([[0.9, -3], [1.5, -3], [1.5, 78], [0.9, 78]])
    poly_open = Polygon(open_pts, fill=True, facecolor=COLORS['region_openended'],
                        alpha=region_alpha, edgecolor=COLORS['region_openended'],
                        linestyle='--', linewidth=region_lw, zorder=0)
    ax2.add_patch(poly_open)
    ax2.text(0.92, -1, 'Open-\nEnded', fontsize=7, color=COLORS['region_openended'],
             fontweight='bold', ha='left', va='bottom')

    # ============ LEGEND ============
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=rl_color,
               markersize=8, label='RL Agent', markeredgecolor='white', markeredgewidth=0.5),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=llm_color,
               markersize=8, label='LLM Agent', markeredgecolor='white', markeredgewidth=0.5),
        Line2D([0], [0], marker='+', color=heuristic_color, linestyle='None',
               markersize=8, markeredgewidth=1.8, label='Heuristic'),
        # Line2D([0], [0], color=COLORS['human'], linestyle='--', linewidth=1.5, label='Human Expert'),
    ]

    fig.legend(handles=legend_elements, loc='lower left',
               bbox_to_anchor=(0.10, 0.13), frameon=True, framealpha=0.95,
               edgecolor='#cccccc', fontsize=8, ncol=1)

    # Title removed; provided by LaTeX caption

    plt.subplots_adjust(left=0.10, right=0.95, bottom=0.12, top=0.97, wspace=0.05)

    # "Partially Observable Zero-Sum" label at figure level, centered between the two panels
    fig.text(0.70, 0.65, 'Partially\nObservable\nZero-Sum', fontsize=6,
             color=COLORS['region_partial'], fontweight='bold', ha='center', va='center')

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

    output_path = Path(__file__).parent.parent / "rl_vs_llm_benchmarks_v2.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n\nFigure saved to: {output_path}")

    # Also save PDF for LaTeX
    pdf_path = output_path.with_suffix('.pdf')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"PDF saved to: {pdf_path}")

    plt.close(fig)
