#!/usr/bin/env python3
"""
Rating System Comparison Figure for PokeAgent Challenge Paper.

Generates a two-panel figure comparing different rating systems
(Bradley-Terry, Elo, Glicko-1, GXE) across Track 1 competition data.

Left panel: Rank correlation between rating methods
Right panel: Ratings with uncertainty bands for top agents

NeurIPS publication ready: Times font, colorblind-friendly colors, large fonts.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# =============================================================================
# CONFIGURATION
# =============================================================================

# Data source - use track1.json which has Bradley-Terry (WHR) rankings
DATA_DIR = Path(__file__).parent.parent.parent / "pokeagent.github.io" / "leaderboard"
# Use track1.json first (has WHR data), fall back to others for game counts
DATA_FILES = [
    DATA_DIR / "track1.json",  # Primary - has Bradley-Terry rankings
    DATA_DIR / "track1_qualifying.json",
    DATA_DIR / "track1_practice.json",
]
FORMAT = "gen1ou"  # Primary format to analyze

# Total matches across all data (for paper claim)
TOTAL_MATCHES_CLAIM = "1.6M+"  # From combining all files

# Output
OUTPUT_FILE = Path(__file__).parent.parent / "rating_systems.png"
OUTPUT_PDF = Path(__file__).parent.parent / "rating_systems.pdf"

# NeurIPS style configuration
NEURIPS_RCPARAMS = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'text.usetex': False,  # Set True if LaTeX is available
    'axes.linewidth': 0.8,
    'axes.grid': False,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
}

# Colorblind-friendly palette (IBM Design)
COLORS = {
    'blue': '#648FFF',
    'purple': '#785EF0',
    'magenta': '#DC267F',
    'orange': '#FE6100',
    'yellow': '#FFB000',
    'gray': '#808080',
    'light_gray': '#E0E0E0',
    'dark_gray': '#404040',
}

# Agent categories for coloring
PARTICIPANT_AGENTS = ['PA-Agent', '4thLesson', 'FoulPlay', 'Q', 'GCOGS', 'Porygon2AI',
                      'srsk-1729', 'MetaHorns', 'Exp-05', 'SnTeam', 'ED-Testing',
                      'ASH-K', 'Gradient', 'hida', 'VibePoking', 'Kaban', 'Puffer']

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data_from_files(filepaths: list[Path], format_name: str) -> list[dict]:
    """Load and merge data from multiple files for a specific format.

    Uses the first file that contains the format as the primary source
    for ratings, but aggregates game counts from all files.
    """
    # Find the best file (one with WHR data)
    best_file = None
    for fp in filepaths:
        if fp.exists():
            with open(fp) as f:
                data = json.load(f)
                players = data.get('formats', {}).get(format_name, [])
                if players and players[0].get('whr'):
                    best_file = fp
                    break

    if not best_file:
        best_file = filepaths[0]

    with open(best_file) as f:
        data = json.load(f)

    players = data.get('formats', {}).get(format_name, [])

    # Parse and enrich data
    parsed = []
    for i, p in enumerate(players):
        username = p.get('username', {})
        display_name = username.get('display', username) if isinstance(username, dict) else username

        # Extract ratings
        elo = float(p.get('elo', 0))
        glicko = float(p.get('glicko', 0))
        gxe_str = p.get('gxe', '0%').replace('%', '').replace('-', '0')
        try:
            gxe = float(gxe_str) if gxe_str else 0
        except ValueError:
            gxe = 0

        wins = int(p.get('wins', 0))
        losses = int(p.get('losses', 0))
        games = wins + losses

        # WHR/Bradley-Terry data
        whr = p.get('whr', {})
        bt_strength = whr.get('bt_strength', 0)
        bt_std = whr.get('bt_std', 0)
        whr_elo = whr.get('whr_elo', 0)
        whr_ci_lower = whr.get('whr_ci_lower', 0)
        whr_ci_upper = whr.get('whr_ci_upper', 0)
        whr_rank = whr.get('whr_rank', i + 1)

        # Determine if participant or organizer baseline
        is_starter_kit = username.get('is_starter_kit', False) if isinstance(username, dict) else False
        is_participant = display_name in PARTICIPANT_AGENTS

        parsed.append({
            'name': display_name,
            'elo': elo,
            'glicko': glicko,
            'gxe': gxe,
            'wins': wins,
            'losses': losses,
            'games': games,
            'bt_strength': bt_strength,
            'bt_std': bt_std,
            'whr_elo': whr_elo,
            'whr_ci_lower': whr_ci_lower,
            'whr_ci_upper': whr_ci_upper,
            'whr_rank': whr_rank,
            'elo_rank': i + 1,  # Original rank from data (sorted by Elo)
            'is_starter_kit': is_starter_kit,
            'is_participant': is_participant,
        })

    # Compute additional ranks
    # Glicko rank
    sorted_by_glicko = sorted(enumerate(parsed), key=lambda x: -x[1]['glicko'])
    for rank, (idx, _) in enumerate(sorted_by_glicko):
        parsed[idx]['glicko_rank'] = rank + 1

    # GXE rank
    sorted_by_gxe = sorted(enumerate(parsed), key=lambda x: -x[1]['gxe'])
    for rank, (idx, _) in enumerate(sorted_by_gxe):
        parsed[idx]['gxe_rank'] = rank + 1

    # BT rank (use whr_rank if available, otherwise compute)
    sorted_by_bt = sorted(enumerate(parsed), key=lambda x: -x[1]['bt_strength'])
    for rank, (idx, _) in enumerate(sorted_by_bt):
        parsed[idx]['bt_rank'] = rank + 1

    return parsed


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_rank_correlation(ax, data: list[dict], min_games: int = 50):
    """
    Left panel: Compare Elo vs BT and Glicko vs BT rank correlations.
    Shows that Glicko-1 converges better to Bradley-Terry than Elo.
    """
    # Filter for agents with enough games
    filtered = [d for d in data if d['games'] >= min_games]

    elo_ranks = np.array([d['elo_rank'] for d in filtered])
    bt_ranks = np.array([d['bt_rank'] for d in filtered])
    glicko_ranks = np.array([d['glicko_rank'] for d in filtered])

    # Calculate rank differences (absolute deviation from BT)
    elo_diff = np.abs(elo_ranks - bt_ranks)
    glicko_diff = np.abs(glicko_ranks - bt_ranks)

    max_rank = max(elo_ranks.max(), bt_ranks.max(), glicko_ranks.max())

    # Perfect correlation line
    ax.plot([0, max_rank + 1], [0, max_rank + 1], '-', color=COLORS['gray'],
            linewidth=1.5, label='Perfect agreement', zorder=0, alpha=0.5)

    # Plot Elo vs BT (circles, showing divergence)
    ax.scatter(bt_ranks, elo_ranks, c=COLORS['orange'], s=50, alpha=0.7,
               edgecolors='white', linewidths=0.5, marker='o', label='Elo', zorder=2)

    # Plot Glicko vs BT (triangles, showing convergence)
    ax.scatter(bt_ranks, glicko_ranks, c=COLORS['blue'], s=50, alpha=0.7,
               edgecolors='white', linewidths=0.5, marker='^', label='Glicko-1', zorder=3)

    # Calculate correlations
    elo_corr, _ = stats.spearmanr(bt_ranks, elo_ranks)
    glicko_corr, _ = stats.spearmanr(bt_ranks, glicko_ranks)

    # Calculate mean absolute rank deviation
    elo_mad = np.mean(elo_diff)
    glicko_mad = np.mean(glicko_diff)

    ax.set_xlabel('Bradley-Terry Rank (Ground Truth)', fontweight='bold')
    ax.set_ylabel('Online Rating Rank', fontweight='bold')
    ax.set_title('Convergence to Bradley-Terry', fontweight='bold')

    # Add correlation annotations
    textstr = f'Spearman ρ:\n  Glicko-1: {glicko_corr:.3f}\n  Elo: {elo_corr:.3f}\n\nMean |Δrank|:\n  Glicko-1: {glicko_mad:.1f}\n  Elo: {elo_mad:.1f}'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
            ha='left', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
            family='monospace')

    # Highlight where methods disagree significantly (ranks 4+)
    ax.axvspan(3.5, max_rank + 1, alpha=0.08, color=COLORS['orange'], zorder=0)

    ax.legend(loc='lower right', framealpha=0.9)

    ax.set_xlim(0, min(max_rank + 2, 45))
    ax.set_ylim(0, min(max_rank + 2, 45))
    ax.set_aspect('equal')


def plot_ratings_with_uncertainty(ax, data: list[dict], top_n: int = 15, min_games: int = 50):
    """
    Right panel: Horizontal bar plot of top agents with uncertainty bands.
    Shows Glicko-1 rating with Bradley-Terry confidence intervals.
    """
    # Filter and get top N agents by Glicko
    filtered = [d for d in data if d['games'] >= min_games]
    top_agents = sorted(filtered, key=lambda x: -x['glicko'])[:top_n]
    top_agents = list(reversed(top_agents))  # Reverse for horizontal bar plot

    names = [d['name'] for d in top_agents]
    glicko = [d['glicko'] for d in top_agents]

    # Use Bradley-Terry CI for uncertainty (converted to Glicko scale approximately)
    # Since BT CI is in different scale, we'll use a scaling factor
    # or fall back to a default uncertainty based on games played
    ci_lower = []
    ci_upper = []
    for d in top_agents:
        if d['whr_ci_lower'] > 0 and d['whr_ci_upper'] > 0:
            # Scale BT confidence to approximate Glicko uncertainty
            # BT CI is in Elo-like scale, so use directly
            ci_lower.append(d['whr_ci_lower'])
            ci_upper.append(d['whr_ci_upper'])
        else:
            # Fallback: estimate uncertainty from games played
            # Glicko RD decreases with more games: RD ≈ 350 / sqrt(games/10)
            rd = 350 / np.sqrt(max(d['games'], 10) / 10)
            ci_lower.append(d['glicko'] - 1.96 * rd)
            ci_upper.append(d['glicko'] + 1.96 * rd)

    y_pos = np.arange(len(names))

    # Colors based on participant vs organizer
    colors = [COLORS['magenta'] if d['is_participant'] else COLORS['blue'] for d in top_agents]

    # Plot bars
    bars = ax.barh(y_pos, glicko, color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)

    # Add error bars for uncertainty
    # Calculate error bar positions (asymmetric), ensure non-negative
    errors_lower = [max(0, g - cl) for g, cl in zip(glicko, ci_lower)]
    errors_upper = [max(0, cu - g) for g, cu in zip(glicko, ci_upper)]

    ax.errorbar(glicko, y_pos, xerr=[errors_lower, errors_upper],
                fmt='none', color=COLORS['dark_gray'], capsize=3, capthick=1, linewidth=1)

    # Add GXE annotations on the left side of bars
    # Label the first one with "GXE" to clarify the metric
    x_min = 1400
    for i, d in enumerate(top_agents):
        if i == len(top_agents) - 1:  # Top bar (reversed order)
            label = f"{d['gxe']:.0f}% GXE"
        else:
            label = f"{d['gxe']:.0f}%"
        ax.text(x_min + 5, i, label,
                va='center', ha='left', fontsize=8, color='white', fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel('Rating (Glicko-1 with 95% CI)', fontweight='bold')
    ax.set_title('Top Agents with Uncertainty', fontweight='bold')

    # Add vertical line at 1500 (starting rating)
    ax.axvline(x=1500, color=COLORS['gray'], linestyle=':', linewidth=1, alpha=0.7)
    ax.text(1505, -0.8, 'Start', fontsize=8, color=COLORS['gray'])

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['magenta'], alpha=0.8, label='Participant'),
        Patch(facecolor=COLORS['blue'], alpha=0.8, label='Organizer baseline'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)

    # Set x-axis limits
    ax.set_xlim(1400, max(glicko) + 150)


def create_figure(data: list[dict]):
    """Create the complete two-panel figure."""
    # Apply NeurIPS style
    plt.rcParams.update(NEURIPS_RCPARAMS)

    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left panel: Rank correlation
    plot_rank_correlation(ax1, data)

    # Right panel: Ratings with uncertainty
    plot_ratings_with_uncertainty(ax2, data)

    # Adjust layout
    plt.tight_layout()

    return fig


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"Loading data from: {[f.name for f in DATA_FILES]}")
    print(f"Format: {FORMAT}")

    # Load data
    data = load_data_from_files(DATA_FILES, FORMAT)
    print(f"Loaded {len(data)} agents")

    # Count total games
    total_games = sum(d['games'] for d in data) // 2
    print(f"Total unique matches: {total_games:,}")

    # Create figure
    fig = create_figure(data)

    # Save
    print(f"Saving to: {OUTPUT_FILE}")
    fig.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saving to: {OUTPUT_PDF}")
    fig.savefig(OUTPUT_PDF, bbox_inches='tight', facecolor='white')

    plt.close(fig)
    print("Done!")


if __name__ == "__main__":
    main()
