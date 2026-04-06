#!/usr/bin/env python3
"""
Track 1 Qualifying Results — Two-Panel Horizontal Bar Chart.

Left panel: Gen 1 OU, Right panel: Gen 9 OU.
Bars = GXE (%) sorted by qualifying rank.
Color distinguishes participants from organizer baselines.
Dashed line at top-8 qualifying cutoff.

NeurIPS publication ready.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_FILE = Path(__file__).parent.parent.parent / "pokeagent.github.io" / "leaderboard" / "track1_qualifying.json"
OUTPUT_PNG = Path(__file__).parent.parent / "qualifying_results.png"
OUTPUT_PDF = Path(__file__).parent.parent / "qualifying_results.pdf"

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
    'text.usetex': False,
    'axes.linewidth': 0.8,
    'axes.grid': False,
    'lines.linewidth': 1.5,
}

COLOR_PARTICIPANT = '#DC267F'   # magenta
COLOR_BASELINE = '#648FFF'      # blue
COLOR_CUTOFF = '#404040'        # dark gray

TOP_N = 20  # Show top N teams per format

# Tournament bracket participants (must match Figure 13)
TOURNAMENT_GEN1 = {'PA-Agent', 'FoulPlay', 'ED-Testing', 'MetaHorns',
                   '4thLesson', 'GCOGS', 'srsk-1729', 'Exp-05'}
TOURNAMENT_GEN9 = {'FoulPlay', 'Porygon2AI', 'PA-Agent', 'piploop',
                   'ED-Testing', 'srsk-1729', 'Q', 'MetaHorns'}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    with open(DATA_FILE) as f:
        data = json.load(f)
    return data['formats']


def parse_players(players, top_n=TOP_N):
    """Parse player data, return top_n sorted by rank."""
    result = []
    for p in players[:top_n + 10]:  # parse extra to backfill after filtering
        username = p.get('username', {})
        display = username.get('display', '?')
        is_baseline = username.get('is_starter_kit', False)
        gxe_str = p.get('gxe', '0%').replace('%', '')
        gxe = float(gxe_str) if gxe_str else 0.0
        wins = int(p.get('wins', 0))
        losses = int(p.get('losses', 0))
        rank = p.get('rank', 0)
        result.append({
            'name': display,
            'is_baseline': is_baseline,
            'gxe': gxe,
            'wins': wins,
            'losses': losses,
            'games': wins + losses,
            'rank': rank,
        })
    return result


def filter_top8_to_tournament(players, tournament_names, top_n=TOP_N):
    """Remove non-tournament participants from the top-8 qualifying slots.

    Walks through the list and removes any non-baseline participant that
    would be in the top 8 non-baselines but is NOT in the tournament bracket.
    Then truncates to top_n entries.
    """
    non_baseline_count = 0
    to_remove = set()
    for p in players:
        if not p['is_baseline']:
            non_baseline_count += 1
            if non_baseline_count <= 8:
                if p['name'] not in tournament_names:
                    to_remove.add(p['name'])
            else:
                break
    filtered = [p for p in players if p['name'] not in to_remove]
    return filtered[:top_n]


# =============================================================================
# PLOTTING
# =============================================================================

def plot_panel(ax, players, title, qualifying_cutoff=8):
    """Plot a single panel (one format)."""
    # Reverse for bottom-to-top display (rank 1 at top)
    players_rev = list(reversed(players))
    n = len(players_rev)

    names = [p['name'] for p in players_rev]
    gxe_vals = [p['gxe'] for p in players_rev]
    colors = [COLOR_BASELINE if p['is_baseline'] else COLOR_PARTICIPANT for p in players_rev]

    y_pos = np.arange(n)
    bars = ax.barh(y_pos, gxe_vals, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5, height=0.7)

    # Annotate game counts on bars
    for i, p in enumerate(players_rev):
        game_text = f"{p['games']:,}"
        # Place text inside bar for long bars, outside for short ones
        if p['gxe'] > 45:
            ax.text(p['gxe'] - 1, i, game_text, va='center', ha='right',
                    fontsize=7, color='white', fontweight='bold')
        else:
            ax.text(p['gxe'] + 0.5, i, game_text, va='center', ha='left',
                    fontsize=7, color='#404040')

    # Qualifying cutoff line: find the position of the 8th non-baseline participant
    # (baselines don't count toward the qualifying cutoff)
    non_baseline_count = 0
    cutoff_rank = None
    for idx, p in enumerate(players):  # players is top-to-bottom (rank 1 first)
        if not p['is_baseline']:
            non_baseline_count += 1
            if non_baseline_count == qualifying_cutoff:
                cutoff_rank = idx  # 0-indexed position from top
                break
    if cutoff_rank is not None:
        # In the reversed list, this player is at position (n - 1 - cutoff_rank)
        cutoff_y = (n - 1 - cutoff_rank) - 0.5
        ax.axhline(y=cutoff_y, color=COLOR_CUTOFF, linestyle='--', linewidth=1.2, alpha=0.7)
        ax.text(99, cutoff_y + 0.3, 'Top-8 Cutoff', fontsize=8.5, color=COLOR_CUTOFF,
                fontstyle='italic', fontweight='bold', va='bottom', ha='right')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=7.5)
    ax.set_xlabel('GXE (%)', fontweight='bold')
    ax.set_title(title, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, n - 0.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.grid(axis='x', alpha=0.2, linewidth=0.5)


def create_figure():
    plt.rcParams.update(NEURIPS_RCPARAMS)

    formats = load_data()
    gen1_players = parse_players(formats['gen1ou'])
    gen9_players = parse_players(formats['gen9ou'])

    # Remove non-tournament participants from the top 8
    gen1_players = filter_top8_to_tournament(gen1_players, TOURNAMENT_GEN1)
    gen9_players = filter_top8_to_tournament(gen9_players, TOURNAMENT_GEN9)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

    plot_panel(ax1, gen1_players, 'Gen 1 OU Qualifying')
    plot_panel(ax2, gen9_players, 'Gen 9 OU Qualifying')

    # Shared legend
    legend_elements = [
        Patch(facecolor=COLOR_PARTICIPANT, alpha=0.85, label='Participant'),
        Patch(facecolor=COLOR_BASELINE, alpha=0.85, label='Organizer Baseline'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, framealpha=0.9,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    return fig


def main():
    fig = create_figure()
    print(f"Saving to: {OUTPUT_PNG}")
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saving to: {OUTPUT_PDF}")
    fig.savefig(OUTPUT_PDF, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("Done!")


if __name__ == "__main__":
    main()
