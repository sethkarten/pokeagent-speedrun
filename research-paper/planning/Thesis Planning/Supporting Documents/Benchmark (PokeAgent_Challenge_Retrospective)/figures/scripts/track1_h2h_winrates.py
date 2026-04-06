#!/usr/bin/env python3
"""
Track 1 Head-to-Head Win Rate Heatmaps.

Two-panel heatmap: Gen 1 OU (left), Gen 9 OU (right).
Shows win rates between top non-baseline participants.
Cell color = win rate, cell text = W-L record.

NeurIPS publication ready.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(__file__).parent.parent.parent / "pokeagent.github.io" / "leaderboard"
H2H_GEN1 = DATA_DIR / "h2h_gen1ou.json"
H2H_GEN9 = DATA_DIR / "h2h_gen9ou.json"
OUTPUT_PNG = Path(__file__).parent.parent / "track1_h2h.png"
OUTPUT_PDF = Path(__file__).parent.parent / "track1_h2h.pdf"

NEURIPS_RCPARAMS = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'text.usetex': False,
    'axes.linewidth': 0.8,
}

MIN_GAMES_H2H = 1  # Lower threshold to fill more cells

# Tournament participants (must match bracket in Figure 13)
TOURNAMENT_GEN1 = ['PA-Agent', 'FoulPlay', 'ED-Testing', 'MetaHorns',
                   '4thLesson', 'GCOGS', 'srsk-1729', 'Exp-05']
TOURNAMENT_GEN9 = ['FoulPlay', 'Porygon2AI', 'PA-Agent', 'piploop',
                   'ED-Testing', 'srsk-1729', 'Q', 'MetaHorns']

# Best-of-99 tournament matches (team1 wins, team2 loses)
# Each entry: (winner_display_name, winner_wins, loser_display_name, loser_wins)
BRACKET_MATCHES_GEN1 = [
    # QF
    ('PA-Agent', 50, 'FoulPlay', 12),
    ('ED-Testing', 50, 'MetaHorns', 28),
    ('4thLesson', 50, 'GCOGS', 11),
    ('srsk-1729', 50, 'Exp-05', 29),
    # SF
    ('PA-Agent', 50, 'ED-Testing', 13),
    ('4thLesson', 50, 'srsk-1729', 20),
    # F
    ('PA-Agent', 50, '4thLesson', 28),
]

BRACKET_MATCHES_GEN9 = [
    # QF
    ('FoulPlay', 50, 'Porygon2AI', 12),
    ('PA-Agent', 50, 'piploop', 37),
    ('ED-Testing', 50, 'srsk-1729', 20),
    ('Q', 50, 'MetaHorns', 16),
    # SF
    ('FoulPlay', 50, 'PA-Agent', 39),
    ('Q', 50, 'ED-Testing', 22),
    # F
    ('FoulPlay', 50, 'Q', 14),
]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_h2h(filepath: Path) -> list[dict]:
    with open(filepath) as f:
        data = json.load(f)
    return data.get('players', [])


def get_tournament_participants(players: list[dict], names: list[str]) -> list[dict]:
    """Get tournament participants by display name, sorted by Elo (descending)."""
    name_set = set(names)
    selected = [p for p in players if p.get('display_name', '') in name_set]
    selected.sort(key=lambda x: -x.get('elo', 0))
    return selected


def _normalize_key(username: str) -> str:
    """Normalize a username to the h2h key format: lowercase, no hyphens/underscores."""
    return username.lower().replace('-', '').replace('_', '')


def build_h2h_matrix(players: list[dict], all_players: list[dict],
                     bracket_matches: list[tuple] = ()):
    """Build win rate matrix and W-L text matrix.

    Combines qualifying ladder h2h records with tournament bracket results.
    """
    n = len(players)

    names = [p['display_name'] for p in players]
    name_to_idx = {name: i for i, name in enumerate(names)}
    norm_keys = [_normalize_key(p['username']) for p in players]

    # Accumulate raw W-L counts
    wins = np.zeros((n, n), dtype=int)
    losses = np.zeros((n, n), dtype=int)

    # 1) Qualifying ladder records from JSON h2h_data
    for i, p in enumerate(players):
        h2h = p.get('h2h_data', {})
        for j in range(n):
            if i == j:
                continue
            opp_key = norm_keys[j]
            record = h2h.get(opp_key, None)
            if record:
                wins[i][j] += record.get('w', 0)
                losses[i][j] += record.get('l', 0)

    # 2) Tournament bracket matches
    for t1_name, t1_wins, t2_name, t2_wins in bracket_matches:
        i = name_to_idx.get(t1_name)
        j = name_to_idx.get(t2_name)
        if i is not None and j is not None:
            wins[i][j] += t1_wins
            losses[i][j] += t2_wins
            wins[j][i] += t2_wins
            losses[j][i] += t1_wins

    # 3) Compute win rates and build output matrices
    winrate_matrix = np.full((n, n), np.nan)
    text_matrix = [[''] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                elo = players[i].get('elo', 0)
                text_matrix[i][j] = f'{elo:.0f}'
                continue
            total = wins[i][j] + losses[i][j]
            if total >= MIN_GAMES_H2H:
                w = int(wins[i][j])
                l = int(losses[i][j])
                wr = w / total * 100
                winrate_matrix[i][j] = wr
                text_matrix[i][j] = f'{w}-{l}'

    return winrate_matrix, text_matrix, names


# =============================================================================
# PLOTTING
# =============================================================================

def plot_heatmap(ax, winrate_matrix, text_matrix, names, title):
    """Plot a single heatmap panel."""
    n = len(names)

    # Custom diverging colormap: red (0%) -> white (50%) -> green (100%)
    cmap = LinearSegmentedColormap.from_list('winrate',
        [(0.0, '#d73027'), (0.25, '#f4a582'), (0.5, '#f7f7f7'),
         (0.75, '#a1d99b'), (1.0, '#1a9850')])

    # Mask NaN for imshow
    masked = np.ma.masked_invalid(winrate_matrix)
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=100, aspect='equal')

    # Diagonal shading first (behind text)
    for i in range(n):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                     fill=True, facecolor='#d9d9d9', edgecolor='white', linewidth=1))

    # Empty off-diagonal cells get a light hatching
    for i in range(n):
        for j in range(n):
            if i != j and np.isnan(winrate_matrix[i][j]):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                             fill=True, facecolor='#f0f0f0', edgecolor='white', linewidth=0.5))

    # Text annotations
    for i in range(n):
        for j in range(n):
            text = text_matrix[i][j]
            if not text:
                continue
            if i == j:
                ax.text(j, i, text, ha='center', va='center', fontsize=7.5,
                        color='#404040', fontweight='bold')
            else:
                wr = winrate_matrix[i][j]
                # High contrast text
                text_color = 'white' if wr > 80 or wr < 20 else '#1a1a1a'
                ax.text(j, i, text, ha='center', va='center', fontsize=7,
                        color=text_color, fontweight='bold')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=50, ha='right', fontsize=8.5)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_title(title, fontweight='bold', pad=12, fontsize=12)

    # Grid lines
    for i in range(n + 1):
        ax.axhline(y=i - 0.5, color='white', linewidth=1)
        ax.axvline(x=i - 0.5, color='white', linewidth=1)

    return im


def create_figure():
    plt.rcParams.update(NEURIPS_RCPARAMS)

    gen1_players = load_h2h(H2H_GEN1)
    gen9_players = load_h2h(H2H_GEN9)

    gen1_top = get_tournament_participants(gen1_players, TOURNAMENT_GEN1)
    gen9_top = get_tournament_participants(gen9_players, TOURNAMENT_GEN9)

    gen1_wr, gen1_text, gen1_names = build_h2h_matrix(gen1_top, gen1_players, BRACKET_MATCHES_GEN1)
    gen9_wr, gen9_text, gen9_names = build_h2h_matrix(gen9_top, gen9_players, BRACKET_MATCHES_GEN9)

    # Taller figure for squarer matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.5))

    im1 = plot_heatmap(ax1, gen1_wr, gen1_text, gen1_names, 'Gen 1 OU')
    im2 = plot_heatmap(ax2, gen9_wr, gen9_text, gen9_names, 'Gen 9 OU')

    # Colorbar: positioned explicitly to avoid overlap
    cbar_ax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
    cbar = fig.colorbar(im2, cax=cbar_ax)
    cbar.set_label('Win Rate (%)', fontsize=10)
    cbar.set_ticks([0, 25, 50, 75, 100])
    cbar.ax.tick_params(labelsize=8)

    fig.subplots_adjust(left=0.08, right=0.90, bottom=0.18, top=0.92, wspace=0.4)
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
