#!/usr/bin/env python3
"""
Tournament Brackets — Gen 1 OU (left) and Gen 9 OU (right).

Clean tournament bracket visualization with hardcoded results.
Best-of-99 single-elimination: QF → SF → F.

NeurIPS publication ready.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_PNG = Path(__file__).parent.parent / "gen9_tournament.png"
OUTPUT_PDF = Path(__file__).parent.parent / "gen9_tournament.pdf"

NEURIPS_RCPARAMS = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'text.usetex': False,
    'axes.linewidth': 0.8,
}

COLOR_WINNER = '#1a9850'
COLOR_LOSER = '#d73027'
COLOR_BOX = '#f7f7f7'
COLOR_BOX_BORDER = '#404040'

# Gen 1 OU bracket data
BRACKET_GEN1 = {
    'QF': [
        {'team1': 'PA-Agent', 'score1': 50, 'team2': 'FoulPlay', 'score2': 12},
        {'team1': 'ED-Testing', 'score1': 50, 'team2': 'MetaHorns', 'score2': 28},
        {'team1': '4thLesson', 'score1': 50, 'team2': 'GCOGS', 'score2': 11},
        {'team1': 'srsk-1729', 'score1': 50, 'team2': 'Exp-05', 'score2': 29},
    ],
    'SF': [
        {'team1': 'PA-Agent', 'score1': 50, 'team2': 'ED-Testing', 'score2': 13},
        {'team1': '4thLesson', 'score1': 50, 'team2': 'srsk-1729', 'score2': 20},
    ],
    'F': [
        {'team1': 'PA-Agent', 'score1': 50, 'team2': '4thLesson', 'score2': 28},
    ],
}

# Gen 9 OU bracket data
BRACKET_GEN9 = {
    'QF': [
        {'team1': 'FoulPlay', 'score1': 50, 'team2': 'Porygon2AI', 'score2': 12},
        {'team1': 'PA-Agent', 'score1': 50, 'team2': 'piploop', 'score2': 37},
        {'team1': 'ED-Testing', 'score1': 50, 'team2': 'srsk-1729', 'score2': 20},
        {'team1': 'Q', 'score1': 50, 'team2': 'MetaHorns', 'score2': 16},
    ],
    'SF': [
        {'team1': 'FoulPlay', 'score1': 50, 'team2': 'PA-Agent', 'score2': 39},
        {'team1': 'Q', 'score1': 50, 'team2': 'ED-Testing', 'score2': 22},
    ],
    'F': [
        {'team1': 'FoulPlay', 'score1': 50, 'team2': 'Q', 'score2': 14},
    ],
}


# =============================================================================
# DRAWING
# =============================================================================

BOX_W = 2.8
BOX_H = 1.3
X_GAP = 0.9


def draw_match_box(ax, x, y, match):
    """Draw a single match box with two team rows."""
    half_h = BOX_H / 2
    t1_wins = match['score1'] >= match['score2']
    accent_w = 0.07

    # Single outer box with shadow
    outer = mpatches.FancyBboxPatch(
        (x, y - half_h), BOX_W, BOX_H,
        boxstyle="round,pad=0.03", facecolor='white',
        edgecolor='#aaaaaa', linewidth=1.5, zorder=2)
    outer.set_path_effects([
        pe.SimplePatchShadow(offset=(2, -2), shadow_rgbFace='#666666', alpha=0.25),
        pe.Normal(),
    ])
    ax.add_patch(outer)

    # Horizontal divider
    ax.plot([x, x + BOX_W], [y, y], color='#aaaaaa', lw=1.2, zorder=3)

    for row, (team, score, wins) in enumerate([
        (match['team1'], match['score1'], t1_wins),
        (match['team2'], match['score2'], not t1_wins),
    ]):
        row_y = y if row == 0 else y - half_h
        color = COLOR_WINNER if wins else COLOR_LOSER

        # Colored left accent bar
        ax.add_patch(mpatches.Rectangle(
            (x, row_y), accent_w, half_h,
            facecolor=color, edgecolor='none', zorder=4))

        ax.text(x + accent_w + 0.1, row_y + half_h / 2, team,
                va='center', ha='left', fontsize=12, zorder=5,
                fontweight='bold' if wins else 'normal', color='#111111')
        ax.text(x + BOX_W - 0.12, row_y + half_h / 2, str(score),
                va='center', ha='right', fontsize=12, fontweight='bold', color=color, zorder=5)


def draw_bracket(ax, bracket, title):
    """Draw a complete bracket on the given axes."""
    half_h = BOX_H / 2

    qf_x = 0
    sf_x = qf_x + BOX_W + X_GAP
    f_x  = sf_x + BOX_W + X_GAP

    # Spread QF matches evenly with generous vertical gaps
    qf_ys = [6.5, 4.3, 2.1, -0.1]
    sf_ys = [(qf_ys[0] + qf_ys[1]) / 2, (qf_ys[2] + qf_ys[3]) / 2]
    f_ys  = [(sf_ys[0] + sf_ys[1]) / 2]

    for i, match in enumerate(bracket['QF']):
        draw_match_box(ax, qf_x, qf_ys[i], match)
    for i, match in enumerate(bracket['SF']):
        draw_match_box(ax, sf_x, sf_ys[i], match)
    draw_match_box(ax, f_x, f_ys[0], bracket['F'][0])

    lw, lc, la = 2.0, '#aaaaaa', 0.8

    # Connectors QF → SF
    for pair_idx in range(2):
        top_y = qf_ys[pair_idx * 2]
        bot_y = qf_ys[pair_idx * 2 + 1]
        sf_y  = sf_ys[pair_idx]
        mid_x = qf_x + BOX_W + X_GAP / 2
        ax.plot([qf_x + BOX_W, mid_x], [top_y, top_y], color=lc, lw=lw, alpha=la)
        ax.plot([qf_x + BOX_W, mid_x], [bot_y, bot_y], color=lc, lw=lw, alpha=la)
        ax.plot([mid_x, mid_x],         [top_y, bot_y], color=lc, lw=lw, alpha=la)
        ax.plot([mid_x, sf_x],           [sf_y, sf_y],  color=lc, lw=lw, alpha=la)

    # Connectors SF → F
    mid_x = sf_x + BOX_W + X_GAP / 2
    ax.plot([sf_x + BOX_W, mid_x], [sf_ys[0], sf_ys[0]], color=lc, lw=lw, alpha=la)
    ax.plot([sf_x + BOX_W, mid_x], [sf_ys[1], sf_ys[1]], color=lc, lw=lw, alpha=la)
    ax.plot([mid_x, mid_x],         [sf_ys[0], sf_ys[1]], color=lc, lw=lw, alpha=la)
    ax.plot([mid_x, f_x],            [f_ys[0],  f_ys[0]], color=lc, lw=lw, alpha=la)

    # Champion label above the final box
    ax.text(f_x + BOX_W / 2, f_ys[0] + half_h + 0.15,
            'Champion', va='bottom', ha='center',
            fontsize=14, fontweight='bold', color=COLOR_WINNER)

    # Round labels
    top_label_y = qf_ys[0] + half_h + 0.35
    for label, x in [('Quarterfinals', qf_x + BOX_W / 2),
                     ('Semifinals',    sf_x + BOX_W / 2),
                     ('Final',         f_x  + BOX_W / 2)]:
        ax.text(x, top_label_y, label, ha='center', va='bottom',
                fontsize=14, fontweight='bold', color='#333333')

    ax.set_xlim(-0.4, f_x + BOX_W + 0.4)
    ax.set_ylim(qf_ys[-1] - half_h - 0.3, top_label_y + 0.5)
    ax.axis('off')
    ax.set_title(title, fontweight='bold', pad=14, fontsize=16)


def create_figure():
    plt.rcParams.update(NEURIPS_RCPARAMS)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    draw_bracket(ax1, BRACKET_GEN1, 'Gen1OU (Best-of-99)')
    draw_bracket(ax2, BRACKET_GEN9, 'Gen9OU (Best-of-99)')

    plt.tight_layout()
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
