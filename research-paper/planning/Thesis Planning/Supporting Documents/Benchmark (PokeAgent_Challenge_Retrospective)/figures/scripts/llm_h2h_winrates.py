#!/usr/bin/env python3
"""
LLM Baseline Head-to-Head Win Rate Heatmap (Gen 9 OU Long Timer).

Combines Glicko-1/GXE ratings from Table (gen9-baselines) with pairwise
h2h records from the Long Timer ladder.  Only pure-LLM and PokéChamp-LLM
baselines are shown (ABYSSAL, DC, pokellmon, MAX-POWER removed).

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
H2H_GEN9 = DATA_DIR / "h2h_gen9ou.json"
QUAL_FILE = DATA_DIR / "track1_qualifying.json"
OUTPUT_PNG = Path(__file__).parent.parent / "llm_h2h.png"
OUTPUT_PDF = Path(__file__).parent.parent / "llm_h2h.pdf"

NEURIPS_RCPARAMS = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'text.usetex': False,
    'axes.linewidth': 0.8,
}

# ── Player list ordered by Glicko-1 (descending) ────────────────────────────
# (key, display_name, glicko, gxe)
# First 8: from tab:gen9-baselines (Long Timer ladder)
# Remaining: from main competition ladder h2h JSON / qualifying JSON
PLAYERS = [
    ('gem3f',     'Gemini 3 Flash',  1738, 71.4),
    ('gem3p',     'Gemini 3 Pro',    1613, 60.7),
    ('gem25f',    'Gemini 2.5 Flash',1591, 58.6),
    ('pc_gm34b',  'PC-Gemma3-4B',    1566, 56.3),
    ('pc_ll8b',   'PC-Llama3.1-8B',  1531, 52.9),
    ('pc_gm31b',  'PC-Gemma3-1B',    1529, 52.8),
    ('gpt_oss',   'GPT-4o',          1517, 51.6),
    ('qw314b',    'Qwen3-14B',       1512, 51.1),
    ('gm312b',    'Gemma3-12B',      1194, 32.8),
    ('gem25fl',   'Gemini 2.5 FL',   1190, 31.5),
    ('ll8b',      'Llama3.1-8B',     1158, 30.7),
    ('qw38b',     'Qwen3-8B',        1152, 22.6),
    ('qw34b',     'Qwen3-4B',        1147, 22.2),
]

N = len(PLAYERS)
KEY_TO_IDX = {p[0]: i for i, p in enumerate(PLAYERS)}

# ── H2H records: (row_key, col_key): (row_wins, row_losses) ─────────────────
# Parsed from Long Timer ladder data.  Ties are dropped (W-L only).
# Where the original data was truncated, cross-references fill the gap.
_RAW_H2H = {
    # gem3f row
    ('gem3f','gem3p'):(8,14),   ('gem3f','gem25f'):(6,3),
    ('gem3f','pc_gm34b'):(6,1), ('gem3f','pc_ll8b'):(4,1),
    ('gem3f','pc_gm31b'):(3,3), ('gem3f','gpt_oss'):(8,1),
    ('gem3f','qw314b'):(6,2),   ('gem3f','gm312b'):(17,2),
    ('gem3f','gem25fl'):(38,2),  ('gem3f','ll8b'):(24,2),
    ('gem3f','qw34b'):(7,1),
    # gem3p row
    ('gem3p','gem25f'):(3,2),   ('gem3p','pc_gm34b'):(4,0),
    ('gem3p','pc_ll8b'):(3,0),  ('gem3p','pc_gm31b'):(6,0),
    ('gem3p','gpt_oss'):(4,3),  ('gem3p','qw314b'):(2,2),
    ('gem3p','gm312b'):(4,3),   ('gem3p','gem25fl'):(3,1),
    ('gem3p','ll8b'):(3,3),     ('gem3p','qw38b'):(5,1),
    ('gem3p','qw34b'):(9,4),
    # gem25f row
    ('gem25f','pc_gm34b'):(5,3), ('gem25f','pc_ll8b'):(7,2),
    ('gem25f','pc_gm31b'):(1,0), ('gem25f','gpt_oss'):(18,10),
    ('gem25f','qw314b'):(9,4),   ('gem25f','gm312b'):(14,1),
    ('gem25f','gem25fl'):(6,5),  ('gem25f','ll8b'):(22,15),
    ('gem25f','qw38b'):(16,12),  ('gem25f','qw34b'):(2,2),
    # pc_gm34b row
    ('pc_gm34b','pc_ll8b'):(1,1),  ('pc_gm34b','pc_gm31b'):(2,2),
    ('pc_gm34b','gpt_oss'):(4,3),  ('pc_gm34b','qw314b'):(4,2),
    ('pc_gm34b','gm312b'):(7,4),   ('pc_gm34b','gem25fl'):(9,3),
    ('pc_gm34b','ll8b'):(16,7),    ('pc_gm34b','qw38b'):(2,3),
    ('pc_gm34b','qw34b'):(12,5),
    # pc_ll8b row
    ('pc_ll8b','pc_gm31b'):(7,11), ('pc_ll8b','qw314b'):(3,2),
    ('pc_ll8b','gm312b'):(9,10),   ('pc_ll8b','gem25fl'):(12,6),
    ('pc_ll8b','qw38b'):(4,1),     ('pc_ll8b','qw34b'):(6,3),
    # pc_gm31b row
    ('pc_gm31b','gpt_oss'):(6,9),  ('pc_gm31b','qw314b'):(18,7),
    ('pc_gm31b','gm312b'):(5,2),   ('pc_gm31b','gem25fl'):(6,3),
    ('pc_gm31b','ll8b'):(10,4),    ('pc_gm31b','qw38b'):(3,1),
    ('pc_gm31b','qw34b'):(3,5),
    # gpt_oss row
    ('gpt_oss','qw314b'):(3,2),    ('gpt_oss','gm312b'):(17,9),
    ('gpt_oss','gem25fl'):(15,19),  ('gpt_oss','ll8b'):(6,3),
    ('gpt_oss','qw38b'):(2,1),     ('gpt_oss','qw34b'):(11,5),
    # qw314b row
    ('qw314b','gm312b'):(6,7),    ('qw314b','gem25fl'):(0,2),
    ('qw314b','ll8b'):(11,9),     ('qw314b','qw38b'):(5,2),
    ('qw314b','qw34b'):(1,1),
    # gm312b row
    ('gm312b','gem25fl'):(2,3),   ('gm312b','ll8b'):(32,23),
    ('gm312b','qw38b'):(4,4),    ('gm312b','qw34b'):(9,5),
    # gem25fl row
    ('gem25fl','ll8b'):(14,11),   ('gem25fl','qw38b'):(5,2),
    ('gem25fl','qw34b'):(12,8),
    # ll8b row
    ('ll8b','qw38b'):(4,5),      ('ll8b','qw34b'):(5,7),
    # qw38b row
    ('qw38b','qw34b'):(8,3),
}


# =============================================================================
# MATRIX BUILDING
# =============================================================================

def build_matrices():
    """Build win-rate matrix, text matrix, and name list from hardcoded data."""
    names = [p[1] for p in PLAYERS]
    glicko = [p[2] for p in PLAYERS]
    gxe = [p[3] for p in PLAYERS]

    wins = np.zeros((N, N), dtype=int)
    losses = np.zeros((N, N), dtype=int)

    for (a, b), (w, l) in _RAW_H2H.items():
        i, j = KEY_TO_IDX[a], KEY_TO_IDX[b]
        wins[i][j] += w
        losses[i][j] += l
        wins[j][i] += l
        losses[j][i] += w

    # Also merge main-ladder h2h records from JSON
    try:
        with open(H2H_GEN9) as f:
            all_players = json.load(f).get('players', [])
        # Map display_name → key
        dn_to_key = {
            'LLM-gem25f': 'gem25f', 'LLM-gpt-oss': 'gpt_oss',
            'LLM-qwen3-14b': 'qw314b', 'LLM-gemma3-12b': 'gm312b',
            'LLM-gem25fl': 'gem25fl', 'LLM-llama31-8b': 'll8b',
            'LLM-qwen3-8b': 'qw38b', 'LLM-qwen3-4b': 'qw34b',
            'PokéChamp-gemma3-1b': 'pc_gm31b',
            'PokéChamp-gemma3-4b': 'pc_gm34b',
            'PokéChamp-llama31-8b': 'pc_ll8b',
            'PokéChamp-qwen3-8b': 'qw38b',  # not used but safe
        }
        def _nk(u): return u.lower().replace('-', '').replace('_', '')
        for p in all_players:
            dn = p.get('display_name', '')
            if dn not in dn_to_key:
                continue
            key_i = dn_to_key[dn]
            if key_i not in KEY_TO_IDX:
                continue
            i = KEY_TO_IDX[key_i]
            for p2 in all_players:
                dn2 = p2.get('display_name', '')
                if dn2 not in dn_to_key:
                    continue
                key_j = dn_to_key[dn2]
                if key_j not in KEY_TO_IDX or key_j == key_i:
                    continue
                j = KEY_TO_IDX[key_j]
                opp_norm = _nk(p2['username'])
                rec = p.get('h2h_data', {}).get(opp_norm)
                if rec:
                    wins[i][j] += rec.get('w', 0)
                    losses[i][j] += rec.get('l', 0)
    except FileNotFoundError:
        pass  # no JSON available, use hardcoded only

    winrate_matrix = np.full((N, N), np.nan)
    text_matrix = [[''] * N for _ in range(N)]

    for i in range(N):
        for j in range(N):
            if i == j:
                text_matrix[i][j] = f'{glicko[i]}\n{gxe[i]:.1f}%'
                continue
            total = wins[i][j] + losses[i][j]
            if total >= 1:
                w, l = int(wins[i][j]), int(losses[i][j])
                winrate_matrix[i][j] = w / total * 100
                text_matrix[i][j] = f'{w}-{l}'

    return winrate_matrix, text_matrix, names


# =============================================================================
# PLOTTING
# =============================================================================

def plot_heatmap(ax, winrate_matrix, text_matrix, names):
    n = len(names)

    cmap = LinearSegmentedColormap.from_list('winrate',
        [(0.0, '#d73027'), (0.25, '#f4a582'), (0.5, '#f7f7f7'),
         (0.75, '#a1d99b'), (1.0, '#1a9850')])

    masked = np.ma.masked_invalid(winrate_matrix)
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=100, aspect='equal')

    # Diagonal shading
    for i in range(n):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                     fill=True, facecolor='#d9d9d9', edgecolor='white', linewidth=1))

    # Empty off-diagonal cells
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
                ax.text(j, i, text, ha='center', va='center', fontsize=6.5,
                        color='#404040', fontweight='bold', linespacing=1.15)
            else:
                wr = winrate_matrix[i][j]
                text_color = 'white' if wr > 80 or wr < 20 else '#1a1a1a'
                ax.text(j, i, text, ha='center', va='center', fontsize=6.5,
                        color=text_color, fontweight='bold')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=50, ha='right', fontsize=7.5)
    ax.set_yticklabels(names, fontsize=7.5)

    for i in range(n + 1):
        ax.axhline(y=i - 0.5, color='white', linewidth=1)
        ax.axvline(x=i - 0.5, color='white', linewidth=1)

    return im


def create_figure():
    plt.rcParams.update(NEURIPS_RCPARAMS)

    wr, text, names = build_matrices()

    fig, ax = plt.subplots(figsize=(8, 7))
    im = plot_heatmap(ax, wr, text, names)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Win Rate (%)', fontsize=10)
    cbar.set_ticks([0, 25, 50, 75, 100])
    cbar.ax.tick_params(labelsize=8)

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
