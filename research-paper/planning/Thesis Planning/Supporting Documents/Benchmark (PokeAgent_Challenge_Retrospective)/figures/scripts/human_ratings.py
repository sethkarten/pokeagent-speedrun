#!/usr/bin/env python3
"""
Human Ratings Figure for PokeAgent Challenge Paper.

Generates a bar chart showing Metamon variant performance (GXE)
across generations on the public Pokemon Showdown ladder.

NeurIPS publication ready: Times font, colorblind-friendly colors, large fonts.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_FILE = Path(__file__).parent.parent / "human_ratings.png"
OUTPUT_PDF = Path(__file__).parent.parent / "human_ratings.pdf"

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
    'text.usetex': False,
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

# =============================================================================
# DATA - From Metamon GitHub (github.com/ut-austin-rpl/metamon)
# =============================================================================

# Generations evaluated
GENERATIONS = ['Gen 1', 'Gen 2', 'Gen 3', 'Gen 4', 'Gen 9']

# GXE values for Kakuna (142M parameters) - best model
# Source: Metamon readme and Track 1 text
KAKUNA_GXE = [82, 70, 63, 64, 76]

# Leaderboard threshold (76% GXE required to make top 500)
LEADERBOARD_THRESHOLD = 76

# =============================================================================
# PLOTTING
# =============================================================================

def create_figure():
    """Create the human ratings bar chart."""
    plt.rcParams.update(NEURIPS_RCPARAMS)

    fig, ax = plt.subplots(figsize=(6, 4))

    x = np.arange(len(GENERATIONS))
    width = 0.6

    # Color bars based on whether they exceed threshold
    colors = [COLORS['magenta'] if gxe >= LEADERBOARD_THRESHOLD else COLORS['blue']
              for gxe in KAKUNA_GXE]

    # Create bars
    bars = ax.bar(x, KAKUNA_GXE, width, color=colors, alpha=0.85,
                  edgecolor='white', linewidth=1)

    # Add threshold line
    ax.axhline(y=LEADERBOARD_THRESHOLD, color=COLORS['gray'], linestyle='--',
               linewidth=1.5, label=f'Leaderboard threshold ({LEADERBOARD_THRESHOLD}%)')

    # Add value labels on bars
    for i, (bar, gxe) in enumerate(zip(bars, KAKUNA_GXE)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{gxe}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Styling
    ax.set_ylabel('GXE (%)', fontweight='bold')
    ax.set_xlabel('Format', fontweight='bold')
    ax.set_title('Kakuna (142M) Performance vs. Human Players', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(GENERATIONS)
    ax.set_ylim(0, 95)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['magenta'], alpha=0.85, label='Above threshold'),
        Patch(facecolor=COLORS['blue'], alpha=0.85, label='Below threshold'),
        plt.Line2D([0], [0], color=COLORS['gray'], linestyle='--',
                   linewidth=1.5, label=f'Leaderboard ({LEADERBOARD_THRESHOLD}% GXE)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig


def main():
    print("Creating human ratings figure...")

    fig = create_figure()

    print(f"Saving to: {OUTPUT_FILE}")
    fig.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saving to: {OUTPUT_PDF}")
    fig.savefig(OUTPUT_PDF, bbox_inches='tight', facecolor='white')

    plt.close(fig)
    print("Done!")


if __name__ == "__main__":
    main()
