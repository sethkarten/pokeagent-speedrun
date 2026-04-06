#!/usr/bin/env python3
"""
Track 2 Milestone Progress vs. Cumulative Steps.

Same layout as track2_progress_time.py but x-axis = cumulative action steps.
Legend reordered by steps-to-completion (fewest first).

NeurIPS publication ready.
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(__file__).parent.parent.parent / "PokeAgentCompetition_Data" / "processed_data"
OUTPUT_PNG = Path(__file__).parent.parent / "track2_progress_steps.png"
OUTPUT_PDF = Path(__file__).parent.parent / "track2_progress_steps.pdf"

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
    'lines.linewidth': 2.0,
    'lines.markersize': 5,
}

TEAM_COLORS = [
    '#0072B2',  # blue (Deepest)
    '#CC79A7',  # pink (Evelord) - was yellow, hard to see on white
    '#E69F00',  # orange (Heatz)
    '#D55E00',  # vermillion (PokeAgentV2)
    '#999999',  # gray (Human)
    '#56B4E9',  # sky blue (Hamburg)
    '#332288',  # indigo
    '#882255',  # wine
    '#44AA99',  # teal
    '#AA4499',  # purple
]

# 15 milestones matching Figure 10 (speedrun route flowchart).
MILESTONE_MAP = [
    ('Littleroot Town',   ['LITTLEROOT_TOWN']),
    ('Route 101',         ['ROUTE_101']),
    ('Starter Chosen',    ['STARTER_CHOSEN']),
    ('Oldale Town',       ['OLDALE_TOWN']),
    ('Rival Battle',      ['ROUTE_103']),
    ('Birch Lab',         ['BIRCH_LAB_VISITED']),
    ('Route 102',         ['ROUTE_102']),
    ('Petalburg City',    ['PETALBURG_CITY']),
    ("Dad's Gym",         ['DAD_FIRST_MEETING', 'GYM_EXPLANATION']),
    ('Route 104 S',       ['ROUTE_104_SOUTH']),
    ('Petalburg Woods',   ['PETALBURG_WOODS']),
    ('Route 104 N',       ['ROUTE_104_NORTH']),
    ('Rustboro City',     ['RUSTBORO_CITY']),
    ('Rustboro Gym',      ['RUSTBORO_GYM_ENTERED']),
    ('Roxanne Defeated',  ['ROXANNE_DEFEATED']),
]

MILESTONE_LABELS = [label for label, _ in MILESTONE_MAP]

# Teams ordered by steps-to-completion (fewest first)
SHOW_TEAMS_BY_STEPS = [
    'Deepest',       # 649 steps
    'Evelord',       # 976 steps
    'Heatz',         # 1608 steps
    'PokeAgentV2',   # 2268 steps
    'Human',         # 2986 steps (partial)
    'Hamburg_PokeRunners',  # 4398 steps
]

TEAM_DISPLAY_NAMES = {
    'Heatz': 'Heatz',
    'Hamburg_PokeRunners': 'Hamburg PokéRunners',
    'anthonys': 'anthonys',
    'Evelord': 'Evelord',
    'Deepest': 'Deepest',
    'PokeAgentV2': 'PokeAgentV2*',
    'Human': 'Human',
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_team_data(team_name: str) -> list[dict]:
    """Load milestone data for a team, returning the 15 figure-10 milestones with steps."""
    filepath = DATA_DIR / f"{team_name}_milestones.csv"
    if not filepath.exists():
        return []

    rows = []
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Index CSV rows by milestone name
    csv_steps = {}
    for row in rows:
        raw = row['milestone']
        csv_steps[raw] = int(row['cumulative_steps'])

    # For each of the 15 milestones, take the max steps across its CSV keys
    result = []
    for label, csv_keys in MILESTONE_MAP:
        steps = [csv_steps[k] for k in csv_keys if k in csv_steps]
        if steps:
            result.append({'milestone': label, 'steps': max(steps)})

    # Enforce monotonicity: x[i] = max(x[i], x[i-1])
    for i in range(1, len(result)):
        result[i]['steps'] = max(result[i]['steps'], result[i - 1]['steps'])

    return result


# =============================================================================
# PLOTTING
# =============================================================================

def create_figure():
    plt.rcParams.update(NEURIPS_RCPARAMS)

    fig, ax = plt.subplots(figsize=(8, 5))

    milestone_to_idx = {m: i for i, m in enumerate(MILESTONE_LABELS)}

    for i, team in enumerate(SHOW_TEAMS_BY_STEPS):
        data = load_team_data(team)
        if not data:
            continue

        # Skip teams with all-zero steps (anthonys has 0 steps in data)
        steps_vals = [d['steps'] for d in data]
        if max(steps_vals) == 0:
            continue

        indices = [milestone_to_idx[d['milestone']] for d in data]

        display_name = TEAM_DISPLAY_NAMES.get(team, team)
        color = TEAM_COLORS[i % len(TEAM_COLORS)]
        linestyle = '--' if team == 'Human' else '-'
        linewidth = 1.5 if team == 'Human' else 2.0
        alpha = 0.7 if team == 'Human' else 0.9
        zorder = 1 if team == 'Human' else 2

        ax.step(steps_vals, indices, where='post', color=color, linestyle=linestyle,
                linewidth=linewidth, alpha=alpha, label=display_name, zorder=zorder)
        ax.plot(steps_vals[-1], indices[-1], 'o', color=color, markersize=5, alpha=alpha, zorder=zorder)

    ax.set_xlabel('Cumulative Steps (actions)', fontweight='bold')
    ax.set_ylabel('Milestone', fontweight='bold')
    ax.set_title('Track 2: Milestone Progress vs. Steps', fontweight='bold')

    ax.set_yticks(range(len(MILESTONE_LABELS)))
    ax.set_yticklabels(MILESTONE_LABELS, fontsize=8)
    ax.set_ylim(-0.5, len(MILESTONE_LABELS) - 0.5)

    ax.set_xlim(-50, 4600)

    ax.grid(axis='x', alpha=0.3, linewidth=0.5)
    ax.grid(axis='y', alpha=0.15, linewidth=0.5)

    ax.legend(loc='lower right', framealpha=0.9, fontsize=9, ncol=1)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

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
