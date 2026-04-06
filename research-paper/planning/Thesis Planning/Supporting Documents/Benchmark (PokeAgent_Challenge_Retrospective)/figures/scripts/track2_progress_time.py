#!/usr/bin/env python3
"""
Track 2 Milestone Progress vs. Wall-Clock Time and Cumulative Steps.

Generates a two-panel step-line plot showing milestone completion over time
(left) and over cumulative steps (right) for top-performing teams in the
Pokémon Emerald speedrunning track.

NeurIPS publication ready: Times font, colorblind-friendly colors, clean axes.
"""

import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(__file__).parent.parent.parent / "PokeAgentCompetition_Data" / "processed_data"
OUTPUT_PNG = Path(__file__).parent.parent / "track2_progress_time.png"
OUTPUT_PDF = Path(__file__).parent.parent / "track2_progress_time.pdf"

NEURIPS_RCPARAMS = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'text.usetex': False,
    'axes.linewidth': 0.8,
    'axes.grid': False,
    'lines.linewidth': 2.0,
    'lines.markersize': 5,
}

# Colorblind-friendly palette
TEAM_COLORS = [
    '#E69F00',  # orange (Heatz)
    '#56B4E9',  # sky blue (Hamburg)
    '#009E73',  # green (anthonys)
    '#CC79A7',  # pink (Evelord)
    '#0072B2',  # blue (Deepest)
    '#D55E00',  # vermillion (PokeAgentV2)
    '#999999',  # gray (Human)
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

# Teams to show - ordered by completion time
SHOW_TEAMS = [
    'Heatz', 'Hamburg_PokeRunners', 'anthonys', 'Evelord', 'Deepest', 'PokeAgentV2',
    'Human_WR',
]

TEAM_DISPLAY_NAMES = {
    'Heatz': 'Heatz (SPD)',
    'Hamburg_PokeRunners': 'Hamburg (Rec. PPO)',
    'anthonys': 'anthonys (LLM+A*)',
    'Evelord': 'Evelord (LLM Scaff.)',
    'Deepest': 'Deepest (VLM+Tools)',
    'PokeAgentV2': 'PokéAgent Simple Baseline',
    'Human_WR': 'Human (WR)',
}

# Teams to skip on the steps panel (no step data)
SKIP_STEPS = {'Human_WR', 'anthonys'}


# =============================================================================
# DATA LOADING
# =============================================================================

def parse_time_to_hours(time_str: str) -> float:
    """Convert HH:MM:SS or MM:SS time string to hours."""
    parts = time_str.strip().split(':')
    if len(parts) == 3:
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
    elif len(parts) == 2:
        h, m, s = 0, int(parts[0]), int(parts[1])
    else:
        return 0.0
    return h + m / 60.0 + s / 3600.0


def load_team_data(team_name: str) -> list[dict]:
    """Load milestone data for a team, returning the 15 milestones with time and steps."""
    filepath = DATA_DIR / f"{team_name}_milestones.csv"
    if not filepath.exists():
        return []

    rows = []
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Index CSV rows by milestone name
    csv_data = {}
    for row in rows:
        raw = row['milestone']
        cum_time = parse_time_to_hours(row['cumulative_time'])
        cum_steps = int(row.get('cumulative_steps', 0))
        csv_data[raw] = {'time_hours': cum_time, 'steps': cum_steps}

    # For each of the 15 milestones, take the max time/steps across its CSV keys
    result = []
    for label, csv_keys in MILESTONE_MAP:
        entries = [csv_data[k] for k in csv_keys if k in csv_data]
        if entries:
            result.append({
                'milestone': label,
                'time_hours': max(e['time_hours'] for e in entries),
                'steps': max(e['steps'] for e in entries),
            })

    # Enforce monotonicity
    for i in range(1, len(result)):
        result[i]['time_hours'] = max(result[i]['time_hours'], result[i - 1]['time_hours'])
        result[i]['steps'] = max(result[i]['steps'], result[i - 1]['steps'])

    return result


# =============================================================================
# PLOTTING
# =============================================================================

def plot_panel(ax, teams, x_key, xlabel, show_ylabel=True, show_legend=False):
    """Plot a single panel (time or steps)."""
    milestone_to_idx = {m: i for i, m in enumerate(MILESTONE_LABELS)}

    for i, team in enumerate(teams):
        if x_key == 'steps' and team in SKIP_STEPS:
            continue

        data = load_team_data(team)
        if not data:
            continue

        x_vals = [d[x_key] for d in data]
        indices = [milestone_to_idx[d['milestone']] for d in data]

        display_name = TEAM_DISPLAY_NAMES.get(team, team)
        color = TEAM_COLORS[i % len(TEAM_COLORS)]
        is_human = team == 'Human_WR'
        linestyle = '--' if is_human else '-'
        linewidth = 1.5 if is_human else 2.0
        alpha = 0.7 if is_human else 0.9
        zorder = 1 if is_human else 2

        ax.step(x_vals, indices, where='post', color=color, linestyle=linestyle,
                linewidth=linewidth, alpha=alpha, label=display_name, zorder=zorder)
        ax.plot(x_vals[-1], indices[-1], 'o', color=color, markersize=4, alpha=alpha, zorder=zorder)

    ax.set_xlabel(xlabel, fontweight='bold')
    if show_ylabel:
        ax.set_ylabel('Milestone', fontweight='bold')

    ax.set_yticks(range(len(MILESTONE_LABELS)))
    if show_ylabel:
        ax.set_yticklabels(MILESTONE_LABELS, fontsize=9)
    else:
        ax.tick_params(labelleft=False)
    ax.set_ylim(-0.5, len(MILESTONE_LABELS) - 0.5)

    ax.grid(axis='x', alpha=0.3, linewidth=0.5)
    ax.grid(axis='y', alpha=0.15, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if show_legend:
        ax.legend(loc='lower right', framealpha=0.9, fontsize=8, ncol=1)


def create_figure():
    plt.rcParams.update(NEURIPS_RCPARAMS)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True,
                                    gridspec_kw={'wspace': 0.08})

    # Left panel: time
    plot_panel(ax1, SHOW_TEAMS, 'time_hours', 'Cumulative Time (hours)',
              show_ylabel=True, show_legend=True)
    ax1.set_xlim(-0.05, 5.0)
    ax1.xaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}' if x % 1 else f'{int(x)}'))

    # Right panel: steps
    plot_panel(ax2, SHOW_TEAMS, 'steps', 'Cumulative Steps',
              show_ylabel=False, show_legend=False)
    ax2.set_xlim(-50, None)

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
