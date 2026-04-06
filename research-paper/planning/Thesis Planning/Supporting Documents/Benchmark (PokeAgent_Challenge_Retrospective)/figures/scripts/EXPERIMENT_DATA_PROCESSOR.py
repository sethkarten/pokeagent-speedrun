#!/usr/bin/env python3
"""
Experiment Data Processor — Methodology-agnostic Pipeline for PokeAgent Experiment Analysis.

Orchestrates loading cumulative_metrics.json from experiment directories,
processes objective completion data, and generates publication-ready figures.

Usage examples:
    # Line graph: milestone progress vs steps for all Gemini 3-Flash experiments
    python EXPERIMENT_DATA_PROCESSOR.py \
        --models gemini \
        --variants 3-flash \
        --experiment-settings all_tools \
        --experiment-nums 1 2 \
        --figure-type line \
        --x-axis steps \
        --y-axis milestones \
        --output figures/output/gemini_flash_progress

    # Line graph: milestone progress vs time
    python EXPERIMENT_DATA_PROCESSOR.py \
        --models claude gemini \
        --variants 3-flash 3-pro sonnet-4-5 \
        --experiment-settings all_tools \
        --experiment-nums 1 2 \
        --figure-type line \
        --x-axis time \
        --y-axis milestones \
        --output figures/output/all_experiments_time_vs_milestones

    # Bar chart: tokens per milestone across models
    python EXPERIMENT_DATA_PROCESSOR.py \
        --models gemini claude \
        --variants 3-flash 3-pro sonnet-4-5 \
        --experiment-settings all_tools \
        --experiment-nums 1 \
        --figure-type bar \
        --x-axis milestones \
        --y-axis tokens \
        --output figures/output/model_comparison_tokens

    # Bar chart: cost per milestone (split basis)
    python EXPERIMENT_DATA_PROCESSOR.py \
        --models gemini claude \
        --variants 3-flash 3-pro sonnet-4-5 \
        --experiment-settings all_tools \
        --experiment-nums 1 \
        --figure-type bar \
        --x-axis milestones \
        --y-axis split_cost \
        --output figures/output/model_comparison_cost

    # Line chart: cumulative cost vs steps / time
    python EXPERIMENT_DATA_PROCESSOR.py \
        --models gemini claude \
        --experiment-settings all_tools \
        --experiment-nums 1 2 \
        --figure-type line \
        --x-axis steps --y-axis cost \
        --output figures/output/cost_vs_steps
    python EXPERIMENT_DATA_PROCESSOR.py \
        --figure-type line --x-axis time --y-axis cost \
        --output figures/output/cost_vs_time ...

    # Line chart: milestone progress vs cumulative cost
    python EXPERIMENT_DATA_PROCESSOR.py \
        --figure-type line --x-axis cost --y-axis milestones \
        --output figures/output/milestones_vs_cost ...

    # Add --log-x or --log-y for continuous axes (steps, time, cost, tokens)

    # Cap data to time window (like milestone_confidence.py)
    python EXPERIMENT_DATA_PROCESSOR.py ... --cap-hours 2

    # Aggregate: coalesce experiments into min-max average per (model, variant, prompt_opt_freq)
    python EXPERIMENT_DATA_PROCESSOR.py ... --aggregate

    # Include Prompt Optimization (under baseline, adjacent to all_tools)
    # Structure: baseline/prompt_optimization/{model}/{variant}/{freq} Steps/Experiment N/
    python EXPERIMENT_DATA_PROCESSOR.py ... --experiment-settings all_tools prompt_optimization
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# =============================================================================
# CONSTANTS
# =============================================================================

EXPERIMENT_DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "experiment_data"

# ── Objective-ID → Milestone label mapping ──────────────────────────────────
# The 15 key milestones we track.  Each maps a human-readable label to the
# objective_id found in cumulative_metrics.json → objectives[].objective_id.
MILESTONE_OBJECTIVES: list[tuple[str, str]] = [
    ("Littleroot Town",  "tutorial_000"),
    ("Route 101",        "tutorial_006"),
    ("Starter Chosen",   "tutorial_007"),
    ("Oldale Town",      "early_011"),
    ("Rival Battle",     "early_014"),
    ("Birch Lab",        "early_015"),
    ("Route 102",        "petalburg_019"),
    ("Petalburg City",   "petalburg_020"),
    ("Dad's Gym",        "petalburg_021"),
    ("Route 104 S",      "petalburg_023"),
    ("Petalburg Woods",  "petalburg_024"),
    ("Route 104 N",      "petalburg_028"),
    ("Rustboro City",    "petalburg_029"),
    ("Rustboro Gym",     "rustboro_031"),
    ("Roxanne Defeated", "rustboro_033"),
]

MILESTONE_LABELS: list[str] = [label for label, _ in MILESTONE_OBJECTIVES]
MILESTONE_OBJ_IDS: list[str] = [obj_id for _, obj_id in MILESTONE_OBJECTIVES]

# CLI runs use milestones[] with milestone_id; map to canonical objective_id for comparison
CLI_MILESTONE_TO_OBJECTIVE: dict[str, str] = {
    "LITTLEROOT_TOWN":       "tutorial_000",
    "ROUTE_101":             "tutorial_006",
    "STARTER_CHOSEN":        "tutorial_007",
    "OLDALE_TOWN":           "early_011",
    "RIVAL_BATTLE_WON":      "early_014",
    "RECEIVED_POKEDEX":      "early_015",
    "ROUTE_102":             "petalburg_019",
    "PETALBURG_CITY":        "petalburg_020",
    "DAD_FIRST_MEETING":     "petalburg_021",
    "ROUTE_104_SOUTH":       "petalburg_023",
    "PETALBURG_WOODS":       "petalburg_024",
    "ROUTE_104_NORTH":       "petalburg_028",
    "RUSTBORO_CITY":         "petalburg_029",
    "RUSTBORO_GYM_ENTERED":  "rustboro_031",
    "ROXANNE_DEFEATED":      "rustboro_033",
}

# ── Valid axis combinations per figure type ─────────────────────────────────
# Keys: (figure_type, x_axis, y_axis)
VALID_AXIS_COMBOS: dict[str, list[tuple[str, str]]] = {
    "line": [
        ("steps", "milestones"),          # step-line: milestone progress vs cumulative steps
        ("time", "milestones"),           # step-line: milestone progress vs cumulative time
        ("cost", "milestones"),           # step-line: milestone progress vs cumulative cost
        ("steps", "tokens"),              # line: cumulative tokens vs steps
        ("steps", "time"),                # line: cumulative time vs steps
        ("steps", "cost"),                # line: cumulative cost vs steps
        ("time", "cost"),                 # line: cumulative cost vs time
    ],
    "bar": [
        ("milestones", "tokens"),         # bar: tokens consumed per milestone
        ("milestones", "steps"),          # bar: steps taken per milestone
        ("milestones", "time"),           # bar: time taken per milestone
        ("milestones", "split_cost"),     # bar: cost per milestone (split basis)
        ("milestones", "split_tokens"),   # bar: split tokens per milestone
        ("milestones", "split_steps"),    # bar: split steps per milestone
        ("milestones", "split_time"),     # bar: split time per milestone
    ],
}

# ── NeurIPS-style matplotlib params ─────────────────────────────────────────
NEURIPS_RCPARAMS: dict[str, Any] = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
    "text.usetex": False,
    "axes.linewidth": 0.8,
    "axes.grid": False,
    "lines.linewidth": 2.0,
    "lines.markersize": 5,
}

# Colorblind-friendly palette (Wong 2011 + extensions)
COLORS: list[str] = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#CC79A7",  # pink
    "#D55E00",  # vermillion
    "#56B4E9",  # sky blue
    "#332288",  # indigo
    "#882255",  # wine
    "#44AA99",  # teal
    "#AA4499",  # purple
    "#999999",  # gray
    "#117733",  # forest
    "#661100",  # brown
    "#DDCC77",  # sand
    "#88CCEE",  # light blue
]


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ObjectiveData:
    """Processed data for a single completed objective."""
    objective_id: str
    category: str
    objective_index: int
    timestamp: float
    cumulative_steps: int
    cumulative_total_tokens: int
    cumulative_prompt_tokens: int
    cumulative_completion_tokens: int
    cumulative_cached_tokens: int
    split_steps: int
    split_total_tokens: int
    split_prompt_tokens: int
    split_completion_tokens: int
    split_cached_tokens: int
    split_time_seconds: float


@dataclass
class ExperimentData:
    """All data loaded from a single experiment run."""
    model: str
    variant: str
    setting: str
    experiment_num: int
    experiment_suffix: str          # e.g. "" or "_incomplete"
    prompt_opt_freq: int | None     # None = baseline, N = optimize every N steps (separate series)
    label: str                      # display label for legends
    run_id: str
    total_tokens: int
    total_cost: float               # total $ from cumulative_metrics.json
    total_actions: int
    total_run_time: float
    total_llm_calls: int
    start_time: float
    objectives: list[ObjectiveData] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# LAYER 1: CLI ARGUMENT PARSING
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PokeAgent Experiment Data Processor — load metrics & generate figures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Data selection ──────────────────────────────────────────────────────
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model families to include (e.g. gemini claude).",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        required=True,
        help="Model variants to include (e.g. 3-flash 3-pro sonnet-4-5).",
    )
    parser.add_argument(
        "--experiment-settings",
        nargs="+",
        required=True,
        help="Experiment settings (e.g. all_tools vision_only self_optimization).",
    )
    parser.add_argument(
        "--experiment-nums",
        nargs="+",
        type=int,
        required=True,
        help="Experiment numbers to include (e.g. 1 2 3).",
    )
    parser.add_argument(
        "--method",
        default="baseline",
        help="Methodology for agent execution (default: baseline... eventually will support prompt-evolution, cli agents).",
    )

    # ── Figure configuration ────────────────────────────────────────────────
    parser.add_argument(
        "--figure-type",
        required=True,
        choices=["line", "bar"],
        help="Type of figure to generate.",
    )
    parser.add_argument(
        "--x-axis",
        required=True,
        help="Independent variable for x-axis (e.g. steps, milestones).",
    )
    parser.add_argument(
        "--y-axis",
        required=True,
        help="Dependent variable for y-axis (e.g. milestones, tokens, steps, time, cost, split_cost, split_tokens, split_steps, split_time).",
    )

    # ── Output ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--output",
        required=True,
        help="Output file path (without extension).",
    )
    parser.add_argument(
        "--format",
        nargs="+",
        default=["png", "pdf"],
        choices=["png", "pdf"],
        help="Output formats (default: png pdf).",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Custom figure title. If not set, auto-generated from args.",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Override experiment_data root directory.",
    )
    parser.add_argument(
        "--log-x",
        action="store_true",
        help="Use logarithmic scale for x-axis (applies only when x is steps, time, or cost).",
    )
    parser.add_argument(
        "--log-y",
        action="store_true",
        help="Use logarithmic scale for y-axis (applies only when y is tokens, time, or cost).",
    )
    parser.add_argument(
        "--cap-hours",
        type=float,
        default=None,
        help="Only include milestone data within this many hours (truncates, does not drop runs). Same as milestone_confidence.py.",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Coalesce experiments into min-max average per (model, variant, prompt_opt_freq). One line/bar per group with shaded band.",
    )

    return parser


def validate_axes(args: argparse.Namespace) -> None:
    """Validate that the (figure_type, x_axis, y_axis) combination is valid."""
    combo = (args.x_axis, args.y_axis)
    valid = VALID_AXIS_COMBOS.get(args.figure_type, [])
    if combo not in valid:
        valid_strs = [f"  --x-axis {x} --y-axis {y}" for x, y in valid]
        print(
            f"ERROR: Invalid axis combination --x-axis {args.x_axis} --y-axis {args.y_axis} "
            f"for --figure-type {args.figure_type}.\n"
            f"Valid combinations for '{args.figure_type}':\n" + "\n".join(valid_strs),
            file=sys.stderr,
        )
        sys.exit(1)


# =============================================================================
# LAYER 2: DATA DISCOVERY & LOADING
# =============================================================================

def discover_experiment_dirs(
    data_root: Path,
    method: str,
    settings: list[str],
    models: list[str],
    variants: list[str],
    experiment_nums: list[int],
) -> list[dict[str, Any]]:
    """
    Walk the directory tree and discover matching experiment directories.

    Returns a list of dicts with keys:
        model, variant, setting, experiment_num, experiment_suffix, cache_dir
    """
    discovered: list[dict[str, Any]] = []

    for setting in settings:
        if setting == "cli":
            # CLI: baseline/cli/{agent_type}/experiment_N (no model level)
            for variant in variants:
                agent_dir = data_root / method / setting / variant
                if not agent_dir.is_dir():
                    print(f"  [SKIP] CLI agent dir not found: {agent_dir}")
                    continue
                for exp_num in experiment_nums:
                    pattern = f"experiment_{exp_num}*"
                    matches = sorted(agent_dir.glob(pattern))
                    if not matches:
                        print(f"  [SKIP] No experiment_{exp_num}* in {agent_dir}")
                        continue
                    for exp_dir in matches:
                        if not exp_dir.is_dir():
                            continue
                        dir_name = exp_dir.name
                        prefix = f"experiment_{exp_num}"
                        suffix = dir_name[len(prefix):]
                        cache_dirs = sorted(exp_dir.glob("pokeagent_cache_*"))
                        if not cache_dirs:
                            print(f"  [SKIP] No pokeagent_cache_* in {exp_dir}")
                            continue
                        for cache_dir in cache_dirs:
                            if not cache_dir.is_dir():
                                continue
                            discovered.append({
                                "model": "cli",
                                "variant": variant,
                                "setting": setting,
                                "experiment_num": exp_num,
                                "experiment_suffix": suffix,
                                "prompt_opt_freq": None,
                                "cache_dir": cache_dir,
                            })
            continue

        # Prompt Optimization: baseline/prompt_optimization/{model}/{variant}/{freq} Steps/Experiment {n}/pokeagent_cache_*/
        if setting.lower().replace(" ", "_") in ("prompt_optimization", "prompt_opt"):
            prompt_opt_base = data_root / method / "prompt_optimization"
            if not prompt_opt_base.is_dir():
                prompt_opt_base = data_root / method / "Prompt Optimization"
            if prompt_opt_base.is_dir():
                for model in models:
                    model_dir = prompt_opt_base / model
                    if not model_dir.is_dir():
                        continue
                    for variant in variants:
                        variant_dir = model_dir / variant
                        if not variant_dir.is_dir():
                            continue
                        for freq_dir in sorted(variant_dir.iterdir()):
                            if not freq_dir.is_dir():
                                continue
                            freq_match = re.match(r"(\d+)\s*Steps?", freq_dir.name, re.I)
                            freq = int(freq_match.group(1)) if freq_match else None
                            if freq is None:
                                continue
                            for exp_dir in sorted(freq_dir.iterdir()):
                                if not exp_dir.is_dir():
                                    continue
                                exp_match = re.match(r"Experiment\s*(\d+)", exp_dir.name, re.I)
                                if not exp_match:
                                    exp_match = re.match(r"experiment_(\d+)", exp_dir.name, re.I)
                                if not exp_match:
                                    continue
                                exp_num = int(exp_match.group(1))
                                if exp_num not in experiment_nums:
                                    continue
                                for cache_dir in sorted(exp_dir.glob("pokeagent_cache_*")):
                                    if not cache_dir.is_dir():
                                        continue
                                    discovered.append({
                                        "model": model,
                                        "variant": variant,
                                        "setting": "prompt_optimization",
                                        "experiment_num": exp_num,
                                        "experiment_suffix": f"freq-{freq}",
                                        "prompt_opt_freq": freq,
                                        "cache_dir": cache_dir,
                                    })
            continue

        for model in models:
            model_dir = data_root / method / setting / model
            if not model_dir.is_dir():
                print(f"  [SKIP] Model dir not found: {model_dir}")
                continue

            for variant in variants:
                variant_dir = model_dir / variant
                if not variant_dir.is_dir():
                    # Some models (like claude) may not have variant subdirs —
                    # check if experiments live directly under model_dir
                    continue

                for exp_num in experiment_nums:
                    # Glob for experiment_N* to catch suffixes like _incomplete
                    pattern = f"experiment_{exp_num}*"
                    matches = sorted(variant_dir.glob(pattern))
                    if not matches:
                        print(f"  [SKIP] No experiment_{exp_num}* in {variant_dir}")
                        continue

                    for exp_dir in matches:
                        if not exp_dir.is_dir():
                            continue

                        # Extract suffix (e.g. "_incomplete" from "experiment_1_incomplete")
                        dir_name = exp_dir.name
                        prefix = f"experiment_{exp_num}"
                        suffix = dir_name[len(prefix):]

                        # Find pokeagent_cache_* directories inside
                        cache_dirs = sorted(exp_dir.glob("pokeagent_cache_*"))
                        if not cache_dirs:
                            print(f"  [SKIP] No pokeagent_cache_* in {exp_dir}")
                            continue

                        for cache_dir in cache_dirs:
                            if not cache_dir.is_dir():
                                continue
                            discovered.append({
                                "model": model,
                                "variant": variant,
                                "setting": setting,
                                "experiment_num": exp_num,
                                "experiment_suffix": suffix,
                                "prompt_opt_freq": None,
                                "cache_dir": cache_dir,
                            })

    return discovered


def load_cumulative_metrics(cache_dir: Path) -> dict[str, Any] | None:
    """Load and return parsed cumulative_metrics.json from a cache directory."""
    metrics_path = cache_dir / "cumulative_metrics.json"
    if not metrics_path.exists():
        print(f"  [WARN] Missing cumulative_metrics.json in {cache_dir}")
        return None

    with open(metrics_path) as f:
        return json.load(f)


def parse_objectives(raw_objectives: list[dict[str, Any]]) -> list[ObjectiveData]:
    """Parse raw objective dicts from cumulative_metrics.json into ObjectiveData objects."""
    objectives: list[ObjectiveData] = []
    for obj in raw_objectives:
        objectives.append(ObjectiveData(
            objective_id=obj["objective_id"],
            category=obj["category"],
            objective_index=obj["objective_index"],
            timestamp=obj["timestamp"],
            cumulative_steps=obj["cumulative_steps"],
            cumulative_total_tokens=obj["cumulative_total_tokens"],
            cumulative_prompt_tokens=obj["cumulative_prompt_tokens"],
            cumulative_completion_tokens=obj["cumulative_completion_tokens"],
            cumulative_cached_tokens=obj.get("cumulative_cached_tokens", 0),
            split_steps=obj["split_steps"],
            split_total_tokens=obj["split_total_tokens"],
            split_prompt_tokens=obj["split_prompt_tokens"],
            split_completion_tokens=obj["split_completion_tokens"],
            split_cached_tokens=obj.get("split_cached_tokens", 0),
            split_time_seconds=obj["split_time_seconds"],
        ))
    return objectives


def parse_milestones_from_cli(raw_milestones: list[dict[str, Any]]) -> list[ObjectiveData]:
    """Parse CLI milestones[] into ObjectiveData (canonical objective_id) for comparison."""
    objectives: list[ObjectiveData] = []
    for idx, m in enumerate(raw_milestones):
        mid = m.get("milestone_id")
        if mid not in CLI_MILESTONE_TO_OBJECTIVE:
            continue
        oid = CLI_MILESTONE_TO_OBJECTIVE[mid]
        objectives.append(ObjectiveData(
            objective_id=oid,
            category="cli",
            objective_index=idx,
            timestamp=m["timestamp"],
            cumulative_steps=m["cumulative_steps"],
            cumulative_total_tokens=m["cumulative_total_tokens"],
            cumulative_prompt_tokens=m["cumulative_prompt_tokens"],
            cumulative_completion_tokens=m["cumulative_completion_tokens"],
            cumulative_cached_tokens=m.get("cumulative_cached_tokens", 0),
            split_steps=m["split_steps"],
            split_total_tokens=m["split_total_tokens"],
            split_prompt_tokens=m["split_prompt_tokens"],
            split_completion_tokens=m["split_completion_tokens"],
            split_cached_tokens=m.get("split_cached_tokens", 0),
            split_time_seconds=m["split_time_seconds"],
        ))
    return objectives


def build_experiment_label(
    model: str, variant: str, experiment_num: int, suffix: str,
    prompt_opt_freq: int | None = None,
) -> str:
    """Generate a display label for an experiment."""
    if model == "cli":
        variant_display = variant.replace("-", " ").title()
        label = f"{variant_display} Exp {experiment_num}"
    else:
        model_display = model.capitalize()
        variant_display = variant.replace("-", " ").title()
        label = f"{model_display} {variant_display} Exp {experiment_num}"
    if prompt_opt_freq is not None:
        label += f" (opt every {prompt_opt_freq})"
    if suffix:
        label += f" ({suffix.strip('_')})"
    return label


def load_all_experiments(
    data_root: Path,
    method: str,
    settings: list[str],
    models: list[str],
    variants: list[str],
    experiment_nums: list[int],
) -> list[ExperimentData]:
    """
    Discover and load all matching experiments.

    Returns a list of ExperimentData objects with objectives populated.
    """
    print(f"Discovering experiments in: {data_root}")
    discoveries = discover_experiment_dirs(
        data_root, method, settings, models, variants, experiment_nums,
    )

    if not discoveries:
        print("ERROR: No experiments found matching the given filters.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(discoveries)} experiment(s). Loading data...\n")

    experiments: list[ExperimentData] = []
    for disc in discoveries:
        raw = load_cumulative_metrics(disc["cache_dir"])
        if raw is None:
            continue

        raw_objectives = raw.get("objectives", [])
        raw_milestones = raw.get("milestones", [])
        if raw_objectives:
            objectives = parse_objectives(raw_objectives)
        elif raw_milestones:
            objectives = parse_milestones_from_cli(raw_milestones)
        else:
            print(f"  [WARN] No objectives/milestones in {disc['cache_dir'].name}")
            continue
        metadata = raw.get("metadata", {})
        run_id = metadata.get("run_id", disc["cache_dir"].name)

        label = build_experiment_label(
            disc["model"], disc["variant"], disc["experiment_num"], disc["experiment_suffix"],
            prompt_opt_freq=disc.get("prompt_opt_freq"),
        )

        exp = ExperimentData(
            model=disc["model"],
            variant=disc["variant"],
            setting=disc["setting"],
            experiment_num=disc["experiment_num"],
            experiment_suffix=disc["experiment_suffix"],
            prompt_opt_freq=disc.get("prompt_opt_freq"),
            label=label,
            run_id=run_id,
            total_tokens=raw.get("total_tokens", 0),
            total_cost=raw.get("total_cost", 0.0),
            total_actions=raw.get("total_actions", 0),
            total_run_time=raw.get("total_run_time", 0.0),
            total_llm_calls=raw.get("total_llm_calls", 0),
            start_time=raw.get("start_time", 0.0),
            objectives=objectives,
            metadata=metadata,
        )
        experiments.append(exp)
        n_milestones = sum(
            1 for o in objectives if o.objective_id in MILESTONE_OBJ_IDS
        )
        print(f"  Loaded: {label} — {len(objectives)} objectives, {n_milestones}/15 milestones")

    print(f"\nSuccessfully loaded {len(experiments)} experiment(s).\n")
    return experiments


# =============================================================================
# LAYER 3: DATA PROCESSING
# =============================================================================

def extract_milestone_data(experiment: ExperimentData) -> list[dict[str, Any]]:
    """
    Extract the 15 milestone objectives from an experiment.

    Returns a list of dicts (in milestone order) with keys:
        label, objective_id, cumulative_steps, cumulative_total_tokens,
        split_steps, split_total_tokens, split_time_seconds, timestamp
    Only includes milestones that were actually completed.
    """
    # Index objectives by ID for fast lookup
    obj_by_id: dict[str, ObjectiveData] = {
        o.objective_id: o for o in experiment.objectives
    }

    milestones: list[dict[str, Any]] = []
    for label, obj_id in MILESTONE_OBJECTIVES:
        obj = obj_by_id.get(obj_id)
        if obj is None:
            continue  # milestone not reached in this experiment
        milestones.append({
            "label": label,
            "objective_id": obj_id,
            "cumulative_steps": obj.cumulative_steps,
            "cumulative_total_tokens": obj.cumulative_total_tokens,
            "cumulative_prompt_tokens": obj.cumulative_prompt_tokens,
            "cumulative_completion_tokens": obj.cumulative_completion_tokens,
            "split_steps": obj.split_steps,
            "split_total_tokens": obj.split_total_tokens,
            "split_prompt_tokens": obj.split_prompt_tokens,
            "split_completion_tokens": obj.split_completion_tokens,
            "split_time_seconds": obj.split_time_seconds,
            "timestamp": obj.timestamp,
        })

    # Enforce monotonicity on cumulative_steps (each >= previous)
    for i in range(1, len(milestones)):
        milestones[i]["cumulative_steps"] = max(
            milestones[i]["cumulative_steps"],
            milestones[i - 1]["cumulative_steps"],
        )
        milestones[i]["cumulative_total_tokens"] = max(
            milestones[i]["cumulative_total_tokens"],
            milestones[i - 1]["cumulative_total_tokens"],
        )

    return milestones


def compute_cumulative_time(experiment: ExperimentData, milestones: list[dict[str, Any]]) -> None:
    """Add cumulative_time_hours to each milestone dict (time since run start).

    Relies on timestamps in cumulative_metrics.json being gap-compressed
    (idle periods between multi-part runs removed by merge_cumulative_metrics.py),
    so (timestamp - start_time) reflects active run time.
    """
    start = experiment.start_time
    for m in milestones:
        m["cumulative_time_hours"] = (m["timestamp"] - start) / 3600.0


def filter_milestones_by_cap_hours(
    milestones: list[dict[str, Any]], cap_hours: float
) -> list[dict[str, Any]]:
    """Return only milestones reached within cap_hours. Truncates, does not drop runs."""
    return [m for m in milestones if m.get("cumulative_time_hours", 0) <= cap_hours]


def compute_cost_allocations(experiment: ExperimentData, milestones: list[dict[str, Any]]) -> None:
    """Add split_cost and cumulative_cost to each milestone via proportional token allocation."""
    total_tokens = experiment.total_tokens
    total_cost = experiment.total_cost
    if total_tokens <= 0:
        for m in milestones:
            m["split_cost"] = 0.0
            m["cumulative_cost"] = 0.0
        return
    for m in milestones:
        m["split_cost"] = (m["split_total_tokens"] / total_tokens) * total_cost
        m["cumulative_cost"] = (m["cumulative_total_tokens"] / total_tokens) * total_cost


# =============================================================================
# LAYER 4: FIGURE GENERATION
# =============================================================================

def _apply_style() -> None:
    plt.rcParams.update(NEURIPS_RCPARAMS)


def _clean_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _lighten_color(hex_color: str, factor: float) -> str:
    """Blend hex color with white. factor=1.0 gives full color, factor=0.5 gives 50% blend."""
    rgb = np.array(mcolors.to_rgb(hex_color))
    white = np.ones(3)
    blended = (1 - factor) * white + factor * rgb
    return mcolors.to_hex(np.clip(blended, 0, 1))


# Linestyles for experiments within same (model, variant): solid, dashed, dotted, dashdot
LINE_STYLES: list[str] = ["-", "--", ":", "-."]
# Hatch patterns for bar charts (same grouping)
HATCH_STYLES: list[str] = ["", "//", "..", "xx"]


def assign_colors_and_styles(experiments: list[ExperimentData]) -> list[tuple[str, str, str]]:
    """
    Assign (color, linestyle, hatch) per experiment with two-level hierarchy:

    1. Sub-model/variant (e.g. Gemini 3-flash vs Gemini 3-pro): distinct colors from palette
    2. Experiment (Exp 1 vs Exp 2): same color, different linestyle/hatch
    """
    variant_to_color: dict[tuple[str, str], str] = {}  # (model, variant) -> hex color
    variant_exp_rank: dict[tuple[str, str], int] = {}  # (model, variant) -> next exp rank
    result: list[tuple[str, str, str]] = []

    for exp in experiments:
        model, variant = exp.model, exp.variant
        key = (model, variant)

        if key not in variant_to_color:
            idx = len(variant_to_color) % len(COLORS)
            variant_to_color[key] = COLORS[idx]
            variant_exp_rank[key] = 0

        exp_rank = variant_exp_rank[key]
        variant_exp_rank[key] += 1

        color = variant_to_color[key]
        linestyle = LINE_STYLES[exp_rank % len(LINE_STYLES)]
        hatch = HATCH_STYLES[exp_rank % len(HATCH_STYLES)]
        result.append((color, linestyle, hatch))

    return result


def _group_experiments_for_aggregation(
    experiments: list[ExperimentData],
) -> list[tuple[tuple[str, str, int | None], list[ExperimentData]]]:
    """Group by (model, variant, prompt_opt_freq). Each group = separate series."""
    groups: dict[tuple[str, str, int | None], list[ExperimentData]] = {}
    for exp in experiments:
        key = (exp.model, exp.variant, exp.prompt_opt_freq)
        groups.setdefault(key, []).append(exp)
    # Sort by (model, variant, prompt_opt_freq) with None last
    def _sort_key(item: tuple) -> tuple:
        k, _ = item
        m, v, f = k
        return (m, v, (f if f is not None else 999999))

    return [(k, v) for k, v in sorted(groups.items(), key=_sort_key)]


def _aggregate_group_label(model: str, variant: str, prompt_opt_freq: int | None) -> str:
    """Label for an aggregated group."""
    model_d = model.capitalize()
    variant_d = variant.replace("-", " ").title()
    base = f"{model_d} {variant_d}"
    if prompt_opt_freq is not None:
        base += f" (opt every {prompt_opt_freq})"
    return base


def generate_line_figure(
    experiments: list[ExperimentData],
    x_axis: str,
    y_axis: str,
    title: str | None,
    log_x: bool = False,
    log_y: bool = False,
    cap_hours: float | None = None,
    aggregate: bool = False,
) -> plt.Figure:
    """
    Generate a step-line figure.

    Supported combos:
        x=steps, y=milestones  — milestone progress vs cumulative steps
        x=time, y=milestones   — milestone progress vs cumulative time
        x=cost, y=milestones   — milestone progress vs cumulative cost
        x=steps, y=tokens      — cumulative tokens vs steps
        x=steps, y=time        — cumulative time vs steps
        x=steps, y=cost        — cumulative cost vs steps
        x=time, y=cost         — cumulative cost vs time
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    milestone_to_idx = {m: i for i, m in enumerate(MILESTONE_LABELS)}
    exp_styles = assign_colors_and_styles(experiments)

    if aggregate:
        # Coalesce into min-max average per (model, variant, prompt_opt_freq)
        groups = _group_experiments_for_aggregation(experiments)
        group_colors = {g[0]: COLORS[i % len(COLORS)] for i, g in enumerate(groups)}

        x_key = "cumulative_steps" if x_axis == "steps" else "cumulative_time_hours" if x_axis == "time" else "cumulative_cost"
        y_key = "cumulative_total_tokens" if y_axis == "tokens" else "cumulative_time_hours" if y_axis == "time" else "cumulative_cost"

        for (model, variant, prompt_opt_freq), group_exps in groups:
            # Build matrices: rows = runs, cols = milestones (by objective_id order)
            n_milestones = len(MILESTONE_OBJ_IDS)
            matrix_x = np.full((len(group_exps), n_milestones), np.nan)
            matrix_y = np.full((len(group_exps), n_milestones), np.nan)

            for ri, exp in enumerate(group_exps):
                milestones = extract_milestone_data(exp)
                if not milestones:
                    continue
                compute_cumulative_time(exp, milestones)
                if cap_hours is not None:
                    milestones = filter_milestones_by_cap_hours(milestones, cap_hours)
                if not milestones:
                    continue
                compute_cost_allocations(exp, milestones)

                for m in milestones:
                    mi = MILESTONE_OBJ_IDS.index(m["objective_id"]) if m["objective_id"] in MILESTONE_OBJ_IDS else -1
                    if mi >= 0:
                        matrix_x[ri, mi] = m[x_key]
                        matrix_y[ri, mi] = m[y_key] if y_axis != "milestones" else mi

            with np.errstate(all="ignore"):
                if y_axis == "milestones":
                    means_x = np.nanmean(matrix_x, axis=0)
                    mins_x = np.nanmin(matrix_x, axis=0)
                    maxs_x = np.nanmax(matrix_x, axis=0)
                    means_y = np.arange(n_milestones)
                    mins_y = means_y
                    maxs_y = means_y
                else:
                    means_x = np.nanmean(matrix_x, axis=0)
                    mins_x = np.nanmin(matrix_x, axis=0)
                    maxs_x = np.nanmax(matrix_x, axis=0)
                    means_y = np.nanmean(matrix_y, axis=0)
                    mins_y = np.nanmin(matrix_y, axis=0)
                    maxs_y = np.nanmax(matrix_y, axis=0)

            valid_mask = ~np.isnan(means_x) & ~np.isnan(means_y)
            if not valid_mask.any():
                continue

            last_valid = int(np.max(np.where(valid_mask)[0]))
            valid_idx = np.where(valid_mask)[0]
            if len(valid_idx) >= 2:
                fill_idx = np.arange(last_valid + 1)
                means_x_f = np.interp(fill_idx, valid_idx, means_x[valid_idx])
                mins_x_f = np.interp(fill_idx, valid_idx, mins_x[valid_idx])
                maxs_x_f = np.interp(fill_idx, valid_idx, maxs_x[valid_idx])
                means_y_f = np.interp(fill_idx, valid_idx, means_y[valid_idx])
                mins_y_f = np.interp(fill_idx, valid_idx, mins_y[valid_idx])
                maxs_y_f = np.interp(fill_idx, valid_idx, maxs_y[valid_idx])
            else:
                means_x_f = means_x[: last_valid + 1]
                mins_x_f = mins_x[: last_valid + 1]
                maxs_x_f = maxs_x[: last_valid + 1]
                means_y_f = means_y[: last_valid + 1]
                mins_y_f = mins_y[: last_valid + 1]
                maxs_y_f = maxs_y[: last_valid + 1]

            color = group_colors[(model, variant, prompt_opt_freq)]
            label = _aggregate_group_label(model, variant, prompt_opt_freq)

            if y_axis == "milestones":
                ax.plot(means_x_f, means_y_f, color=color, linewidth=2.0, alpha=0.9, label=label,
                    marker="o", markersize=6, zorder=2)
                if len(group_exps) > 1:
                    ax.fill_betweenx(means_y_f, mins_x_f, maxs_x_f, color=color, alpha=0.15, zorder=1)
            else:
                ax.plot(means_x_f, means_y_f, color=color, linewidth=2.0, alpha=0.9, label=label, marker="o", markersize=4, zorder=2)
                if len(group_exps) > 1:
                    ax.fill_between(means_x_f, mins_y_f, maxs_y_f, color=color, alpha=0.15, zorder=1)

    else:
        for i, exp in enumerate(experiments):
            milestones = extract_milestone_data(exp)
            if not milestones:
                print(f"  [WARN] No milestones for {exp.label}, skipping.")
                continue

            compute_cumulative_time(exp, milestones)
            if cap_hours is not None:
                milestones = filter_milestones_by_cap_hours(milestones, cap_hours)
            if not milestones:
                continue
            compute_cost_allocations(exp, milestones)

            color, linestyle, _ = exp_styles[i]

            if x_axis == "steps" and y_axis == "milestones":
                x_vals = [m["cumulative_steps"] for m in milestones]
                y_vals = [milestone_to_idx[m["label"]] for m in milestones]

                ax.plot(x_vals, y_vals, color=color, linestyle=linestyle, linewidth=2.0, alpha=0.9,
                    label=exp.label, marker="o", markersize=6, zorder=2)

            elif x_axis == "time" and y_axis == "milestones":
                x_vals = [m["cumulative_time_hours"] for m in milestones]
                y_vals = [milestone_to_idx[m["label"]] for m in milestones]

                ax.plot(x_vals, y_vals, color=color, linestyle=linestyle, linewidth=2.0, alpha=0.9,
                    label=exp.label, marker="o", markersize=6, zorder=2)

            elif x_axis == "cost" and y_axis == "milestones":
                x_vals = [m["cumulative_cost"] for m in milestones]
                y_vals = [milestone_to_idx[m["label"]] for m in milestones]

                ax.plot(x_vals, y_vals, color=color, linestyle=linestyle, linewidth=2.0, alpha=0.9,
                    label=exp.label, marker="o", markersize=6, zorder=2)

            elif x_axis == "steps" and y_axis == "tokens":
                x_vals = [m["cumulative_steps"] for m in milestones]
                y_vals = [m["cumulative_total_tokens"] for m in milestones]

                ax.plot(x_vals, y_vals, color=color, linestyle=linestyle, linewidth=2.0, alpha=0.9,
                    label=exp.label, marker="o", markersize=4, zorder=2)

            elif x_axis == "steps" and y_axis == "time":
                x_vals = [m["cumulative_steps"] for m in milestones]
                y_vals = [m["cumulative_time_hours"] for m in milestones]

                ax.plot(x_vals, y_vals, color=color, linestyle=linestyle, linewidth=2.0, alpha=0.9,
                    label=exp.label, marker="o", markersize=4, zorder=2)

            elif x_axis == "steps" and y_axis == "cost":
                x_vals = [m["cumulative_steps"] for m in milestones]
                y_vals = [m["cumulative_cost"] for m in milestones]

                ax.plot(x_vals, y_vals, color=color, linestyle=linestyle, linewidth=2.0, alpha=0.9,
                    label=exp.label, marker="o", markersize=4, zorder=2)

            elif x_axis == "time" and y_axis == "cost":
                x_vals = [m["cumulative_time_hours"] for m in milestones]
                y_vals = [m["cumulative_cost"] for m in milestones]

                ax.plot(x_vals, y_vals, color=color, linestyle=linestyle, linewidth=2.0, alpha=0.9,
                    label=exp.label, marker="o", markersize=4, zorder=2)

    # ── Axis labels & formatting ────────────────────────────────────────────
    x_labels = {
        "steps": "Cumulative Steps (tool invocations)",
        "time": "Cumulative Time (hours)",
        "cost": "Cumulative Cost ($)",
    }
    ax.set_xlabel(x_labels.get(x_axis, x_axis), fontweight="bold")

    if y_axis == "milestones":
        ax.set_yticks(range(len(MILESTONE_LABELS)))
        ax.set_yticklabels(MILESTONE_LABELS, fontsize=8)
        ax.set_ylim(-0.5, len(MILESTONE_LABELS) - 0.5)
        ax.set_ylabel("Milestone", fontweight="bold")
    elif y_axis == "tokens":
        ax.set_ylabel("Cumulative Tokens", fontweight="bold")
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x / 1e6:.1f}M" if x >= 1e6 else f"{x / 1e3:.0f}K")
        )
    elif y_axis == "time":
        ax.set_ylabel("Cumulative Time (hours)", fontweight="bold")
    elif y_axis == "cost":
        ax.set_ylabel("Cumulative Cost ($)", fontweight="bold")

    # Apply log scale only for continuous metrics (steps, time, cost, tokens)
    if log_x and x_axis in ("steps", "time", "cost"):
        ax.set_xscale("log")
    if log_y and y_axis in ("tokens", "time", "cost"):
        ax.set_yscale("log")

    if title:
        ax.set_title(title, fontweight="bold")
    else:
        ax.set_title(
            f"Milestone Progress vs. {x_axis.capitalize()}", fontweight="bold"
        )

    ax.grid(axis="x", alpha=0.3, linewidth=0.5)
    ax.grid(axis="y", alpha=0.15, linewidth=0.5)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9, ncol=1)
    _clean_axes(ax)
    plt.tight_layout()

    return fig


def generate_bar_figure(
    experiments: list[ExperimentData],
    x_axis: str,
    y_axis: str,
    title: str | None,
    cap_hours: float | None = None,
    aggregate: bool = False,
) -> plt.Figure:
    """
    Generate a grouped bar chart.

    x_axis = "milestones" (fixed)
    y_axis = tokens | steps | time | split_tokens | split_steps | split_time
    """
    _apply_style()

    n_milestones = len(MILESTONE_LABELS)
    if not experiments:
        print("ERROR: No experiments to plot.", file=sys.stderr)
        sys.exit(1)

    # Map y_axis arg to the data key in milestone dicts
    y_key_map = {
        "tokens":       "cumulative_total_tokens",
        "steps":        "cumulative_steps",
        "time":         "split_time_seconds",
        "split_cost":   "split_cost",
        "split_tokens": "split_total_tokens",
        "split_steps":  "split_steps",
        "split_time":   "split_time_seconds",
    }
    data_key = y_key_map[y_axis]

    fig, ax = plt.subplots(figsize=(14, 6))
    x_positions = np.arange(n_milestones)

    if aggregate:
        groups = _group_experiments_for_aggregation(experiments)
        n_groups = len(groups)
        bar_width = 0.8 / n_groups

        for gi, ((model, variant, prompt_opt_freq), group_exps) in enumerate(groups):
            # Build matrix: rows = runs, cols = milestones
            matrix = np.full((len(group_exps), n_milestones), np.nan)
            for ri, exp in enumerate(group_exps):
                milestones = extract_milestone_data(exp)
                if not milestones:
                    continue
                compute_cumulative_time(exp, milestones)
                if cap_hours is not None:
                    milestones = filter_milestones_by_cap_hours(milestones, cap_hours)
                if not milestones:
                    continue
                compute_cost_allocations(exp, milestones)
                for m in milestones:
                    try:
                        mi = MILESTONE_LABELS.index(m["label"])
                        matrix[ri, mi] = m[data_key]
                    except ValueError:
                        pass

            with np.errstate(all="ignore"):
                means = np.nanmean(matrix, axis=0)
                mins = np.nanmin(matrix, axis=0)
                maxs = np.nanmax(matrix, axis=0)
            means = np.nan_to_num(means, nan=0.0)
            err_lo = means - np.nan_to_num(mins, nan=means)
            err_hi = np.nan_to_num(maxs, nan=means) - means

            color = COLORS[gi % len(COLORS)]
            offset = (gi - n_groups / 2 + 0.5) * bar_width
            label = _aggregate_group_label(model, variant, prompt_opt_freq)

            ax.bar(
                x_positions + offset, means, bar_width,
                label=label, color=color, alpha=0.85, edgecolor="white", linewidth=0.5,
            )
            if len(group_exps) > 1:
                ax.errorbar(
                    x_positions + offset, means,
                    yerr=[err_lo, err_hi], fmt="none", color="black", capsize=2,
                )
    else:
        n_experiments = len(experiments)
        bar_width = 0.8 / n_experiments
        exp_styles = assign_colors_and_styles(experiments)

        for i, exp in enumerate(experiments):
            milestones = extract_milestone_data(exp)
            if not milestones:
                continue
            compute_cumulative_time(exp, milestones)
            if cap_hours is not None:
                milestones = filter_milestones_by_cap_hours(milestones, cap_hours)
            if not milestones:
                continue
            compute_cost_allocations(exp, milestones)

            milestone_lookup: dict[str, float] = {}
            for m in milestones:
                milestone_lookup[m["label"]] = m[data_key]

            values = [milestone_lookup.get(label, 0) for label in MILESTONE_LABELS]
            color, _, hatch = exp_styles[i]
            offset = (i - n_experiments / 2 + 0.5) * bar_width

            ax.bar(
                x_positions + offset, values, bar_width,
                label=exp.label, color=color, alpha=0.85, edgecolor="white", linewidth=0.5,
                hatch=hatch if hatch else None,
            )

    # ── Axis labels ─────────────────────────────────────────────────────────
    ax.set_xticks(x_positions)
    ax.set_xticklabels(MILESTONE_LABELS, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Milestone", fontweight="bold")

    y_labels = {
        "tokens":       "Cumulative Tokens",
        "steps":        "Cumulative Steps",
        "time":         "Time (seconds)",
        "split_cost":   "Cost ($, per milestone)",
        "split_tokens": "Split Tokens (per objective)",
        "split_steps":  "Split Steps (per objective)",
        "split_time":   "Split Time (seconds, per objective)",
    }
    ax.set_ylabel(y_labels.get(y_axis, y_axis), fontweight="bold")

    if y_axis in ("tokens", "split_tokens"):
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x / 1e6:.1f}M" if x >= 1e6 else f"{x / 1e3:.0f}K")
        )

    if title:
        ax.set_title(title, fontweight="bold")
    else:
        ax.set_title(
            f"{y_labels.get(y_axis, y_axis)} per Milestone", fontweight="bold",
        )

    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=8, ncol=2)
    _clean_axes(ax)
    plt.tight_layout()

    return fig


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Validate axis combination
    validate_axes(args)

    # Resolve data root
    data_root = Path(args.data_root) if args.data_root else EXPERIMENT_DATA_ROOT
    if not data_root.is_dir():
        print(f"ERROR: Data root not found: {data_root}", file=sys.stderr)
        sys.exit(1)

    if args.cap_hours is not None:
        print(f"  [cap] Truncating data to {args.cap_hours}h window")

    # ── Load data ───────────────────────────────────────────────────────────
    experiments = load_all_experiments(
        data_root=data_root,
        method=args.method,
        settings=args.experiment_settings,
        models=args.models,
        variants=args.variants,
        experiment_nums=args.experiment_nums,
    )

    # ── Generate figure ─────────────────────────────────────────────────────
    if args.figure_type == "line":
        fig = generate_line_figure(
            experiments, args.x_axis, args.y_axis, args.title,
            log_x=args.log_x,
            log_y=args.log_y,
            cap_hours=args.cap_hours,
            aggregate=args.aggregate,
        )
    elif args.figure_type == "bar":
        fig = generate_bar_figure(
            experiments, args.x_axis, args.y_axis, args.title,
            cap_hours=args.cap_hours,
            aggregate=args.aggregate,
        )
    else:
        print(f"ERROR: Unknown figure type: {args.figure_type}", file=sys.stderr)
        sys.exit(1)

    # ── Save output ─────────────────────────────────────────────────────────
    output_base = Path(args.output)
    output_base.parent.mkdir(parents=True, exist_ok=True)

    for fmt in args.format:
        out_path = output_base.with_suffix(f".{fmt}")
        out_path = out_path.with_stem(f"{args.figure_type}_".upper() + out_path.stem)
        dpi = 300 if fmt == "png" else None
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out_path}")

    plt.close(fig)
    print("\nDone!")


if __name__ == "__main__":
    main()
