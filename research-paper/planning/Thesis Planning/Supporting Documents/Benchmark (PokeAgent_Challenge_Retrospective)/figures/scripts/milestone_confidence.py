#!/usr/bin/env python3
"""
Per-milestone 2×2 confidence range figure for PokeAgent Track 2 baselines.

Generates a single figure with 4 subplots (milestones on x-axis):
  (a) Cumulative Wall-Clock Time   (b) Cumulative Actions (steps)
  (c) Cumulative Tokens            (d) Cumulative Cost (USD)

Each model shows mean line + shaded min–max band across runs.

Usage:
    python milestone_confidence.py --output figures/output/milestone_confidence
    python milestone_confidence.py --output figures/output/milestone_confidence --data-root /path/to/data
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Milestones ───────────────────────────────────────────────────────────────
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

MILESTONE_LABELS = [label for label, _ in MILESTONE_OBJECTIVES]
MILESTONE_IDS = [obj_id for _, obj_id in MILESTONE_OBJECTIVES]

# CLI runs use milestones[] with milestone_id; map to canonical objective_id for comparison
CLI_MILESTONE_TO_OBJECTIVE: dict[str, str] = {
    "LITTLEROOT_TOWN":       "tutorial_000",
    "ROUTE_101":             "tutorial_006",
    "STARTER_CHOSEN":        "tutorial_007",
    "OLDALE_TOWN":           "early_011",
    "RIVAL_BATTLE_WON":      "early_014",
    "ROUTE_103":             "early_014",
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

# ── Cost correction (experimental) ──────────────────────────────────────────
# Claude Code CLI reports usage differently than our metric tracking, when using 
# Claude via the Claude Code Subscription; empirical
# comparison of reported vs. expected cost yields ~1.9× underreporting. 
# Multiply CLI Claude Code costs by this factor for cross-model comparison
# when data was generated using the Claude Code subscription.
# Derived from experimental analysis of usage reports vs. PokeAgent metrics.
CLI_CLAUDE_CODE_COST_CORRECTION_FACTOR = 1

EXPERIMENT_DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "experiment_data"

# ── Model display config ─────────────────────────────────────────────────────
# Marker convention: GPT = triangle (^), Gemini = circle (o), Claude = square (s)
MODEL_CONFIG: dict[str, dict[str, Any]] = {
    "gemini/3-flash":       {"label": "Gemini 3 Flash",      "color": "#0072B2", "marker": "o"},
    "gemini/3-pro":         {"label": "Gemini 3 Pro",        "color": "#009E73", "marker": "o"},
    "gemini/3.1-pro":       {"label": "Gemini 3.1 Pro",      "color": "#56B4E9", "marker": "o"},
    "claude/sonnet-4-5":    {"label": "Claude Sonnet 4.5",   "color": "#CC79A7", "marker": "s"},
    "cli/codex-cli":        {"label": "GPT 5.3 Codex",       "color": "#D55E00", "marker": "^"},
    "gpt/5.2":              {"label": "GPT 5.2",             "color": "#E69F00", "marker": "^"},
    "gpt/o3-mini":          {"label": "GPT o3-mini",         "color": "#D55E00", "marker": "^"},
    "gpt/4o":               {"label": "GPT 4o",              "color": "#F0E442", "marker": "^"},
    "cli/claude-code":      {"label": "Claude Sonnet 4.6",  "color": "#882255", "marker": "s"},
    "cli/gemini-cli":       {"label": "Gemini 3.1 Pro",      "color": "#56B4E9", "marker": "o"},
}

# ── Pricing (USD per 1M tokens, as of Feb 2026) ─────────────────────────────
# Format: {model_key: (input_per_1M, cached_input_per_1M, output_per_1M)}
MODEL_PRICING: dict[str, tuple[float, float, float]] = {
    # OpenAI GPT-5
    "gpt/5":                (1.25,   0.125,   10.0),
    "gpt/5-mini":           (0.25,   0.025,   2.0),
    "gpt/5-nano":           (0.05,   0.005,   0.4),
    "gpt/5.1":              (1.25,   0.125,   10.0),
    "gpt/5.2":              (1.75,   0.175,   14.0),
    "gpt/5.2-pro":          (21.0,   10.5,    168.0),   # no official cached, using prompt/2
    "gpt/5-pro":            (15.0,   7.5,     120.0),   # no official cached, using prompt/2
    "gpt/5.2-chat-latest":  (1.75,   0.175,   14.0),
    "gpt/5.1-chat-latest":  (1.25,   0.125,   10.0),
    "gpt/5-chat-latest":    (1.25,   0.125,   10.0),
    "gpt/5.2-codex":        (1.75,   0.175,   14.0),
    "gpt/5.1-codex-max":    (1.25,   0.125,   10.0),
    "gpt/5.1-codex":        (1.25,   0.125,   10.0),
    "gpt/5-codex":          (1.25,   0.125,   10.0),
    
    # OpenAI GPT-4
    "gpt/4o":               (2.5,    1.25,    10.0),
    "gpt/4o-mini":          (0.15,   0.075,   0.6),
    "gpt/4.1":              (2.0,    0.5,     8.0),
    "gpt/4.1-mini":         (0.4,    0.1,     1.6),
    "gpt/4.1-nano":         (0.1,    0.025,   0.4),
    
    # OpenAI o-series
    "gpt/o4-mini":          (1.1,    0.275,   4.4),
    "gpt/o3-mini":          (1.1,    0.55,    4.4),
    "gpt/o3":               (2.0,    0.5,     8.0),
    "gpt/o3-pro":           (20.0,   10.0,    80.0),    # no official cached, using prompt/2
    "gpt/o1":               (15.0,   7.5,     60.0),
    "gpt/o1-pro":           (150.0,  75.0,    600.0),   # no official cached, using prompt/2
    
    # Anthropic Claude (Base input / Cache write / Cache hits / Output per 1M)
    # Note: cache_write_prompt is for writing to cache, not reading cached
    "claude/sonnet-4.5":    (3.0,    0.3,     15.0),
    "claude/sonnet-4":      (3.0,    0.3,     15.0),
    "claude/sonnet-3.7":    (3.0,    0.3,     15.0),
    "claude/sonnet-3.5":    (3.0,    0.3,     15.0),
    "claude/opus-4.1":      (15.0,   1.5,     75.0),
    "claude/opus-4":        (15.0,   1.5,     75.0),
    "claude/opus-3":        (15.0,   1.5,     75.0),
    "claude/haiku-4.5":     (1.0,    0.1,     5.0),
    "claude/haiku-3.5":     (0.8,    0.08,    4.0),
    "claude/haiku-3":       (0.25,   0.03,    1.25),
    
    # Gemini 2.x
    "gemini/2.5-flash":     (0.3,    0.03,    2.5),
    "gemini/2.5-flash-preview-09-2025": (0.3, 0.03, 2.5),
    "gemini/2.5-pro":       (1.25,   0.125,   10.0),
    "gemini/2.5-flash-lite": (0.1,   0.01,    0.4),
    "gemini/2.5-flash-lite-preview-09-2025": (0.1, 0.01, 0.4),
    "gemini/2.0-flash":     (0.15,   0.075,   0.6),    # no cached in original, using prompt/2
    
    # Gemini 3.x
    "gemini/3-pro-preview": (2.0,    0.2,     12.0),
    "gemini/3-pro":         (2.0,    0.2,     12.0),
    "gemini/3.1-pro":       (2.0,    0.2,     12.0),  # Same as 3-pro
    "gemini/3-flash":       (0.5,    0.05,    3.0),
    "gemini/3-flash-preview": (0.5,  0.05,    3.0),
    
    # CLI agents (Claude Code uses Sonnet 4.5/4.6; Gemini CLI uses 3.1 Pro; Codex CLI uses GPT 5.3)
    "cli/claude-code":      (3.0,    0.3,     15.0),
    "cli/gemini-cli":       (2.0,    0.2,     12.0),  # Gemini 3.1 Pro pricing
    "cli/codex-cli":        (1.75,   0.175,   14.0),  # GPT 5.3 Codex (same ballpark as 5.2)
}

FALLBACK_COLORS = ["#56B4E9", "#332288", "#882255", "#44AA99", "#AA4499"]

# Legend order for model section: GPT-5.2 then GPT 5.3 Codex; Gemini 3 Flash, 3 Pro, 3.1 Pro; then Claude
MODEL_LEGEND_ORDER = [
    "gpt/5.2", "cli/codex-cli",
    "gemini/3-flash", "gemini/3-pro", "gemini/3.1-pro",
    "claude/sonnet-4-5", "cli/claude-code", "cli/gemini-cli",
]

# ── NeurIPS style ────────────────────────────────────────────────────────────
NEURIPS_RCPARAMS = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "text.usetex": False,
    "axes.linewidth": 1.0,
    "axes.grid": False,
    "lines.linewidth": 2.2,
    "lines.markersize": 6,
}


# =============================================================================
# DATA LOADING
# =============================================================================

def compress_gaps(data: dict[str, Any], gap_threshold: float = 300.0) -> None:
    """
    In-place compress timestamps by removing idle gaps > gap_threshold.
    Mirrors merge_cumulative_metrics.py logic to reconcile pauses.
    """
    steps = data.get("steps", [])
    if not steps or len(steps) < 2:
        return

    deductions: list[tuple[float, float]] = []
    cum_idle = 0.0

    # Sort steps just in case, though usually sorted
    # We use a simple list scan assuming sorted order for O(N)
    for i in range(1, len(steps)):
        gap = steps[i]["timestamp"] - steps[i-1]["timestamp"]
        if gap > gap_threshold:
            excess = gap - gap_threshold
            cum_idle += excess
            deductions.append((steps[i]["timestamp"], cum_idle))

    if cum_idle == 0:
        return

    def get_adjusted_ts(ts: float) -> float:
        # Find the latest deduction that applies to this timestamp
        for start_ts, total_deduct in reversed(deductions):
            if ts >= start_ts:
                return ts - total_deduct
        return ts

    for key in ["milestones", "objectives"]:
        if key in data and data[key]:
            for item in data[key]:
                if "timestamp" in item:
                    item["timestamp"] = get_adjusted_ts(item["timestamp"])


def _load_run_into_results(
    data: dict[str, Any],
    results: dict[str, list[dict[str, dict[str, float]]]],
    model_key: str,
    exp_dir_name: str,
) -> None:
    """Parse objectives or CLI milestones from cumulative_metrics and append to results[model_key]."""
    compress_gaps(data)
    start_time = data.get("start_time", 0.0)
    obj_lookup: dict[str, dict[str, float]] = {}

    # All-tools runs use objectives[] with objective_id; CLI runs use milestones[] with milestone_id
    raw_items = data.get("objectives", [])
    if raw_items:
        for obj in raw_items:
            oid = obj["objective_id"]
            obj_lookup[oid] = {
                "steps": obj["cumulative_steps"],
                "tokens": obj["cumulative_total_tokens"],
                "prompt_tokens": obj["cumulative_prompt_tokens"],
                "completion_tokens": obj["cumulative_completion_tokens"],
                "cached_tokens": obj.get("cumulative_cached_tokens", 0),
                "time_hours": (obj["timestamp"] - start_time) / 3600.0,
            }
    else:
        # CLI: milestones[] with milestone_id; map to canonical objective_id
        for m in data.get("milestones", []):
            mid = m.get("milestone_id")
            if mid not in CLI_MILESTONE_TO_OBJECTIVE:
                continue
            oid = CLI_MILESTONE_TO_OBJECTIVE[mid]
            obj_lookup[oid] = {
                "steps": m["cumulative_steps"],
                "tokens": m["cumulative_total_tokens"],
                "prompt_tokens": m["cumulative_prompt_tokens"],
                "completion_tokens": m["cumulative_completion_tokens"],
                "cached_tokens": m.get("cumulative_cached_tokens", 0),
                "time_hours": (m["timestamp"] - start_time) / 3600.0,
            }

    if not obj_lookup:
        print(f"  [SKIP] No objectives/milestones in {exp_dir_name}")
        return

    # Cost via proportional allocation (avoids CLI token-accounting mismatch)
    total_cost = data.get("total_cost", 0.0)
    total_tokens = data.get("total_tokens", 1)
    if model_key == "cli/claude-code":
        total_cost *= CLI_CLAUDE_CODE_COST_CORRECTION_FACTOR
    for oid in obj_lookup:
        obj_lookup[oid]["cost"] = total_cost * (obj_lookup[oid]["tokens"] / total_tokens)

    # Enforce monotonicity on cumulative fields
    prev = {"steps": 0, "tokens": 0, "prompt_tokens": 0,
            "completion_tokens": 0, "cached_tokens": 0, "time_hours": 0, "cost": 0}
    for _, oid in MILESTONE_OBJECTIVES:
        if oid in obj_lookup:
            for k in prev:
                obj_lookup[oid][k] = max(obj_lookup[oid][k], prev[k])
                prev[k] = obj_lookup[oid][k]

    results[model_key].append(obj_lookup)
    print(f"  Loaded: {model_key} / {exp_dir_name}")


def discover_and_load(data_root: Path) -> dict[str, list[dict[str, dict[str, float]]]]:
    """
    Load all experiments grouped by model key.

    Scans baseline/all_tools (model/variant hierarchy) and baseline/cli (agent_type hierarchy).
    Each run stores per-milestone metrics:
        {obj_id: {"steps": ..., "tokens": ..., "prompt_tokens": ...,
                  "completion_tokens": ..., "cached_tokens": ...,
                  "time_hours": ...}}
    """
    results: dict[str, list[dict[str, dict[str, float]]]] = defaultdict(list)

    # 1. Scan baseline/all_tools: model/variant/experiment_N
    baseline_all_tools = data_root / "baseline" / "all_tools"
    if baseline_all_tools.is_dir():
        for model_dir in sorted(baseline_all_tools.iterdir()):
            if not model_dir.is_dir():
                continue
            for variant_dir in sorted(model_dir.iterdir()):
                if not variant_dir.is_dir():
                    continue
                model_key = f"{model_dir.name}/{variant_dir.name}"
                for exp_dir in sorted(variant_dir.iterdir()):
                    if not exp_dir.is_dir() or not exp_dir.name.startswith("experiment_"):
                        continue
                    for cache_dir in sorted(exp_dir.glob("pokeagent_cache_*")):
                        metrics_path = cache_dir / "cumulative_metrics.json"
                        if not metrics_path.exists():
                            continue
                        with open(metrics_path) as f:
                            data = json.load(f)
                        _load_run_into_results(data, results, model_key, exp_dir.name)

    # 2. Scan baseline/cli: agent_type/experiment_N
    baseline_cli = data_root / "baseline" / "cli"
    if baseline_cli.is_dir():
        for agent_dir in sorted(baseline_cli.iterdir()):
            if not agent_dir.is_dir():
                continue
            model_key = f"cli/{agent_dir.name}"
            for exp_dir in sorted(agent_dir.iterdir()):
                if not exp_dir.is_dir() or not exp_dir.name.startswith("experiment_"):
                    continue
                for cache_dir in sorted(exp_dir.glob("pokeagent_cache_*")):
                    metrics_path = cache_dir / "cumulative_metrics.json"
                    if not metrics_path.exists():
                        continue
                    with open(metrics_path) as f:
                        data = json.load(f)
                    _load_run_into_results(data, results, model_key, exp_dir.name)

    if not results:
        print("ERROR: No experiments found.", file=sys.stderr)
        sys.exit(1)

    return results


def compute_cost(prompt_tokens: float, completion_tokens: float,
                 cached_tokens: float, pricing: tuple[float, float, float]) -> float:
    """Compute USD cost from token counts and per-1M pricing."""
    input_price, cached_price, output_price = pricing
    # cached_tokens are a subset of prompt_tokens in some APIs;
    # treat them as replacing that portion at the cached rate
    uncached_prompt = prompt_tokens - cached_tokens
    cost = (uncached_prompt * input_price / 1e6
            + cached_tokens * cached_price / 1e6
            + completion_tokens * output_price / 1e6)
    return cost


# =============================================================================
# FIGURE GENERATION
# =============================================================================

def generate_figure(all_data: dict[str, list[dict]], title: str | None,
                    cap_hours: float | None = None, tokens_log: bool = False,
                    exclude_wall_clock: bool = False) -> plt.Figure:
    plt.rcParams.update(NEURIPS_RCPARAMS)

    if exclude_wall_clock:
        fig = plt.figure(figsize=(16, 12))
        grid = fig.add_gridspec(2, 2)
        ax_tokens = fig.add_subplot(grid[0, :])
        ax_actions = fig.add_subplot(grid[1, 0], sharey=ax_tokens)
        ax_cost = fig.add_subplot(grid[1, 1], sharey=ax_tokens)
        axes_list = [ax_tokens, ax_actions, ax_cost]
        subplot_config = [
            (0, "tokens", "Tokens", "(a) Cumulative Tokens"),
            (1, "steps", "Actions", "(b) Cumulative Actions"),
            (2, "cost", "Cost (USD)", "(c) Cumulative Cost"),
        ]
        y_label_axes = [axes_list[0], axes_list[1]]
    else:
        fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharey=True)
        axes_list = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]
        subplot_config = [
            (0, "time_hours", "Hours", "(a) Wall-Clock Time"),
            (1, "steps", "Actions", "(b) Cumulative Actions"),
            (2, "tokens", "Tokens", "(c) Cumulative Tokens"),
            (3, "cost", "Cost (USD)", "(d) Cumulative Cost"),
        ]
        y_label_axes = [axes[0, 0], axes[1, 0]]

    y = np.arange(len(MILESTONE_LABELS))
    fallback_idx = 0
    legend_handles = []

    for model_key, runs in sorted(all_data.items()):
        cfg = MODEL_CONFIG.get(model_key)
        if cfg is None:
            color = FALLBACK_COLORS[fallback_idx % len(FALLBACK_COLORS)]
            fallback_idx += 1
            label = model_key.replace("/", " ").title()
            marker = "o"
        else:
            color = cfg["color"]
            label = cfg["label"]
            marker = cfg["marker"]

        pricing = MODEL_PRICING.get(model_key, (1.0, 0.5, 5.0))

        for (ax_idx, metric_key, _xlabel, _subplot_title) in subplot_config:
            # Wall-clock time panel: exclude CLI agents (timestamp semantics differ)
            if metric_key == "time_hours" and model_key.startswith("cli/"):
                continue
            ax = axes_list[ax_idx]

            # Build matrix: rows = runs, cols = milestones
            # cap_hours: only include data for milestones reached within the time window
            matrix = np.full((len(runs), len(MILESTONE_IDS)), np.nan)
            for ri, run in enumerate(runs):
                for mi, oid in enumerate(MILESTONE_IDS):
                    if oid not in run:
                        continue
                    if cap_hours is not None and run[oid]["time_hours"] > cap_hours:
                        continue  # beyond cap window
                    if metric_key == "cost":
                        matrix[ri, mi] = run[oid]["cost"]  # proportional from total_cost
                    else:
                        matrix[ri, mi] = run[oid][metric_key]

            with np.errstate(all="ignore"):
                means = np.nanmean(matrix, axis=0)
                mins = np.nanmin(matrix, axis=0)
                maxs = np.nanmax(matrix, axis=0)

            valid_mask = ~np.isnan(means)
            if not valid_mask.any():
                continue

            last_valid = np.max(np.where(valid_mask))
            valid_idx = np.where(valid_mask)[0]

            # Fill missing milestones so the line connects (e.g. CLI agents with undetected milestones)
            if len(valid_idx) >= 2:
                fill_idx = np.arange(last_valid + 1)
                means[fill_idx] = np.interp(fill_idx, valid_idx, means[valid_idx])
                mins[fill_idx] = np.interp(fill_idx, valid_idx, mins[valid_idx])
                maxs[fill_idx] = np.interp(fill_idx, valid_idx, maxs[valid_idx])

            py = y[:last_valid + 1]
            pm = means[:last_valid + 1]
            pmin = mins[:last_valid + 1]
            pmax = maxs[:last_valid + 1]

            # Linestyle: solid = PokéAgent, dashed = Claude Code, dotted = Gemini CLI, dash-dot = Codex CLI
            if model_key == "cli/claude-code":
                linestyle = "--"
            elif model_key == "cli/gemini-cli":
                linestyle = ":"
            elif model_key == "cli/codex-cli":
                linestyle = "-."
            else:
                linestyle = "-"

            # Milestones on y-axis, metric on x-axis
            line, = ax.plot(pm, py, color=color, linestyle=linestyle, linewidth=2.2,
                            marker=marker, markersize=6, alpha=0.9, zorder=3)

            if len(runs) > 1:
                ax.fill_betweenx(py, pmin, pmax, color=color, alpha=0.15, zorder=1)

            # Legend is built separately below (2-line: Scaffold + Model)

    # ── Build 2-line legend: Scaffold (line type only) + Model (shape + colour only) ──
    scaffold_handles = [
        mlines.Line2D([], [], color="gray", linestyle="-", linewidth=2.5, label="PokéAgent"),
        mlines.Line2D([], [], color="gray", linestyle="--", linewidth=2.5, label="Claude Code"),
        mlines.Line2D([], [], color="gray", linestyle=":", linewidth=2.5, label="Gemini CLI"),
        mlines.Line2D([], [], color="gray", linestyle="-.", linewidth=2.5, label="Codex CLI"),
    ]
    # Model section: coloured marker only (no line), in MODEL_LEGEND_ORDER then any remainder
    model_handles = []
    seen_labels = set()
    keys_done = set()
    for model_key in MODEL_LEGEND_ORDER + sorted(all_data.keys()):
        if model_key not in all_data or model_key in keys_done:
            continue
        keys_done.add(model_key)
        cfg = MODEL_CONFIG.get(model_key)
        if cfg and cfg["label"] not in seen_labels:
            seen_labels.add(cfg["label"])
            model_handles.append(
                mlines.Line2D([], [], color=cfg["color"], linestyle="None", linewidth=0,
                              marker=cfg["marker"], markersize=10, markerfacecolor=cfg["color"],
                              markeredgecolor=cfg["color"], label=cfg["label"])
            )

    legend_handles = scaffold_handles + model_handles

    # ── Format each subplot ──────────────────────────────────────────────────
    for (ax_idx, metric_key, xlabel, subplot_title) in subplot_config:
        ax = axes_list[ax_idx]
        ax.set_title(subplot_title, fontweight="bold", fontsize=18, pad=10)
        ax.set_xlabel(xlabel, fontweight="bold", fontsize=18)
        ax.grid(axis="x", alpha=0.3, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=14)

        if metric_key == "tokens":
            if tokens_log:
                ax.set_xscale("log")
                ax.set_xlim(left=1e6)
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, _: f"{v / 1e6:.1f}M" if v >= 1e6 else f"{v / 1e3:.0f}K")
            )
        elif metric_key == "cost":
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, _: f"${v:.0f}" if v >= 1 else f"${v:.2f}")
            )
        elif metric_key == "time_hours":
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, _: f"{v:.1f}h")
            )

    # Y-axis milestone labels on left column only (shared y)
    for ax in y_label_axes:
        ax.set_yticks(y)
        ax.set_yticklabels(MILESTONE_LABELS, fontsize=14)

    # Layout: title, legend, subplots (plots up, title down, less gap between legend and plots)
    plt.tight_layout(rect=[0, 0, 1, 0.82])
    if title:
        fig.suptitle(title, fontweight="normal", fontsize=30, y=0.92)
    fig.legend(handles=legend_handles, loc="upper center", ncol=6,
               framealpha=0.9, fontsize=12, bbox_to_anchor=(0.5, 0.88))
    return fig


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Per-milestone 2x2 confidence range figure.")
    parser.add_argument("--output", default="figures/output/milestone_confidence",
                        help="Output path (without extension).")
    parser.add_argument("--format", nargs="+", default=["png", "pdf"], choices=["png", "pdf"])
    parser.add_argument("--title", default="Speedrunning: Harness vs. Model",
                        help="Figure suptitle (default: Speedrunning: Harness vs. Model).")
    parser.add_argument("--data-root", default=None, help="Override data root.")
    parser.add_argument("--cap-hours", type=float, default=None,
                        help="Only include milestone data within this many hours (truncates, does not drop runs).")
    parser.add_argument("--tokens-log", action="store_true",
                        help="Use log scale for the cumulative tokens subplot (c).")
    parser.add_argument("--exclude-wall-clock", action="store_true",
                        help="Drop Wall-Clock Time panel; show Cumulative Tokens full-width on top row.")
    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else EXPERIMENT_DATA_ROOT
    print(f"Loading from: {data_root}")

    all_data = discover_and_load(data_root)
    if not all_data:
        print("ERROR: No data found.", file=sys.stderr)
        sys.exit(1)

    if args.cap_hours is not None:
        print(f"  [cap] Truncating data to {args.cap_hours}h window")

    fig = generate_figure(all_data, args.title, cap_hours=args.cap_hours, tokens_log=args.tokens_log,
                         exclude_wall_clock=args.exclude_wall_clock)

    out_base = Path(args.output)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    extra = [fig._suptitle] if fig._suptitle else []
    if fig.legends:
        extra.append(fig.legends[0])
    for fmt in args.format:
        out_path = out_base.with_suffix(f".{fmt}")
        fig.savefig(out_path, dpi=300 if fmt == "png" else None, bbox_inches="tight",
                    bbox_extra_artists=extra, facecolor="white")
        print(f"Saved: {out_path}")

    plt.close(fig)
    print("Done!")


if __name__ == "__main__":
    main()
