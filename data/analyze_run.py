"""Tool-call + milestone analysis plots for a single PokeAgent run.

Reads ``trajectory_history.jsonl`` and ``milestones_progress.json``
from a `.pokeagent_cache/<run_id>/` directory and writes line plots
of:

- Cumulative tool-call counts vs step (one line per tool name)
- Cumulative button-press count vs step (each individual button
  pressed inside a press_buttons call counts separately)
- Specific skill usage vs step (one line per distinct skill name
  pulled from run_skill / process_skill / skill creation events)
- Cumulative button presses to reach each milestone

Usage:
    .venv/bin/python -m data.analyze_run \\
        --run .pokeagent_cache/20260408_195739_autonomous_autonomous_objective_creation_ae_autoevolve \\
        --output train_runs/analysis_20260408_195739
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger("analyze_run")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _parse_iso_to_epoch(s: str | None) -> float | None:
    if not s:
        return None
    try:
        # ISO format with optional fractional seconds
        return datetime.fromisoformat(s).timestamp()
    except Exception:
        return None


def load_trajectory(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_milestones(path: Path) -> List[Tuple[str, float]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    out: List[Tuple[str, float]] = []
    for name, info in data.get("milestones", {}).items():
        if info.get("completed") and info.get("timestamp"):
            out.append((name, float(info["timestamp"])))
    out.sort(key=lambda x: x[1])
    return out


def extract_tool_events(entries: List[dict]):
    """Walk trajectory entries and yield per-step tool events.

    Yields one dict per trajectory entry with:
        step, timestamp_epoch, tool_calls (list of {name, args}),
        button_presses_this_step (int), skills_run_this_step (set),
        skills_created_this_step (set)
    """
    for e in entries:
        step = e.get("step")
        ts = _parse_iso_to_epoch(e.get("timestamp"))
        tc_list = (e.get("action") or {}).get("tool_calls") or []
        button_presses = 0
        skills_run = set()
        skills_created = set()
        tool_calls_named = []
        for tc in tc_list:
            name = tc.get("name", "?")
            args = tc.get("args") or {}
            tool_calls_named.append(name)
            if name == "press_buttons":
                btns = args.get("buttons") or []
                if isinstance(btns, list):
                    button_presses += len(btns)
            elif name == "run_skill":
                # run_skill identifies the skill via skill_id
                sn = args.get("skill_id") or args.get("skill_name") \
                     or args.get("skill") or args.get("name")
                if sn:
                    skills_run.add(str(sn))
            elif name == "process_skill":
                # process_skill takes a list of `entries`, each entry
                # creates/updates a skill identified by `id` (or `name`).
                entries = args.get("entries") or []
                if isinstance(entries, list):
                    for ent in entries:
                        sn = (ent.get("id") or ent.get("name")
                              if isinstance(ent, dict) else None)
                        if sn:
                            skills_created.add(str(sn))
                # Fallback for legacy single-entry callsigns
                sn = args.get("skill_id") or args.get("skill_name") or args.get("name")
                if sn:
                    skills_created.add(str(sn))
        yield {
            "step": step,
            "timestamp": ts,
            "tool_calls": tool_calls_named,
            "button_presses_this_step": button_presses,
            "skills_run_this_step": skills_run,
            "skills_created_this_step": skills_created,
        }


def cumulative(values: List[int]) -> List[int]:
    out, acc = [], 0
    for v in values:
        acc += v
        out.append(acc)
    return out


def map_milestones_to_steps(
    milestones: List[Tuple[str, float]],
    events: List[dict],
) -> List[Tuple[str, int, int]]:
    """Map each completed milestone to the step it was reached at.

    Returns a list of (milestone_name, step, cumulative_button_presses_at_that_step).
    Skips milestones whose timestamp falls outside the trajectory's
    timestamp range (these are usually stale-cached pre-run milestones).
    """
    if not events:
        return []
    ts_min = min((e["timestamp"] for e in events if e["timestamp"]), default=0)
    ts_max = max((e["timestamp"] for e in events if e["timestamp"]), default=0)
    cum_presses_by_step: Dict[int, int] = {}
    running = 0
    for e in events:
        running += e["button_presses_this_step"]
        cum_presses_by_step[e["step"]] = running

    sorted_events_with_ts = [(e["step"], e["timestamp"]) for e in events if e["timestamp"]]
    sorted_events_with_ts.sort(key=lambda x: x[1])

    out: List[Tuple[str, int, int]] = []
    for name, ts in milestones:
        if ts < ts_min or ts > ts_max:
            continue
        # find the trajectory event closest in timestamp
        best_step = None
        best_dt = float("inf")
        for step, ets in sorted_events_with_ts:
            dt = abs(ets - ts)
            if dt < best_dt:
                best_dt = dt
                best_step = step
            if ets > ts and dt > best_dt:
                break
        if best_step is None:
            continue
        out.append((name, best_step, cum_presses_by_step.get(best_step, 0)))
    return out


def write_plots(output: Path, events: List[dict],
                milestones_at_steps: List[Tuple[str, int, int]],
                run_id: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    steps = [e["step"] for e in events]

    # Per-tool cumulative counts over time
    all_tools = set()
    for e in events:
        for n in e["tool_calls"]:
            all_tools.add(n)
    per_tool_count_at_step: Dict[str, List[int]] = {t: [0] * len(events) for t in all_tools}
    running = {t: 0 for t in all_tools}
    for i, e in enumerate(events):
        for n in e["tool_calls"]:
            running[n] += 1
        for t in all_tools:
            per_tool_count_at_step[t][i] = running[t]

    # ---------------- Plot 1: cumulative tool-call counts ----------------
    fig, ax = plt.subplots(figsize=(12, 6))
    # rank by final count desc
    ranked_tools = sorted(all_tools, key=lambda t: -per_tool_count_at_step[t][-1])
    for t in ranked_tools:
        ax.plot(steps, per_tool_count_at_step[t], label=t, linewidth=1.5)
    ax.set_xlabel("trajectory step")
    ax.set_ylabel("cumulative tool calls")
    ax.set_title(f"Cumulative tool-call counts vs step\n{run_id}")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = output / "01_tool_calls_vs_step.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    logger.info("wrote %s", p)

    # ---------------- Plot 2: cumulative button presses ----------------
    button_press_cum = cumulative([e["button_presses_this_step"] for e in events])
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(steps, button_press_cum, color="tab:blue", label="cumulative button presses")
    # Overlay milestones
    for name, ms_step, ms_presses in milestones_at_steps:
        ax.axvline(ms_step, color="tab:red", alpha=0.25, linewidth=0.8)
        ax.annotate(
            name,
            xy=(ms_step, ms_presses),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
            rotation=45,
            ha="left",
            va="bottom",
            color="tab:red",
        )
    ax.set_xlabel("trajectory step")
    ax.set_ylabel("cumulative button presses")
    ax.set_title(
        f"Cumulative button presses vs step + milestones reached\n{run_id}"
    )
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = output / "02_button_presses_vs_step.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    logger.info("wrote %s", p)

    # ---------------- Plot 3: skills run vs step ----------------
    all_skills_run: set = set()
    for e in events:
        all_skills_run.update(e["skills_run_this_step"])
    if all_skills_run:
        per_skill_run: Dict[str, List[int]] = {
            s: [0] * len(events) for s in all_skills_run
        }
        running = {s: 0 for s in all_skills_run}
        for i, e in enumerate(events):
            for s in e["skills_run_this_step"]:
                running[s] += 1
            for s in all_skills_run:
                per_skill_run[s][i] = running[s]

        fig, ax = plt.subplots(figsize=(12, 6))
        ranked_skills_run = sorted(all_skills_run, key=lambda s: -per_skill_run[s][-1])
        for s in ranked_skills_run[:20]:  # cap to top 20 skills
            ax.plot(steps, per_skill_run[s], label=s, linewidth=1.2)
        ax.set_xlabel("trajectory step")
        ax.set_ylabel("cumulative run_skill invocations")
        ax.set_title(f"Specific skill usage vs step (top 20)\n{run_id}")
        ax.legend(loc="upper left", fontsize=7, ncol=2)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = output / "03_skill_usage_vs_step.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        logger.info("wrote %s", p)
    else:
        logger.info("no run_skill calls — skipping plot 3")

    # ---------------- Plot 4: skills created vs step ----------------
    all_skills_created: set = set()
    for e in events:
        all_skills_created.update(e["skills_created_this_step"])
    if all_skills_created:
        per_skill_created: Dict[str, List[int]] = {
            s: [0] * len(events) for s in all_skills_created
        }
        running = {s: 0 for s in all_skills_created}
        for i, e in enumerate(events):
            for s in e["skills_created_this_step"]:
                running[s] += 1
            for s in all_skills_created:
                per_skill_created[s][i] = running[s]

        fig, ax = plt.subplots(figsize=(12, 6))
        # cumulative count of UNIQUE skills created
        unique_count = []
        seen = set()
        for e in events:
            for s in e["skills_created_this_step"]:
                seen.add(s)
            unique_count.append(len(seen))
        ax.plot(steps, unique_count, color="tab:green", linewidth=2,
                label="unique skills created")
        ax.set_xlabel("trajectory step")
        ax.set_ylabel("unique skills created")
        ax.set_title(f"Skill creation vs step\n{run_id}")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = output / "04_skills_created_vs_step.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        logger.info("wrote %s", p)
    else:
        logger.info("no process_skill calls — skipping plot 4")

    # ---------------- Plot 5: milestones reached vs cumulative button presses ----------------
    if milestones_at_steps:
        fig, ax = plt.subplots(figsize=(12, 6))
        ms_x = [m[2] for m in milestones_at_steps]
        ms_y = list(range(1, len(milestones_at_steps) + 1))
        ax.step(ms_x, ms_y, where="post", color="tab:red", linewidth=2,
                label="milestones reached")
        for name, ms_step, ms_presses in milestones_at_steps:
            ax.annotate(
                name, xy=(ms_presses, ms_y[milestones_at_steps.index((name, ms_step, ms_presses))]),
                xytext=(5, 0), textcoords="offset points", fontsize=7,
                ha="left", va="center"
            )
        ax.set_xlabel("cumulative button presses")
        ax.set_ylabel("milestones reached")
        ax.set_title(
            f"Milestones reached vs cumulative button presses\n{run_id}"
        )
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = output / "05_milestones_vs_button_presses.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        logger.info("wrote %s", p)

    # ---------------- Summary table ----------------
    summary = {
        "run_id": run_id,
        "total_steps": int(steps[-1]) if steps else 0,
        "total_trajectory_entries": len(events),
        "total_button_presses": button_press_cum[-1] if button_press_cum else 0,
        "tool_call_totals": {t: per_tool_count_at_step[t][-1] for t in ranked_tools},
        "milestones_reached_in_run": len(milestones_at_steps),
        "milestones": [
            {"name": n, "step": s, "button_presses_at_milestone": p}
            for n, s, p in milestones_at_steps
        ],
    }
    sp = output / "summary.json"
    sp.write_text(json.dumps(summary, indent=2))
    logger.info("wrote %s", sp)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True,
                    help="Path to .pokeagent_cache/<run_id>/ directory.")
    ap.add_argument("--output", type=Path, required=True,
                    help="Output dir for PNG plots + summary.json.")
    args = ap.parse_args()

    traj_path = args.run / "trajectory_history.jsonl"
    if not traj_path.exists():
        logger.error("missing %s", traj_path)
        return 1

    entries = load_trajectory(traj_path)
    logger.info("loaded %d trajectory entries from %s", len(entries), traj_path)

    milestones = load_milestones(args.run / "milestones_progress.json")
    logger.info("loaded %d completed milestones", len(milestones))

    events = list(extract_tool_events(entries))
    milestones_at_steps = map_milestones_to_steps(milestones, events)
    logger.info("matched %d milestones to trajectory steps", len(milestones_at_steps))

    write_plots(args.output, events, milestones_at_steps, args.run.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
