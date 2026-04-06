#!/usr/bin/env python3
"""
Merge N-part experiment cumulative_metrics.json files into one.

When an experiment is stopped and resumed, each segment produces its own
cumulative_metrics.json. This script merges them into a single consistent file:

  1. Concatenate steps from all parts with global step indices.
  2. Remove idle gaps between parts by compressing timestamps — any gap between
     consecutive steps larger than --gap-threshold seconds is collapsed, so that
     timestamps represent continuous *active* run time.
  3. Deduplicate objectives (one entry per objective_id, kept from the part
     whose time window actually completed it).
  4. Recompute cumulative_* and split_* on the merged, gap-compressed data.

Optionally merges submission.log files from each part as well.

Usage:
  # Merge 2 parts
  python merge_cumulative_metrics.py \\
    --parts path/to/part1/cumulative_metrics.json \\
           path/to/part2/cumulative_metrics.json \\
    --output path/to/merged_dir

  # Merge 3 parts with custom gap threshold and run-id
  python merge_cumulative_metrics.py \\
    --parts part1.json part2.json part3.json \\
    --output path/to/merged_dir \\
    --gap-threshold 600 \\
    --run-id "my-model_experiment_1_merged"

  # Legacy 3-part Claude defaults (no --parts needed)
  python merge_cumulative_metrics.py \\
    --output path/to/merged_dir
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_GAP_THRESHOLD_SECONDS = 300  # 5 minutes — any inter-step gap larger is idle


# =============================================================================
# SUBMISSION LOG HELPERS
# =============================================================================

def _runtime_to_seconds(runtime_str: str) -> int:
    """Parse RUNTIME=HH:MM:SS or H:MM:SS to total seconds."""
    m = re.search(r"RUNTIME=(\d+):(\d+):(\d+)", runtime_str)
    if not m:
        return 0
    h, m_, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return h * 3600 + m_ * 60 + s


def _seconds_to_runtime(seconds: int) -> str:
    """Format total seconds as HH:MM:SS."""
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _rewrite_step_line(line: str, new_step: int, runtime_offset_seconds: int) -> str:
    """Replace STEP=N and RUNTIME=HH:MM:SS with new values."""
    line = re.sub(r"STEP=\d+", f"STEP={new_step}", line, count=1)
    sec = _runtime_to_seconds(line)
    new_sec = runtime_offset_seconds + sec
    line = re.sub(r"RUNTIME=\d+:\d+:\d+", f"RUNTIME={_seconds_to_runtime(new_sec)}", line, count=1)
    return line


def merge_submission_logs(
    log_paths: list[Path], out_path: Path, start_time_str: str, n_parts: int,
) -> None:
    """
    Concatenate submission.log files from each part.
    Renumber STEP globally and set RUNTIME to cumulative across parts.
    """
    def read_step_lines(path: Path) -> tuple[list[str], int]:
        lines = path.read_text().splitlines()
        step_lines = [ln for ln in lines if ln.strip().startswith("STEP=")]
        last_runtime = _runtime_to_seconds(step_lines[-1]) if step_lines else 0
        return step_lines, last_runtime

    part_data: list[tuple[list[str], int]] = []
    for p in log_paths:
        part_data.append(read_step_lines(p))

    header = (
        "=== POKEMON EMERALD AGENT SUBMISSION LOG ===\n"
        f"Model: SERVER_MODE | Start Time: {start_time_str} (merged from {n_parts} parts)\n"
        "Format: STEP | POS | MAP | MILESTONE | STATE | MONEY | PARTY | ACTION | MODE | DECISION_TIME | RUNTIME | STATE_HASH | AVG_TIME | ERROR_RATE | EXPLORE_RATIO | BACKTRACK_RATIO | TIME_VAR\n"
        "========================================================================================================================\n"
    )

    out_lines = [header.rstrip()]
    step_num = 1
    cumulative_runtime = 0
    for lines, part_runtime in part_data:
        for ln in lines:
            out_lines.append(_rewrite_step_line(ln, step_num, cumulative_runtime))
            step_num += 1
        cumulative_runtime += part_runtime

    out_path.write_text("\n".join(out_lines) + "\n")


# =============================================================================
# JSON HELPERS
# =============================================================================

def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# =============================================================================
# STEP MERGING & TIMESTAMP COMPRESSION
# =============================================================================

def merge_steps(parts: list[dict]) -> tuple[list[dict], list[tuple]]:
    """
    Concatenate steps from all parts. Renumber step to global index.
    Return (merged_steps, step_sums) where step_sums[i] has cumulative
    (prompt, completion, cached, total) up to and including step i.
    """
    merged: list[dict] = []
    running_prompt = 0
    running_completion = 0
    running_cached = 0
    running_total = 0
    step_sums: list[tuple] = []

    for part in parts:
        for s in part.get("steps", []):
            step_copy = copy.deepcopy(s)
            step_copy["step"] = len(merged)
            merged.append(step_copy)

            pt = s.get("prompt_tokens", 0)
            ct = s.get("completion_tokens", 0)
            cached = s.get("cached_tokens", 0)
            tot = s.get("total_tokens", pt + ct + cached)
            running_prompt += pt
            running_completion += ct
            running_cached += cached
            running_total += tot
            step_sums.append((running_prompt, running_completion, running_cached, running_total))

    return merged, step_sums


def compress_timestamps(
    merged_steps: list[dict],
    gap_threshold: float,
) -> float:
    """
    Remove idle gaps from timestamps in-place.

    Any gap between consecutive steps larger than gap_threshold seconds is
    collapsed: all subsequent timestamps are shifted backwards by the excess.
    This makes timestamps represent continuous active run time.

    Returns total_idle_removed (seconds) for diagnostics.
    """
    total_idle_removed = 0.0

    for i in range(1, len(merged_steps)):
        prev_ts = merged_steps[i - 1]["timestamp"]
        curr_ts = merged_steps[i]["timestamp"]
        gap = curr_ts - prev_ts

        if gap > gap_threshold:
            # Keep gap_threshold worth of time (normal inter-step delay),
            # remove the rest as idle time.
            excess = gap - gap_threshold
            total_idle_removed += excess
            # Shift this and all subsequent steps back by excess
            for j in range(i, len(merged_steps)):
                merged_steps[j]["timestamp"] -= excess

    return total_idle_removed


# =============================================================================
# OBJECTIVE MERGING
# =============================================================================

def get_part_time_windows(parts: list[dict]) -> list[tuple[float, float]]:
    """Return [(start_ts, end_ts), ...] for each part from their step timestamps."""
    windows: list[tuple[float, float]] = []
    for part in parts:
        steps = part.get("steps", [])
        if not steps:
            windows.append((0.0, 0.0))
            continue
        timestamps = [s["timestamp"] for s in steps]
        windows.append((min(timestamps), max(timestamps)))
    return windows


def select_objectives_by_part_windows(
    parts: list[dict], windows: list[tuple[float, float]],
) -> list[dict]:
    """
    From all parts' objectives, keep one entry per objective_id: the one
    whose timestamp falls in the part's run window (the part that completed it).
    Sort by timestamp.
    """
    by_id: dict[str, dict] = {}
    for part_idx, part in enumerate(parts):
        low, high = windows[part_idx]
        for obj in part.get("objectives", []):
            ts = obj.get("timestamp", 0)
            if low <= ts <= high:
                oid = obj.get("objective_id", "")
                if oid not in by_id or ts < by_id[oid].get("timestamp", float("inf")):
                    by_id[oid] = copy.deepcopy(obj)
    return sorted(by_id.values(), key=lambda o: o.get("timestamp", 0))


def find_step_index_for_timestamp(steps: list[dict], timestamp: float) -> int:
    """Return the index of the last step with step['timestamp'] <= timestamp. -1 if none."""
    lo, hi = 0, len(steps) - 1
    best = -1
    while lo <= hi:
        mid = (lo + hi) // 2
        if steps[mid]["timestamp"] <= timestamp:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def remap_objective_timestamps(
    objectives: list[dict],
    original_steps: list[dict],
    compressed_steps: list[dict],
) -> None:
    """
    Update objective timestamps to match compressed step timestamps.

    For each objective, find the step (in original steps) closest to the
    objective's timestamp, then set the objective's timestamp to that step's
    compressed timestamp.
    """
    for obj in objectives:
        ts = obj.get("timestamp", 0)
        # Find nearest step in original (pre-compression) timeline
        idx = find_step_index_for_timestamp(original_steps, ts)
        if idx < 0:
            idx = 0
        obj["timestamp"] = compressed_steps[idx]["timestamp"]


def recompute_objectives(
    objectives: list[dict], merged_steps: list[dict], step_sums: list[tuple],
) -> list[dict]:
    """
    For each objective, set cumulative_* from merged steps (by timestamp),
    then set split_* = cumulative[i] - cumulative[i-1].
    """
    if not merged_steps or not step_sums:
        return objectives

    result: list[dict] = []
    prev_cumulative = (0, 0, 0, 0, 0)  # steps, prompt, completion, cached, total

    for i, obj in enumerate(objectives):
        ts = obj.get("timestamp", 0)
        step_idx = find_step_index_for_timestamp(merged_steps, ts)
        if step_idx < 0:
            step_idx = 0
        cum_steps = step_idx + 1
        cum_prompt, cum_completion, cum_cached, cum_total = step_sums[step_idx]

        out = copy.deepcopy(obj)
        out["cumulative_steps"] = cum_steps
        out["cumulative_prompt_tokens"] = cum_prompt
        out["cumulative_completion_tokens"] = cum_completion
        out["cumulative_cached_tokens"] = cum_cached
        out["cumulative_total_tokens"] = cum_total

        out["split_steps"] = cum_steps - prev_cumulative[0]
        out["split_prompt_tokens"] = cum_prompt - prev_cumulative[1]
        out["split_completion_tokens"] = cum_completion - prev_cumulative[2]
        out["split_cached_tokens"] = cum_cached - prev_cumulative[3]
        out["split_total_tokens"] = cum_total - prev_cumulative[4]

        if out.get("split_time_seconds", 0) < 0:
            out["split_time_seconds"] = 0.0

        prev_cumulative = (cum_steps, cum_prompt, cum_completion, cum_cached, cum_total)
        out["objective_index"] = i
        result.append(out)

    return result


# =============================================================================
# MAIN
# =============================================================================

def compress_existing_file(input_path: Path, gap_threshold: float) -> None:
    """
    Post-process an existing cumulative_metrics.json to compress idle gaps.

    This is useful when the file was already merged (or is a single-run file
    with long pauses) and only needs timestamp compression applied.
    Writes back to the same file.
    """
    doc = load_json(input_path)
    steps = doc.get("steps", [])
    objectives = doc.get("objectives", [])

    if not steps:
        print("No steps in file, nothing to compress.")
        return

    # Save original timestamps for objective remapping
    original_timestamps = [s["timestamp"] for s in steps]

    # Compress step timestamps
    total_idle = compress_timestamps(steps, gap_threshold)
    print(f"Compressed {total_idle:.1f}s ({total_idle / 3600:.2f}h) of idle time "
          f"(threshold: {gap_threshold}s).")

    if total_idle == 0:
        print("No idle gaps found. File unchanged.")
        return

    # Remap objective timestamps
    original_steps = [{"timestamp": ts} for ts in original_timestamps]
    remap_objective_timestamps(objectives, original_steps, steps)

    # Update top-level timing fields
    start_time = doc.get("start_time", 0)
    last_ts = steps[-1]["timestamp"]
    doc["total_run_time"] = last_ts - start_time if start_time else doc.get("total_run_time", 0)
    doc["last_update_time"] = last_ts

    # Record compression in metadata
    meta = doc.get("metadata", {})
    meta["gap_compression"] = {
        "gap_threshold_seconds": gap_threshold,
        "total_idle_removed_seconds": round(total_idle, 2),
    }
    doc["metadata"] = meta

    with open(input_path, "w") as f:
        json.dump(doc, f, indent=2)

    active_hours = (last_ts - start_time) / 3600 if start_time else 0
    print(f"Active run time after compression: {active_hours:.2f}h")
    print(f"Written: {input_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge N-part cumulative_metrics.json files with idle-gap compression.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # ── Sub-command: merge ──────────────────────────────────────────────────
    merge_parser = subparsers.add_parser(
        "merge",
        help="Merge N part files into one with gap compression.",
    )
    merge_parser.add_argument(
        "--parts",
        nargs="+",
        type=Path,
        required=True,
        help="Paths to part cumulative_metrics.json files, in chronological order.",
    )
    merge_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory (will write cumulative_metrics.json here).",
    )
    merge_parser.add_argument(
        "--gap-threshold",
        type=float,
        default=DEFAULT_GAP_THRESHOLD_SECONDS,
        help=f"Max allowed inter-step gap in seconds before compression (default: {DEFAULT_GAP_THRESHOLD_SECONDS}).",
    )
    merge_parser.add_argument(
        "--run-id",
        default=None,
        help="Custom run_id for merged metadata. Auto-generated if not set.",
    )

    # ── Sub-command: compress ───────────────────────────────────────────────
    compress_parser = subparsers.add_parser(
        "compress",
        help="Compress idle gaps in an existing cumulative_metrics.json (in-place).",
    )
    compress_parser.add_argument(
        "file",
        type=Path,
        help="Path to cumulative_metrics.json to compress.",
    )
    compress_parser.add_argument(
        "--gap-threshold",
        type=float,
        default=DEFAULT_GAP_THRESHOLD_SECONDS,
        help=f"Max allowed inter-step gap in seconds before compression (default: {DEFAULT_GAP_THRESHOLD_SECONDS}).",
    )

    args = parser.parse_args()

    if args.command == "compress":
        if not args.file.exists():
            print(f"ERROR: File not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        compress_existing_file(args.file, args.gap_threshold)
        return

    if args.command == "merge":
        _do_merge(args)
        return

    # No sub-command given — show help
    parser.print_help()
    sys.exit(1)


def _do_merge(args: argparse.Namespace) -> None:
    """Execute the merge sub-command."""
    n_parts = len(args.parts)
    if n_parts < 2:
        print("ERROR: Need at least 2 parts to merge.", file=sys.stderr)
        sys.exit(1)

    for i, p in enumerate(args.parts):
        if not p.exists():
            print(f"ERROR: Part {i + 1} not found: {p}", file=sys.stderr)
            sys.exit(1)

    parts = [load_json(p) for p in args.parts]
    print(f"Loaded {n_parts} parts.")

    # ── Get part time windows BEFORE merging (for objective dedup) ──────────
    windows = get_part_time_windows(parts)
    selected_objectives = select_objectives_by_part_windows(parts, windows)
    print(f"Selected {len(selected_objectives)} unique objectives across parts.")

    # ── Merge steps ─────────────────────────────────────────────────────────
    merged_steps, step_sums = merge_steps(parts)
    n_steps = len(merged_steps)

    # Save a shallow copy of original timestamps for objective remapping
    original_timestamps = [s["timestamp"] for s in merged_steps]

    # ── Compress idle gaps ──────────────────────────────────────────────────
    total_idle = compress_timestamps(merged_steps, args.gap_threshold)
    print(f"Compressed {total_idle:.1f}s ({total_idle / 3600:.2f}h) of idle time from timestamps "
          f"(threshold: {args.gap_threshold}s).")

    # ── Remap objective timestamps to compressed timeline ───────────────────
    original_steps = [{"timestamp": ts} for ts in original_timestamps]
    remap_objective_timestamps(selected_objectives, original_steps, merged_steps)

    # ── Recompute cumulative/split on merged data ───────────────────────────
    objectives = recompute_objectives(selected_objectives, merged_steps, step_sums)

    # ── Compute totals ──────────────────────────────────────────────────────
    total_prompt = sum(s.get("prompt_tokens", 0) for s in merged_steps)
    total_completion = sum(s.get("completion_tokens", 0) for s in merged_steps)
    total_cached = sum(s.get("cached_tokens", 0) for s in merged_steps)
    total_tokens = total_prompt + total_completion + total_cached
    total_cost = sum(p.get("total_cost", 0) for p in parts)
    total_llm_calls = sum(p.get("total_llm_calls", 0) for p in parts)

    start_time = parts[0].get("start_time", 0)
    last_compressed_ts = merged_steps[-1]["timestamp"] if merged_steps else 0
    total_run_time = last_compressed_ts - start_time if start_time else sum(
        p.get("total_run_time", 0) for p in parts
    )

    # ── Build metadata ──────────────────────────────────────────────────────
    meta = copy.deepcopy(parts[0].get("metadata", {}))
    if args.run_id:
        meta["run_id"] = args.run_id
    else:
        base_run_id = meta.get("run_id", "experiment")
        meta["run_id"] = f"{base_run_id}_merged"
    meta["start_time"] = parts[0].get("metadata", {}).get("start_time", "")
    last_meta = parts[-1].get("metadata", {})
    meta["end_time"] = last_meta.get("end_time", last_meta.get("start_time", ""))
    meta["command"] = parts[0].get("metadata", {}).get("command", "") + f" [merged from {n_parts} parts]"
    if "command_args" in meta:
        meta["command_args"] = dict(meta["command_args"], _merged_from_n_parts=n_parts)
    meta["merge_info"] = {
        "n_parts": n_parts,
        "gap_threshold_seconds": args.gap_threshold,
        "total_idle_removed_seconds": round(total_idle, 2),
        "part_files": [str(p) for p in args.parts],
    }

    # ── Write output ────────────────────────────────────────────────────────
    merged_doc = {
        "total_tokens": total_tokens,
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "cached_tokens": total_cached,
        "total_cost": total_cost,
        "total_actions": n_steps,
        "start_time": start_time,
        "total_llm_calls": total_llm_calls,
        "total_run_time": total_run_time,
        "last_update_time": last_compressed_ts,
        "metadata": meta,
        "steps": merged_steps,
        "objectives": objectives,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    out_file = args.output / "cumulative_metrics.json"
    with open(out_file, "w") as f:
        json.dump(merged_doc, f, indent=2)

    print(f"\nMerged {n_steps} steps, {len(objectives)} objectives.")
    print(f"Active run time: {total_run_time:.1f}s ({total_run_time / 3600:.2f}h)")
    print(f"Written: {out_file}")

    # ── Merge submission.log (optional) ─────────────────────────────────────
    log_paths = [p.parent / "submission.log" for p in args.parts]
    existing_logs = [p for p in log_paths if p.exists()]
    if len(existing_logs) == n_parts:
        log_out = args.output / "submission.log"
        start_time_str = parts[0].get("metadata", {}).get("start_time", "unknown")
        merge_submission_logs(existing_logs, log_out, start_time_str, n_parts)
        print(f"Written: {log_out}")
    else:
        missing = [p for p in log_paths if not p.exists()]
        print(f"Note: skipped submission.log merge (missing: {[str(m) for m in missing]})", file=sys.stderr)


if __name__ == "__main__":
    main()
