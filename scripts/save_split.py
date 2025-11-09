#!/usr/bin/env python3
"""
Copy the latest cached emulator artifacts into an Emerald-GBAdvance split.

This script is useful when you already have the desired `.state`, milestone,
and map stitcher files inside `.pokeagent_cache/` (for example after the
server creates a checkpoint). It does NOT contact the running server—it simply
copies files that exist locally in the cache.

Example:
    python scripts/save_split.py 06_road_3
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from typing import Iterable, List, Tuple


def resolve_path(base_dir: str, filename: str) -> str:
    """Return `filename` if absolute, otherwise join it with `base_dir`."""
    if os.path.isabs(filename):
        return filename
    return os.path.join(base_dir, filename)


def copy_file(src: str, dst: str, overwrite: bool) -> Tuple[bool, str]:
    """Copy file from `src` to `dst` respecting the overwrite flag."""
    if not os.path.exists(src):
        return False, f"Source file does not exist: {src}"

    if os.path.exists(dst) and not overwrite:
        return False, f"Destination already exists (use --overwrite): {dst}"

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return True, f"Copied {src} -> {dst}"


def validate_required_files(paths: Iterable[str]) -> List[str]:
    """Return a list of missing files from `paths`."""
    return [path for path in paths if not os.path.exists(path)]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy cached emulator state + metadata into Emerald-GBAdvance/splits."
    )
    parser.add_argument("name", help="Name of the split to create (e.g. 06_road_3)")
    parser.add_argument(
        "--cache-dir",
        default=".pokeagent_cache",
        help="Directory containing cached state/milestone/map stitcher files "
        "(default: .pokeagent_cache)",
    )
    parser.add_argument(
        "--state-file",
        default="checkpoint.state",
        help="State file inside cache directory to copy (default: checkpoint.state)",
    )
    parser.add_argument(
        "--milestones-file",
        default="milestones_progress.json",
        help="Milestone progress file inside cache directory (default: milestones_progress.json)",
    )
    parser.add_argument(
        "--map-stitcher-file",
        default="map_stitcher_data.json",
        help="Map stitcher file inside cache directory (default: map_stitcher_data.json)",
    )
    parser.add_argument(
        "--grids-file",
        default="checkpoint_grids.json",
        help="Optional persistent grids file inside cache directory (default: checkpoint_grids.json)",
    )
    parser.add_argument(
        "--splits-root",
        default="Emerald-GBAdvance/splits",
        help="Root directory that contains split folders (default: Emerald-GBAdvance/splits)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing files in the destination split directory",
    )
    parser.add_argument(
        "--no-grids",
        action="store_true",
        help="Skip copying the grids file even if it exists in the cache",
    )
    args = parser.parse_args()

    cache_dir = os.path.abspath(args.cache_dir)
    splits_root = os.path.abspath(args.splits_root)
    split_name = args.name.strip()
    if not split_name:
        parser.error("Split name must be a non-empty string.")

    # Compute source paths inside cache
    state_src = resolve_path(cache_dir, args.state_file)
    milestones_src = resolve_path(cache_dir, args.milestones_file)
    map_stitcher_src = resolve_path(cache_dir, args.map_stitcher_file)
    grids_src = resolve_path(cache_dir, args.grids_file)

    required_sources = [state_src, milestones_src, map_stitcher_src]
    missing_sources = validate_required_files(required_sources)

    if missing_sources:
        missing = "\n  - ".join(missing_sources)
        print(
            f"❌ Cannot create split because these required files are missing:\n  - {missing}",
            file=sys.stderr,
        )
        return 1

    split_dir = os.path.abspath(os.path.join(splits_root, split_name))
    os.makedirs(split_dir, exist_ok=True)

    # Destination files
    state_dst = os.path.join(split_dir, f"{split_name}.state")
    milestones_dst = os.path.join(split_dir, f"{split_name}_milestones.json")
    map_stitcher_dst = os.path.join(split_dir, f"{split_name}_map_stitcher.json")
    grids_dst = os.path.join(split_dir, f"{split_name}_grids.json")

    results: List[Tuple[bool, str]] = []
    results.append(copy_file(state_src, state_dst, args.overwrite))
    results.append(copy_file(milestones_src, milestones_dst, args.overwrite))
    results.append(copy_file(map_stitcher_src, map_stitcher_dst, args.overwrite))

    if not args.no_grids and os.path.exists(grids_src):
        results.append(copy_file(grids_src, grids_dst, args.overwrite))
    elif not args.no_grids:
        results.append(
            (
                False,
                f"Grids file not found in cache (skipped): {grids_src}",
            )
        )

    success = True
    for ok, message in results:
        if ok:
            print(f"✅ {message}")
        else:
            print(f"⚠️  {message}")
            # Missing grids is not fatal; other failures are.
            if "Grids file" not in message:
                success = False

    if success:
        print("\n🎉 Split saved to:")
        print(f"  • {state_dst}")
        print(f"  • {milestones_dst}")
        print(f"  • {map_stitcher_dst}")
        if os.path.exists(grids_dst) and not args.no_grids:
            print(f"  • {grids_dst}")
        return 0

    print("\n❌ Encountered errors while copying split data. Review messages above.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""
Utility script to capture the current emulator state, milestones, and map stitcher
data into the Emerald-GBAdvance splits directory.

Usage:
    python scripts/save_split.py 06_road_3

By default this will create (or reuse) `Emerald-GBAdvance/splits/06_road_3/`
and instruct the running server (http://127.0.0.1:8238) to save the current
state to `06_road_3.state`. The server's save_state endpoint also writes the
matching milestones and map stitcher files alongside the state file.
"""

import argparse
import os
import shutil
import sys
import time
from typing import Iterable, Tuple

import requests


def wait_for_files(paths: Iterable[str], timeout: float = 5.0) -> Tuple[bool, Iterable[str]]:
    """Wait up to `timeout` seconds for all files in `paths` to exist."""
    deadline = time.time() + timeout
    pending = set(paths)

    while pending and time.time() < deadline:
        existing = {path for path in list(pending) if os.path.exists(path)}
        pending.difference_update(existing)
        if pending:
            time.sleep(0.25)

    return (len(pending) == 0, pending)


def copy_if_missing(src: str, dst: str) -> bool:
    """Copy `src` to `dst` if `dst` does not already exist."""
    if os.path.exists(dst):
        return True
    if not os.path.exists(src):
        return False

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Save the current emulator state and related data into splits directory.")
    parser.add_argument("name", help="Split name (e.g. 06_road_3)")
    parser.add_argument(
        "--splits-root",
        default="Emerald-GBAdvance/splits",
        help="Root directory where split folders live (default: Emerald-GBAdvance/splits)",
    )
    parser.add_argument(
        "--server",
        default="http://127.0.0.1:8238",
        help="Base URL for the running pokeagent server (default: http://127.0.0.1:8238)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout when calling the save_state endpoint (seconds, default: 30)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing state/milestone/map-stitcher files",
    )
    args = parser.parse_args()

    split_name = args.name.strip()
    if not split_name:
        parser.error("Split name must be a non-empty string.")

    splits_root = os.path.abspath(args.splits_root)
    split_dir = os.path.abspath(os.path.join(splits_root, split_name))
    os.makedirs(split_dir, exist_ok=True)

    state_path = os.path.abspath(os.path.join(split_dir, f"{split_name}.state"))
    milestone_path = os.path.abspath(os.path.join(split_dir, f"{split_name}_milestones.json"))
    map_stitcher_path = os.path.abspath(os.path.join(split_dir, f"{split_name}_map_stitcher.json"))
    grids_path = os.path.abspath(os.path.join(split_dir, f"{split_name}_grids.json"))

    if not args.overwrite:
        for path in (state_path, milestone_path, map_stitcher_path):
            if os.path.exists(path):
                parser.error(f"File already exists: {path}. Use --overwrite to replace it.")

    payload = {"filepath": state_path}
    url = f"{args.server.rstrip('/')}/save_state"

    try:
        response = requests.post(url, json=payload, timeout=args.timeout)
    except requests.exceptions.RequestException as exc:
        print(f"❌ Failed to contact server at {url}: {exc}", file=sys.stderr)
        return 1

    if response.status_code != 200:
        print(f"❌ Server returned {response.status_code}: {response.text}", file=sys.stderr)
        return 1

    print(f"✅ Requested save to {state_path}")

    # Wait briefly for files to be written by the server.
    expected_files = [state_path, milestone_path, map_stitcher_path]
    all_present, missing = wait_for_files(expected_files, timeout=5.0)

    if not all_present:
        # Attempt to fill gaps using cache artifacts if available.
        cache_dir = os.path.abspath(".pokeagent_cache")
        cache_map = os.path.join(cache_dir, "map_stitcher_data.json")
        if map_stitcher_path in missing:
            if copy_if_missing(cache_map, map_stitcher_path):
                missing.discard(map_stitcher_path)
                print(f"ℹ️  Copied map stitcher data from cache to {map_stitcher_path}")

        cache_milestones = os.path.join(cache_dir, "milestones_progress.json")
        if milestone_path in missing:
            if copy_if_missing(cache_milestones, milestone_path):
                missing.discard(milestone_path)
                print(f"ℹ️  Copied milestone data from cache to {milestone_path}")

    if missing:
        missing_list = "\n  - ".join(sorted(missing))
        print(
            "⚠️  Save request completed but some files are missing:\n"
            f"  - {missing_list}\n"
            "Check server logs or re-run with --overwrite once the emulator is ready.",
            file=sys.stderr,
        )
        return 1

    print("🎉 Split files created:")
    print(f"  • {state_path}")
    print(f"  • {milestone_path}")
    print(f"  • {map_stitcher_path}")
    if os.path.exists(grids_path):
        print(f"  • {grids_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

