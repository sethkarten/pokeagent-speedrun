"""
Parse a Pokémon Showdown ladder TSV and report GXE statistics.

Usage:
    python ladder_gxe_stats.py <path_to_ladder.tsv>

Expected TSV format (tab-separated, copied from Showdown leaderboard):
    rank  username  elo  gxe%  glicko ± uncertainty  coil
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np


def parse_gxe_from_ladder(path: str) -> np.ndarray:
    """Extract all GXE percentages from a Showdown ladder TSV."""
    gxe_values = []
    with open(path) as f:
        for line in f:
            match = re.search(r"(\d+\.\d+)%", line)
            if match:
                gxe_values.append(float(match.group(1)))
    return np.array(gxe_values)


def report(gxe: np.ndarray, name: str) -> None:
    """Print a summary of GXE statistics."""
    print(f"\n{'=' * 60}")
    print(f"  {name}  —  {len(gxe)} players")
    print(f"{'=' * 60}")

    print(f"  Min:      {gxe.min():.1f}%")
    print(f"  Max:      {gxe.max():.1f}%")
    print(f"  Mean:     {gxe.mean():.1f}%")
    print(f"  Median:   {np.median(gxe):.1f}%")
    print(f"  Std Dev:  {gxe.std():.1f}%")

    q1 = np.percentile(gxe, 25)
    q3 = np.percentile(gxe, 75)
    print(f"  IQR:      [{q1:.1f}%, {q3:.1f}%]  (width {q3 - q1:.1f}%)")

    for ci_level, lo_p, hi_p in [(90, 5, 95), (95, 2.5, 97.5)]:
        lo = np.percentile(gxe, lo_p)
        hi = np.percentile(gxe, hi_p)
        print(f"  {ci_level}% range: [{lo:.1f}%, {hi:.1f}%]")

    print(f"\n  Bottom 5: {sorted(gxe)[:5]}")
    print(f"  Top 5:    {sorted(gxe)[-5:]}")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tsv", nargs="+", help="Path(s) to ladder TSV file(s)")
    args = parser.parse_args()

    for tsv_path in args.tsv:
        p = Path(tsv_path)
        if not p.exists():
            print(f"File not found: {tsv_path}", file=sys.stderr)
            continue
        name = p.stem.replace("_ladder", "").replace("_", " ").upper()
        gxe = parse_gxe_from_ladder(tsv_path)
        if len(gxe) == 0:
            print(f"No GXE values found in {tsv_path}", file=sys.stderr)
            continue
        report(gxe, name)


if __name__ == "__main__":
    main()
