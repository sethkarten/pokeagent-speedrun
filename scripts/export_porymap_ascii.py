#!/usr/bin/env python3
"""Export a porymap ASCII layout to a text file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.porymap_json_builder import build_json_map_for_llm


def export_ascii(map_name: str, output_path: Path, pokeemerald_root: Path) -> None:
    json_map = build_json_map_for_llm(map_name, pokeemerald_root)
    if not json_map or "ascii" not in json_map:
        raise RuntimeError(f"Failed to build ASCII map for '{map_name}'")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json_map["ascii"], encoding="utf-8")

    print(f"✅ Saved ASCII map for {map_name} to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export a porymap ASCII layout")
    parser.add_argument("map_name", help="Porymap map name, e.g. Route104")
    parser.add_argument(
        "--root",
        default="porymap_data",
        help="Path to pokeemerald root containing data/maps (default: porymap_data)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path. Defaults to maps_ascii/<map_name>.txt",
    )

    args = parser.parse_args()

    pokeemerald_root = Path(args.root).resolve()
    if not (pokeemerald_root / "data" / "maps").exists():
        raise FileNotFoundError(f"Could not find maps under {pokeemerald_root}")

    output_path = (
        Path(args.output).resolve()
        if args.output
        else Path("maps_ascii") / f"{args.map_name}.txt"
    )

    export_ascii(args.map_name, output_path, pokeemerald_root)


if __name__ == "__main__":
    main()
