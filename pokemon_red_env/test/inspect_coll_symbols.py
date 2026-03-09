"""Print all unique grid symbols found across all processed_map/*.json files.

Run:
    python pokemon_red_env/test/inspect_coll_symbols.py
"""

import json
import pathlib
from collections import Counter

PROCESSED_MAP_DIR = pathlib.Path(__file__).parent.parent / "data" / "processed_map"

symbols: Counter = Counter()

files = sorted(PROCESSED_MAP_DIR.glob("*.json"))
if not files:
    print(f"No .json files found under {PROCESSED_MAP_DIR}")
    raise SystemExit(1)

for path in files:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for row in data.get("grid", []):
        for cell in row:
            symbols[cell] += 1

print(f"Scanned {len(files)} grid files\n")
print(f"{'Symbol':<40}  {'Count':>8}")
print("-" * 52)
for sym, count in sorted(symbols.items(), key=lambda x: -x[1]):
    print(f"{sym:<40}  {count:>8}")

print(f"\n{len(symbols)} unique symbols total")
