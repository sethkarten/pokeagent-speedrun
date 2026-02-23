"""Print all unique coll_map symbols found across all processed_map/*.py files.

Run:
    python pokemon_red_env/test/inspect_coll_symbols.py
"""

import importlib.util
import pathlib
from collections import Counter

PROCESSED_MAP_DIR = pathlib.Path(__file__).parent.parent / "data" / "processed_map"

symbols: Counter = Counter()

files = sorted(PROCESSED_MAP_DIR.glob("*.py"))
if not files:
    print(f"No .py files found under {PROCESSED_MAP_DIR}")
    raise SystemExit(1)

for path in files:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for row in getattr(mod, "coll_map", []):
        for cell in row:
            symbols[cell] += 1

print(f"Scanned {len(files)} coll_map files\n")
print(f"{'Symbol':<40}  {'Count':>8}")
print("-" * 52)
for sym, count in sorted(symbols.items(), key=lambda x: -x[1]):
    print(f"{sym:<40}  {count:>8}")

print(f"\n{len(symbols)} unique symbols total")
