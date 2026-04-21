"""Read + print DAgger+PRM smoke artifacts from della scratch."""
from __future__ import annotations
import json
from pathlib import Path

OUT = Path("/scratch/gpfs/CHIJ/milkkarten/pokeagent_runs/dagger_prm_red_smoke")


def _cat(p: Path, head: int = 40):
    print(f"\n===== {p} (exists={p.exists()}) =====")
    if not p.exists():
        return
    if p.is_dir():
        for c in sorted(p.iterdir())[:30]:
            print("  ", c.name, "dir" if c.is_dir() else c.stat().st_size)
        return
    with open(p) as f:
        for i, line in enumerate(f):
            if i >= head:
                print(f"  ... (truncated)")
                break
            print(line.rstrip())


print("===== OUTPUT DIR =====")
if OUT.exists():
    for c in sorted(OUT.iterdir()):
        print("  ", c.name, "dir" if c.is_dir() else c.stat().st_size)
else:
    print("OUTPUT DIR DOES NOT EXIST — job never wrote anything")

_cat(OUT / "iteration_log.jsonl")
_cat(OUT / "iter_1" / "rollout_summary.json", head=30)
_cat(OUT / "iter_1" / "dagger_stats.json", head=30)
_cat(OUT / "iter_1" / "dagger_shard.jsonl", head=3)
_cat(OUT / "iter_1" / "rollout_trajectory.jsonl", head=2)

# Also check run_data/run_* for most recent harness run
run_data = Path("/scratch/gpfs/CHIJ/milkkarten/pokeagent-speedrun/run_data")
if run_data.exists():
    runs = sorted(run_data.glob("run_*"))
    if runs:
        print(f"\n===== latest run_dir: {runs[-1]} =====")
        for c in sorted(runs[-1].iterdir()):
            print("  ", c.name)

# And cache
cache = Path("/scratch/gpfs/CHIJ/milkkarten/pokeagent-speedrun/.pokeagent_cache")
if cache.exists():
    for run_c in sorted(cache.iterdir())[-1:]:
        evo = run_c / "evolution_log.jsonl"
        if evo.exists():
            _cat(evo, head=20)

# SLURM logs
slurm = Path("/scratch/gpfs/CHIJ/milkkarten/pokeagent-speedrun")
for f in sorted(slurm.glob("slurm-*.out"))[-2:]:
    print(f"\n===== SLURM log: {f.name} =====")
    size = f.stat().st_size
    print(f"size={size}")
    if size > 0:
        with open(f) as fh:
            lines = fh.readlines()
            # last 80 lines
            for line in lines[-80:]:
                print(line.rstrip())
