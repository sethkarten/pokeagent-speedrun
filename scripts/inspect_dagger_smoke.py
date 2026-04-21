"""Read DAgger+PRM smoke artifacts from della scratch and commit a report back.

Since the gpu-manager log stream is returning empty for completed jobs, we use
git as the side-channel: write a markdown report and push it to the branch.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path

OUT = Path("/scratch/gpfs/CHIJ/milkkarten/pokeagent_runs/dagger_prm_red_smoke")
WORK = Path("/scratch/gpfs/CHIJ/milkkarten/pokeagent-speedrun")
REPORT = WORK / "scripts" / "_last_dagger_smoke_report.md"

lines: list[str] = ["# DAgger+PRM smoke report", ""]


def h(title: str):
    lines.append(f"## {title}")
    lines.append("")


def t(line: str = ""):
    lines.append(line)


def code(txt: str):
    lines.append("```")
    lines.append(txt.rstrip())
    lines.append("```")
    lines.append("")


def read_text(p: Path, limit: int = 10_000) -> str:
    try:
        b = p.read_bytes()
    except Exception as e:
        return f"<read error: {e}>"
    return b.decode("utf-8", errors="replace")[:limit]


def head_lines(p: Path, n: int) -> list[str]:
    try:
        with open(p) as f:
            return [f.readline() for _ in range(n) if (s := f.readline())]
    except Exception as e:
        return [f"<read error: {e}>"]


# --- top-level
h("OUTPUT DIR")
if OUT.exists():
    for c in sorted(OUT.iterdir()):
        kind = "dir" if c.is_dir() else f"{c.stat().st_size}B"
        t(f"- {c.name} ({kind})")
else:
    t("**OUTPUT DIR MISSING — job produced no output**")
t("")

# --- iteration log
h("iteration_log.jsonl")
p = OUT / "iteration_log.jsonl"
if p.exists():
    code(read_text(p))
else:
    t("_missing_")
t("")

# --- iter_1 artifacts
it1 = OUT / "iter_1"
h("iter_1/ contents")
if it1.exists():
    for c in sorted(it1.iterdir()):
        kind = "dir" if c.is_dir() else f"{c.stat().st_size}B"
        t(f"- {c.name} ({kind})")
else:
    t("_iter_1 missing_")
t("")

for name in ("rollout_summary.json", "dagger_stats.json"):
    h(f"iter_1/{name}")
    p = it1 / name
    code(read_text(p) if p.exists() else "<missing>")

# sample shard records
h("iter_1/dagger_shard.jsonl (first 2 records)")
p = it1 / "dagger_shard.jsonl"
if p.exists():
    snippets = []
    with open(p) as f:
        for i, line in enumerate(f):
            if i >= 2:
                break
            try:
                rec = json.loads(line)
            except Exception:
                snippets.append(line[:800])
                continue
            snippets.append(json.dumps({
                "step": rec.get("step"),
                "_reward": rec.get("_reward"),
                "_source": rec.get("_source"),
                "_reward_detail": rec.get("_reward_detail"),
                "prompt_len": len(rec.get("prompt", "")),
                "response_preview": (rec.get("raw_response") or "")[:200],
                "image_path": rec.get("image_path"),
                "pre_state_loc": (rec.get("pre_state") or {}).get("location"),
            }, indent=2))
    code("\n---\n".join(snippets) or "<empty>")
else:
    code("<missing>")

# sft checkpoint
h("iter_1/sft_checkpoint/")
p = it1 / "sft_checkpoint"
if p.exists():
    for c in sorted(p.iterdir()):
        kind = "dir" if c.is_dir() else f"{c.stat().st_size}B"
        t(f"- {c.name} ({kind})")
else:
    t("_missing — SFT didn't run or didn't save_")
t("")

# --- latest run_data
h("latest run_data/run_* contents")
run_data = WORK / "run_data"
if run_data.exists():
    runs = sorted(run_data.glob("run_*"))
    if runs:
        latest = runs[-1]
        t(f"latest: `{latest.name}`")
        for c in sorted(latest.iterdir()):
            kind = "dir" if c.is_dir() else f"{c.stat().st_size}B"
            t(f"- {c.name} ({kind})")
        # count screenshots
        ss = latest / "screenshots"
        if ss.exists():
            nss = len(list(ss.glob("*.png")))
            t(f"- screenshots/: {nss} png files")
    else:
        t("_no run_* dirs under run_data_")
else:
    t("_run_data missing_")
t("")

# --- cache evolution log
h("evolution_log.jsonl (last run)")
cache = WORK / ".pokeagent_cache"
if cache.exists():
    runs_c = sorted(cache.iterdir())
    if runs_c:
        evo = runs_c[-1] / "evolution_log.jsonl"
        if evo.exists():
            code(read_text(evo, limit=8000))
        else:
            t(f"_no evolution_log.jsonl under {runs_c[-1].name}_")
    else:
        t("_cache empty_")
else:
    t("_cache missing_")
t("")

# --- SLURM log tail
h("SLURM log tail (last 150 lines)")
for f in sorted(WORK.glob("slurm-*.out"))[-3:]:
    size = f.stat().st_size
    t(f"### `{f.name}` ({size}B)")
    if size > 0:
        with open(f, errors="replace") as fh:
            tail = fh.readlines()[-150:]
        code("".join(tail))
    t("")

# --- error scan: grep for key strings in slurm logs
h("Error signature scan")
patterns = [
    "Traceback",
    "ERROR",
    "Error",
    "rom not found",
    "FileNotFoundError",
    "ResponseAdapter",
    "GEMINI_API_KEY",
    "harness evolution",
    "Skill evolution",
    "Memory evolution",
    "Prompt optimization",
    "Subagent evolution",
    "PHASE 1",
    "PHASE 2",
    "PHASE 3",
    "rollout done",
    "DAgger shard",
]
scan: list[str] = []
for f in sorted(WORK.glob("slurm-*.out"))[-3:]:
    try:
        with open(f, errors="replace") as fh:
            for line in fh:
                for p in patterns:
                    if p in line:
                        scan.append(f"{f.name}: {line.rstrip()}")
                        break
    except Exception:
        pass
code("\n".join(scan[-200:]) if scan else "<no matching lines>")

# write report
REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text("\n".join(lines))
print(f"wrote {REPORT} ({REPORT.stat().st_size}B)")

# git-commit back so we can read it locally
env = os.environ.copy()
env["GIT_AUTHOR_NAME"] = "della-inspector"
env["GIT_AUTHOR_EMAIL"] = "inspector@della.local"
env["GIT_COMMITTER_NAME"] = "della-inspector"
env["GIT_COMMITTER_EMAIL"] = "inspector@della.local"

def run(cmd: list[str]) -> int:
    print("+", " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(WORK), env=env)
    return r.returncode

try:
    run(["git", "config", "--local", "user.name", "della-inspector"])
    run(["git", "config", "--local", "user.email", "inspector@della.local"])
    run(["git", "add", str(REPORT.relative_to(WORK))])
    run(["git", "commit", "-m", "ops: dagger smoke report from della-pli (automated)"])
    run(["git", "push", "origin", "HEAD:feature/generalized-harness"])
except Exception as e:
    print(f"git side-channel failed: {e}")
    sys.exit(1)
