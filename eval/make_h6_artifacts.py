"""Produce H6 paper artifacts from existing eval + training logs.

Outputs written to /media/milkkarten/data/pokeagent-speedrun/autoevolve-latex/analysis/artifacts/:
- h6_grpo_rewards.jsonl : per-step reward from della GRPO runs
- h6_grpo_eval.json     : post-GRPO eval, same metrics as SFT tables

Usage: uv run eval/make_h6_artifacts.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PAPER_ART = Path("/media/milkkarten/data/pokeagent-speedrun/autoevolve-latex/analysis/artifacts")
PAPER_ART.mkdir(parents=True, exist_ok=True)


# ── h6_grpo_rewards.jsonl ────────────────────────────────────────────

RUN_META = {
    "emerald": {"path": REPO_ROOT / "data/grpo_training_logs/emerald_26b.jsonl", "model_size": "26B", "game": "emerald"},
    "red":     {"path": REPO_ROOT / "data/grpo_training_logs/red_26b.jsonl",     "model_size": "26B", "game": "red"},
}

def dedupe_by_step(rows: list[dict]) -> list[dict]:
    """losses.jsonl has one row per rank per step. Keep one row per step."""
    seen = {}
    for r in rows:
        step = r.get("step")
        if step is None:
            continue
        if step not in seen:
            seen[step] = r
    return [seen[k] for k in sorted(seen.keys())]


def compact_row(row: dict, model_size: str, game: str) -> dict:
    step = int(row.get("step", 0))
    mean_reward = row.get("reward")
    reward_std = row.get("reward_std")
    tool_match = row.get("rewards/tool_match_reward/mean")
    action_sim = row.get("rewards/action_similarity_reward/mean")
    state_acc = row.get("rewards/state_accuracy_reward/mean")
    fmt = row.get("rewards/format_reward/mean")
    # Use tool_match as the "oracle_agreement" proxy for offline GRPO:
    # it's the frac of completions whose tool name matches the Gemini teacher.
    oracle_agreement = tool_match
    out = {
        "step": step,
        "mean_reward": mean_reward,
        "reward_std": reward_std,
        "oracle_agreement": oracle_agreement,
        "model_size": model_size,
        "game": game,
        "_components": {
            "tool_match": tool_match,
            "action_similarity": action_sim,
            "state_accuracy": state_acc,
            "format": fmt,
            "loss": row.get("loss"),
        },
    }
    return out


def build_grpo_rewards() -> Path:
    out_lines = []
    for key, meta in RUN_META.items():
        rows = [json.loads(l) for l in meta["path"].read_text().splitlines() if l.strip()]
        rows = dedupe_by_step(rows)
        for r in rows:
            out_lines.append(json.dumps(compact_row(r, meta["model_size"], meta["game"])))
    out_path = PAPER_ART / "h6_grpo_rewards.jsonl"
    out_path.write_text("\n".join(out_lines) + "\n")
    print(f"wrote {out_path} ({len(out_lines)} rows)")
    return out_path


# ── h6_grpo_eval.json ────────────────────────────────────────────────

EVAL_FILES = {
    "emerald": REPO_ROOT / "data/eval_emerald_paper.json",
    "red": REPO_ROOT / "data/eval_red_paper.json",
}

GRPO_MODEL_BY_GAME = {
    "emerald": "gemma4-emerald-grpo:26b",
    "red": "gemma4-red-grpo:26b",
}

# Matches the SFT h6 table metric set
METRICS = ["tool_format", "actionable", "grounding", "action_relevance",
           "reasoning_similarity", "hallucination", "degenerate", "tok_s"]


def _metric_mean(rows, metric):
    vals = [r["scores"][metric] for r in rows
            if isinstance(r, dict) and metric in r.get("scores", {}) and r["scores"][metric] is not None]
    return (sum(vals) / len(vals)) if vals else None


def _by_state_type(rows, st, metric):
    filtered = [r for r in rows if r.get("state_type") == st]
    return _metric_mean(filtered, metric)


def build_grpo_eval() -> Path:
    out = {
        "_meta": {
            "note": "Post-GRPO eval for Gemma-4 26B on Emerald + Red. "
                    "Same judge + metrics as h6_eval_full_*.tex. Training: offline GRPO from SFT 26B "
                    "adapter, Gemini-teacher reference rewards, 103 unique steps (Emerald), 140 steps (Red).",
            "judge": "gemini-2.5-flash",
            "n_samples": 20,
            "prompt_mode": "real",
            "sizes_covered": ["26B"],
            "sizes_missing": ["31B", "E4B", "E2B"],
            "online_rl_status": "pending (OpenClaw-RL training run to be submitted; this file reports offline GRPO only)",
        },
    }
    for game, path in EVAL_FILES.items():
        data = json.loads(path.read_text())
        model_id = GRPO_MODEL_BY_GAME[game]
        rows = data.get(model_id)
        if rows is None:
            print(f"WARN: missing {model_id} in {path}")
            continue
        # drop non-list entries
        rows = [r for r in rows if isinstance(r, dict)]
        entry = {
            "model": model_id,
            "size": "26B",
            "game": game,
            "overall": {m: _metric_mean(rows, m) for m in METRICS},
            "overworld": {m: _by_state_type(rows, "overworld", m) for m in METRICS},
            "battle": {m: _by_state_type(rows, "battle", m) for m in METRICS},
            "n_rows": len(rows),
        }
        out[game] = entry
    out_path = PAPER_ART / "h6_grpo_eval.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")
    return out_path


if __name__ == "__main__":
    build_grpo_rewards()
    build_grpo_eval()
