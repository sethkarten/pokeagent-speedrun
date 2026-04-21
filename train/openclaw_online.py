"""Online OpenClaw-RL: model plays via the REAL autoevolve harness, then GRPO trains.

Two-phase loop:
  Phase 1 (Rollout): Run the full harness (run.py + server + PokeAgent) with
    --backend unsloth pointing at the current adapter checkpoint. The harness
    handles state formatting, tool declarations, prompt evolution, memory,
    skills, subagents — everything. Trajectory is saved to trajectory_history.jsonl.
  Phase 2 (Train): Convert trajectory to GRPO-compatible shard, run
    grpo_offline.py --use-openclaw-judge via accelerate DDP on the fresh data.

Usage on della (4xH200):
    python train/openclaw_online.py \\
        --adapter /path/to/checkpoint \\
        --output train_runs/openclaw_online_red \\
        --rollout-steps 200 \\
        --iterations 3
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("openclaw_online")


# ── Phase 1: Rollout via the real harness ───────────────────────────

def run_rollout_harness(
    adapter_path: str,
    game: str,
    n_steps: int,
    output_dir: Path,
    device_index: int = 0,
    rom: str = "PokemonRed-GBC/pokered.gbc",
) -> dict:
    """Launch run.py with --backend unsloth and the full autoevolve scaffold.

    Returns summary dict with run_dir path and step count.
    """
    run_name = f"openclaw_online_iter"
    cmd = [
        sys.executable, "run.py",
        "--game", game,
        "--backend", "unsloth",
        "--base-model-id", adapter_path,
        "--model-name", "gemma4-openclaw",
        "--max-steps", str(n_steps),
        "--scaffold", "autoevolve",
        "--rom", rom,
        "--device-index", str(device_index),
        "--run-name", run_name,
        "--headless",
    ]
    logger.info("rollout cmd: %s", " ".join(cmd))

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device_index)

    result = subprocess.run(
        cmd, cwd=str(REPO_ROOT), env=env,
        timeout=n_steps * 120 + 600,  # generous timeout
    )

    if result.returncode != 0:
        logger.error("rollout harness exited %d", result.returncode)

    # Find the run directory (most recent run_data/run_*)
    run_dirs = sorted(glob.glob(str(REPO_ROOT / "run_data" / f"*{run_name}*")))
    if not run_dirs:
        # Fall back to most recent
        run_dirs = sorted(glob.glob(str(REPO_ROOT / "run_data" / "run_*")))
    run_dir = Path(run_dirs[-1]) if run_dirs else None

    if run_dir is None:
        logger.error("no run directory found after rollout")
        return {"steps": 0, "run_dir": None}

    # Count steps
    ss_dir = run_dir / "screenshots"
    n_actual = len(list(ss_dir.glob("*.png"))) if ss_dir.exists() else 0

    # Copy trajectory to output
    traj_src = run_dir / "trajectory_history.jsonl"
    if traj_src.exists():
        traj_dst = output_dir / "rollout_trajectory.jsonl"
        shutil.copy2(traj_src, traj_dst)
        logger.info("copied trajectory: %s → %s", traj_src, traj_dst)

    summary = {
        "steps": n_actual,
        "run_dir": str(run_dir),
        "trajectory": str(traj_dst) if traj_src.exists() else None,
        "exit_code": result.returncode,
    }
    logger.info("rollout done: %d steps from %s", n_actual, run_dir)
    return summary


# ── Convert harness trajectory to GRPO shard ────────────────────────

def trajectory_to_shard(trajectory_path: Path, shard_path: Path, run_dir: Path) -> int:
    """Convert trajectory_history.jsonl to the GRPO-compatible SFT shard format.

    The harness trajectory has: step, pre_state, post_state, tool_name, tool_args,
    raw_response, prompt, screenshot path, etc.
    """
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(trajectory_path) as fin, open(shard_path, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            t = json.loads(line)

            # Build record matching SFT dataset schema
            pre_state = t.get("pre_state", {})
            img_path = t.get("screenshot_path")
            if not img_path:
                step_n = t.get("step", n)
                candidate = run_dir / "screenshots" / f"step_{step_n:05d}.png"
                if candidate.exists():
                    img_path = str(candidate)
            if img_path and not Path(img_path).is_absolute():
                img_path = str(run_dir / img_path)

            # PRM-style teacher hint: includes trajectory context so the judge
            # can detect loops and penalize stagnation, not just score individual steps.
            location = pre_state.get("location", "?")
            in_battle = pre_state.get("is_in_battle", False)
            # Compute trajectory context for this step
            recent_locs = [json.loads(l).get("pre_state", {}).get("location", "?")
                           for l in open(trajectory_path).readlines()[max(0,n-20):n]
                           ] if trajectory_path.exists() and n > 0 else []
            loc_counts = {}
            for rl in recent_locs:
                loc_counts[rl] = loc_counts.get(rl, 0) + 1
            stuck = any(c > 10 for c in loc_counts.values())
            unique_recent = len(set(recent_locs))
            teacher_hint = (
                f"Game state: {location}. "
                f"{'Battle in progress.' if in_battle else 'Overworld exploration.'} "
                f"Recent trajectory: {unique_recent} unique locations in last {len(recent_locs)} steps. "
                f"{'STUCK: agent has been in the same location for many steps. ' if stuck else ''}"
                f"The agent should take actions that advance game progress, "
                f"explore new areas, and avoid repeating the same actions. "
                f"Penalize actions that keep the agent stuck in the same room."
            )

            record = {
                "schema_version": 2,
                "run_id": "openclaw_online",
                "step": t.get("step", n),
                "role": t.get("role", "orchestrator"),
                "interaction_type": "online_rollout",
                "screenshot_step_offset": 0,
                "image_path": img_path,
                "image_b64": t.get("image_b64"),
                "prompt": t.get("llm_prompt") or t.get("prompt", ""),
                "raw_response": teacher_hint,
                "completion": {
                    "reasoning": t.get("reasoning") or t.get("raw_response", ""),
                    "tool_calls": [],
                },
                "pre_state": pre_state,
                "post_state": t.get("post_state", {}),
                "filter_status": "keep",
            }
            fout.write(json.dumps(record) + "\n")
            n += 1

    logger.info("converted %d trajectory records to %s", n, shard_path)
    return n


# ── Phase 2: GRPO training ─────────────────────────────────────────

def run_grpo_training(
    shard_path: str,
    adapter_path: str,
    output_dir: str,
    data_root: str,
    num_gpus: int = 4,
) -> int:
    """Run grpo_offline.py with --use-openclaw-judge via accelerate DDP."""
    cmd = [
        sys.executable, "-m", "accelerate.commands.launch",
        "--num_processes", str(num_gpus),
        "--mixed_precision", "bf16",
        "-m", "train.grpo_offline",
        "--shard", shard_path,
        "--adapter", adapter_path,
        "--data-root", data_root,
        "--output", output_dir,
        "--epochs", "1",
        "--num-generations", "4",
        "--batch-size", "1",
        "--grad-accum", "1",
        "--max-prompt-length", "8192",
        "--max-completion-length", "1024",
        "--save-steps", "25",
        "--use-openclaw-judge",
    ]
    logger.info("GRPO cmd: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return result.returncode


# ── Main loop ──────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--game", type=str, default="red")
    ap.add_argument("--rom", type=str, default="PokemonRed-GBC/pokered.gbc")
    ap.add_argument("--output", type=Path, default=Path("train_runs/openclaw_online_red"))
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--rollout-steps", type=int, default=200)
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--num-gpus", type=int, default=4)
    ap.add_argument("--rollout-device", type=int, default=0)
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    data_root = str(args.data_root or REPO_ROOT)
    current_adapter = str(args.adapter)

    for iteration in range(args.iterations):
        logger.info("======== ITERATION %d/%d ========", iteration + 1, args.iterations)
        iter_dir = args.output / f"iter_{iteration + 1}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        # ── Phase 1: Rollout via real harness ──
        logger.info("PHASE 1: rollout %d steps via full harness with %s",
                     args.rollout_steps, current_adapter)
        summary = run_rollout_harness(
            adapter_path=current_adapter,
            game=args.game,
            n_steps=args.rollout_steps,
            output_dir=iter_dir,
            device_index=args.rollout_device,
            rom=args.rom,
        )
        with open(iter_dir / "rollout_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        if summary["steps"] == 0 or summary.get("trajectory") is None:
            logger.error("rollout produced no data, stopping")
            break

        # ── Convert trajectory to GRPO shard ──
        shard_path = iter_dir / "grpo_shard.jsonl"
        n_records = trajectory_to_shard(
            Path(summary["trajectory"]), shard_path,
            Path(summary["run_dir"]),
        )
        if n_records == 0:
            logger.error("empty shard, stopping")
            break

        # ── Phase 2: GRPO training ──
        train_dir = str(iter_dir / "grpo_checkpoint")
        logger.info("PHASE 2: GRPO on %d rollout records", n_records)
        rc = run_grpo_training(
            shard_path=str(shard_path),
            adapter_path=current_adapter,
            output_dir=train_dir,
            data_root=data_root,
            num_gpus=args.num_gpus,
        )
        if rc != 0:
            logger.error("GRPO failed (exit %d)", rc)
            break

        # Update adapter for next iteration
        ckpt_dirs = sorted(Path(train_dir).glob("checkpoint-*"))
        if ckpt_dirs:
            current_adapter = str(ckpt_dirs[-1])
            logger.info("next iteration adapter: %s", current_adapter)
        else:
            logger.warning("no checkpoint, reusing %s", current_adapter)

        with open(args.output / "iteration_log.jsonl", "a") as f:
            f.write(json.dumps({
                "iteration": iteration + 1,
                "rollout_steps": summary["steps"],
                "adapter": current_adapter,
                "grpo_exit": rc,
            }) + "\n")

    logger.info("===== ONLINE OPENCLAW-RL COMPLETE =====")


if __name__ == "__main__":
    main()
