"""DAgger + PRM-filter online training.

Three-phase loop per iteration:
  Phase 1 (Rollout): Run run.py --backend unsloth --scaffold autoevolve with
    the current adapter. Full ContinualHarness runs (prompt/memory/skill/
    subagent evolution) and writes trajectory_history.jsonl.

  Phase 2 (Score + Relabel): For each step:
      - PRM judge scores the agent's response in context (reward in [0,1]).
      - If reward < --relabel-threshold, query the teacher
        (gemini-3-flash-preview) on the same (image, prompt) for the correct
        action. Replace the agent's response with the teacher's response in
        the training shard.
      - Else keep the agent's response (on-policy, already good).
    The mixed shard is written to iter_N/dagger_shard.jsonl.

  Phase 3 (SFT): Train sft_run.py on the relabeled shard to produce the next
    adapter checkpoint. Next iteration rolls out with that adapter.

Usage:
    python train/dagger_prm.py \
        --adapter adapters/26b_red_v3 \
        --base-model-id unsloth/gemma-4-26B-A4B-it \
        --game red \
        --rom PokemonRed-GBC/pokered.gbc \
        --output train_runs/dagger_prm_red \
        --rollout-steps 200 \
        --iterations 3 \
        --relabel-threshold 0.3
"""

from __future__ import annotations

import argparse
import atexit
import base64
import glob
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")

# HF cache auto-discovery + offline mode. Della compute nodes have no
# internet, so Unsloth's `model_info()` hub check DNS-fails during tokenizer
# load unless HF_HUB_OFFLINE=1 is set. Mirrors sft_run.py's boot logic.
_hf_candidates = [
    os.environ.get("HF_HOME"),
    "/scratch/gpfs/CHIJ/milkkarten/huggingface",
    "/mnt/storage/models/huggingface",
    "/data1/milkkarten/.cache/huggingface",
]
for _c in _hf_candidates:
    if _c and os.path.isdir(_c):
        os.environ["HF_HOME"] = _c
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        break

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("dagger_prm")


# ── SOCKS5 tunnel: compute node → login node → internet ─────────────
# Della compute nodes have no external DNS/HTTP. We open an ssh -D tunnel
# back to the login node, which has internet, and route Gemini API calls
# (judge + teacher) through it via HTTPS_PROXY.

_SOCKS_PROC: subprocess.Popen | None = None


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _pick_port() -> int:
    for p in range(18890, 18990):
        if _port_free(p):
            return p
    raise RuntimeError("no free local port for SOCKS tunnel")


def start_socks_tunnel(login_host: str) -> int | None:
    """Start `ssh -N -D 127.0.0.1:PORT login_host` and export HTTPS_PROXY.

    Returns the port if the tunnel came up, or None if it failed. Caller
    should log the result; downstream code reads HTTPS_PROXY from env.
    """
    global _SOCKS_PROC
    port = _pick_port()
    cmd = [
        "ssh", "-N",
        "-D", f"127.0.0.1:{port}",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "ServerAliveInterval=30",
        "-o", "BatchMode=yes",
        login_host,
    ]
    logger.info("starting SOCKS tunnel: %s", " ".join(cmd))
    try:
        _SOCKS_PROC = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        logger.warning("ssh binary not found; proceeding without SOCKS")
        return None

    # Wait up to 15s for the port to accept connections
    deadline = time.time() + 15
    while time.time() < deadline:
        if _SOCKS_PROC.poll() is not None:
            err = _SOCKS_PROC.stderr.read().decode() if _SOCKS_PROC.stderr else ""
            logger.warning("ssh tunnel exited early: %s", err[:800])
            return None
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            try:
                s.connect(("127.0.0.1", port))
                proxy = f"socks5h://127.0.0.1:{port}"
                os.environ["HTTPS_PROXY"] = proxy
                os.environ["HTTP_PROXY"] = proxy
                # localhost traffic (local server) must NOT go through the tunnel
                os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
                logger.info("SOCKS tunnel up on port %d, HTTPS_PROXY=%s", port, proxy)
                atexit.register(_stop_socks_tunnel)
                return port
            except OSError:
                time.sleep(0.5)
    logger.warning("SOCKS tunnel did not come up within 15s")
    _stop_socks_tunnel()
    return None


def _stop_socks_tunnel():
    global _SOCKS_PROC
    if _SOCKS_PROC is None:
        return
    try:
        _SOCKS_PROC.terminate()
        _SOCKS_PROC.wait(timeout=5)
    except Exception:
        try:
            _SOCKS_PROC.kill()
        except Exception:
            pass
    _SOCKS_PROC = None


# ── Phase 1: Rollout via the real harness ───────────────────────────

def run_rollout_harness(
    adapter_path: str,
    base_model_id: str,
    game: str,
    n_steps: int,
    output_dir: Path,
    device_index: int = 0,
    rom: str = "PokemonRed-GBC/pokered.gbc",
    model_name: str = "gemma4-dagger",
) -> dict:
    run_name = f"dagger_iter"
    cmd = [
        sys.executable, "run.py",
        "--game", game,
        "--backend", "unsloth",
        "--base-model-id", adapter_path,
        "--model-name", model_name,
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
    # Force HF offline in the child so Unsloth's tokenizer hub check DNS-fails
    # cleanly into a local cache lookup. Without these the rollout dies at
    # "Unsloth: The tokenizer is weirdly not loaded?".
    for k, v in (
        ("HF_HUB_OFFLINE", "1"),
        ("TRANSFORMERS_OFFLINE", "1"),
        ("HF_DATASETS_OFFLINE", "1"),
    ):
        env.setdefault(k, v)
    if "HF_HOME" in os.environ:
        env["HF_HOME"] = os.environ["HF_HOME"]

    result = subprocess.run(
        cmd, cwd=str(REPO_ROOT), env=env,
        timeout=n_steps * 120 + 600,
    )
    if result.returncode != 0:
        logger.error("rollout harness exited %d", result.returncode)

    run_dirs = sorted(glob.glob(str(REPO_ROOT / "run_data" / f"*{run_name}*")))
    if not run_dirs:
        run_dirs = sorted(glob.glob(str(REPO_ROOT / "run_data" / "run_*")))
    run_dir = Path(run_dirs[-1]) if run_dirs else None

    if run_dir is None:
        logger.error("no run directory found after rollout")
        return {"steps": 0, "run_dir": None}

    ss_dir = run_dir / "screenshots"
    n_actual = len(list(ss_dir.glob("*.png"))) if ss_dir.exists() else 0

    traj_src = run_dir / "trajectory_history.jsonl"
    traj_dst = None
    if traj_src.exists():
        traj_dst = output_dir / "rollout_trajectory.jsonl"
        shutil.copy2(traj_src, traj_dst)
        logger.info("copied trajectory: %s → %s", traj_src, traj_dst)

    # Count harness-evolution events so we can verify evolution is actually
    # running rather than silently erroring out. The evolver writes to the
    # per-run cache dir (.pokeagent_cache/<run_id>/evolution_log.jsonl);
    # run_dir name ends in the run_id.
    n_evolutions = 0
    run_id = run_dir.name
    for cand in [
        REPO_ROOT / ".pokeagent_cache" / run_id / "evolution_log.jsonl",
        run_dir / "evolution_log.jsonl",
        run_dir / "end_state" / "evolution_log.jsonl",
    ]:
        if cand.exists():
            with open(cand) as f:
                n_evolutions = sum(1 for line in f if line.strip())
            break

    summary = {
        "steps": n_actual,
        "run_dir": str(run_dir),
        "trajectory": str(traj_dst) if traj_dst else None,
        "exit_code": result.returncode,
        "evolutions": n_evolutions,
    }
    logger.info("rollout done: %d steps, %d evolutions, from %s",
                n_actual, n_evolutions, run_dir)
    return summary


# ── Phase 2: Score + Relabel ────────────────────────────────────────

def _resolve_screenshot(run_dir: Path, step_n: int, t: dict) -> str | None:
    """The harness trajectory schema doesn't store screenshot paths; reconstruct."""
    p = t.get("screenshot_path")
    if p:
        if not Path(p).is_absolute():
            p = str(run_dir / p)
        if Path(p).exists():
            return p
    cand = run_dir / "screenshots" / f"step_{int(step_n):05d}.png"
    if cand.exists():
        return str(cand)
    # last-resort: unpadded
    cand = run_dir / "screenshots" / f"step_{int(step_n)}.png"
    return str(cand) if cand.exists() else None


def _score_traj_records(records: list[dict], run_dir: Path, judge_model: str) -> list[dict]:
    """Run PRM judge on each trajectory record; attach reward + detail."""
    from train.openclaw_judge import _score_one, _build_state_summary, _load_image_bytes

    for i, t in enumerate(records):
        pre_state = t.get("pre_state", {})
        location = pre_state.get("location", "?")
        in_battle = pre_state.get("is_in_battle", False)

        window = [r.get("pre_state", {}).get("location", "?")
                  for r in records[max(0, i - 20):i]]
        loc_counts: dict[str, int] = {}
        for rl in window:
            loc_counts[rl] = loc_counts.get(rl, 0) + 1
        stuck = any(c > 10 for c in loc_counts.values())
        unique_recent = len(set(window))
        teacher_hint = (
            f"Game state: {location}. "
            f"{'Battle in progress.' if in_battle else 'Overworld exploration.'} "
            f"Recent trajectory: {unique_recent} unique locations in last {len(window)} steps. "
            f"{'STUCK: agent has been in the same location for many steps. ' if stuck else ''}"
            f"Actions that advance game progress, explore new areas, or break loops "
            f"score high. Actions that repeat recent behavior when stuck score low."
        )

        # Real trajectory schema: reasoning = full student response, llm_prompt = full prompt
        model_text = t.get("reasoning") or t.get("raw_response") or ""
        state_summary = _build_state_summary(pre_state)
        img_path = _resolve_screenshot(run_dir, t.get("step", i), t)
        img = _load_image_bytes(img_path, None)

        reward, detail = _score_one(
            model_text=model_text,
            teacher_text=teacher_hint,
            state_summary=state_summary,
            image_bytes=img,
            judge_model=judge_model,
        )
        t["_reward"] = reward
        t["_reward_detail"] = detail
        t["_teacher_hint"] = teacher_hint
        t["_image_path_abs"] = img_path
    return records


# Teacher (gemini-3-flash-preview) — image + prompt → expert response in the
# harness's tool-call format. We reuse openclaw_judge's REST/session plumbing.

def _teacher_query(
    image_path: str | None,
    prompt_text: str,
    teacher_hint: str,
    teacher_model: str = "gemini-3-flash-preview",
    timeout_s: float = 90.0,
) -> str | None:
    from train.openclaw_judge import _get_session, _load_image_bytes, _load_env

    _load_env()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    img_bytes = _load_image_bytes(image_path, None) if image_path else None

    # Keep the teacher aligned with the harness tool-call format so its outputs
    # drop straight into SFT without re-parsing. The agent's prompt already
    # describes the tool schema; we just need a directive prefix.
    system_directive = (
        "You are an expert Pokemon player guiding a student agent. "
        "The student is stuck or making poor choices.\n\n"
        f"Context: {teacher_hint}\n\n"
        "Produce the SAME response the student should have produced: "
        "a single tool call in the student's format (e.g. `call:press_buttons` "
        "with ANALYZE/PLAN/ACTION sections, or the bracket `[tool_name]` "
        "format if that's what the prompt specifies), plus concise reasoning. "
        "Match the format exactly — do not wrap in markdown, do not add "
        "preamble, do not explain yourself outside the ANALYZE/PLAN sections. "
        "Pick an action that BREAKS LOOPS and MAKES PROGRESS."
    )
    full_text = system_directive + "\n\n---\nSTUDENT PROMPT:\n" + prompt_text

    parts: list[dict] = [{"text": full_text}]
    if img_bytes:
        parts.append({
            "inlineData": {
                "mimeType": "image/png",
                "data": base64.b64encode(img_bytes).decode("ascii"),
            }
        })

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"models/{teacher_model}:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 2048},
    }

    s = _get_session()
    for attempt in range(3):
        try:
            r = s.post(url, json=payload, timeout=timeout_s)
            r.raise_for_status()
            data = r.json()
            candidates = data.get("candidates", [])
            if candidates:
                content = candidates[0].get("content", {})
                for p in content.get("parts", []):
                    if p.get("text"):
                        return p["text"]
            return None
        except Exception as e:
            logger.warning("teacher query error (attempt %d): %s", attempt + 1, e)
            time.sleep(1 + attempt * 2)
    return None


def score_and_relabel(
    trajectory_path: Path,
    run_dir: Path,
    shard_path: Path,
    judge_model: str,
    teacher_model: str,
    relabel_threshold: float,
    max_teacher_calls: int | None = None,
) -> dict:
    """Write a DAgger shard: teacher response for low-reward steps, agent for high.

    Returns stats dict: total, relabeled, kept, teacher_calls, reward_mean,
    reward_low_count.
    """
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    with open(trajectory_path) as fin:
        for line in fin:
            if line.strip():
                records.append(json.loads(line))

    logger.info("PRM scoring %d records with %s", len(records), judge_model)
    records = _score_traj_records(records, run_dir, judge_model)

    rewards = [r.get("_reward", 0.0) for r in records]
    reward_mean = sum(rewards) / max(len(rewards), 1)
    low = [r for r in records if r.get("_reward", 0.0) < relabel_threshold]
    logger.info(
        "PRM scoring done: mean=%.3f, %d/%d below threshold %.2f",
        reward_mean, len(low), len(records), relabel_threshold,
    )

    # Pre-compute per-record routing and gather teacher-query jobs
    jobs: list[tuple[int, dict]] = []      # list of (idx, meta) for teacher calls
    meta_by_i: dict[int, dict] = {}        # per-record metadata
    for i, t in enumerate(records):
        pre_state = t.get("pre_state", {})
        img_path = t.get("_image_path_abs")
        if not img_path:
            img_path = _resolve_screenshot(run_dir, t.get("step", i), t)
        prompt = t.get("llm_prompt") or t.get("prompt") or ""
        reward = t.get("_reward", 0.0)
        teacher_hint = t.get("_teacher_hint", "")
        valid = bool(img_path and Path(img_path).exists() and prompt)
        meta_by_i[i] = {
            "t": t,
            "pre_state": pre_state,
            "img_path": img_path,
            "prompt": prompt,
            "reward": reward,
            "teacher_hint": teacher_hint,
            "valid": valid,
        }

    # Build the teacher-call queue up to budget, preserving trajectory order so
    # the budget cap picks the earliest low-reward steps rather than later ones.
    for i, m in meta_by_i.items():
        if not m["valid"]:
            continue
        if m["reward"] >= relabel_threshold:
            continue
        if max_teacher_calls is not None and len(jobs) >= max_teacher_calls:
            break
        jobs.append((i, m))

    # Parallelize teacher queries (8-way) — same pattern as the PRM judge.
    teacher_responses: dict[int, str | None] = {}
    if jobs:
        logger.info("dispatching %d teacher queries (parallel)", len(jobs))
        import concurrent.futures

        def _run(j):
            i, m = j
            return i, _teacher_query(
                image_path=m["img_path"],
                prompt_text=m["prompt"],
                teacher_hint=m["teacher_hint"],
                teacher_model=teacher_model,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            for i, resp in ex.map(_run, jobs):
                teacher_responses[i] = resp

    teacher_calls = len(jobs)
    relabeled = 0
    kept = 0
    skipped = 0
    with open(shard_path, "w") as fout:
        for i, m in meta_by_i.items():
            t = m["t"]
            if not m["valid"]:
                skipped += 1
                continue
            img_path, prompt, reward = m["img_path"], m["prompt"], m["reward"]
            pre_state = m["pre_state"]
            teacher_hint = m["teacher_hint"]

            if reward < relabel_threshold:
                if i not in teacher_responses:  # over budget
                    skipped += 1
                    continue
                teacher_resp = teacher_responses[i]
                if not teacher_resp:
                    skipped += 1
                    continue
                response_text = teacher_resp
                source = "teacher"
                relabeled += 1
            else:
                response_text = t.get("reasoning") or t.get("raw_response") or ""
                if not response_text:
                    skipped += 1
                    continue
                source = "agent"
                kept += 1

            record = {
                "schema_version": 2,
                "run_id": "dagger_prm",
                "step": t.get("step", i),
                "role": t.get("role", "orchestrator"),
                "interaction_type": f"dagger_{source}",
                "image_path": img_path,
                "prompt": prompt,
                "raw_response": response_text,
                "pre_state": pre_state,
                "post_state": t.get("post_state", {}),
                "_reward": reward,
                "_reward_detail": t.get("_reward_detail", {}),
                "_source": source,
            }
            fout.write(json.dumps(record) + "\n")

    stats = {
        "total": len(records),
        "relabeled": relabeled,
        "kept": kept,
        "skipped": skipped,
        "teacher_calls": teacher_calls,
        "reward_mean": reward_mean,
        "reward_low_count": len(low),
    }
    logger.info("DAgger shard: %s", stats)
    return stats


# ── Phase 3: SFT ────────────────────────────────────────────────────

def run_sft(
    shard_path: str,
    adapter_path: str,
    base_model_id: str,
    output_dir: str,
    data_root: str,
    epochs: int = 1,
    lora_rank: int = 256,
    num_gpus: int = 2,
) -> int:
    """Run sft_run.py on the DAgger shard. Resumes from current adapter LoRA."""
    # NB: sft_run.py loads the adapter via FastVisionModel.from_pretrained when
    # --model-id points at a LoRA checkpoint dir.
    cmd = [
        sys.executable, "-m", "accelerate.commands.launch",
        "--num_processes", str(num_gpus),
        "--mixed_precision", "bf16",
        "-m", "train.sft_run",
        "--shard", shard_path,
        "--data-root", data_root,
        "--output", output_dir,
        "--model-id", adapter_path,
        "--epochs", str(epochs),
        "--batch-size", "1",
        "--grad-accum", "4",
        "--lr", "2e-6",  # small-batch DAgger needs low LR for stability
        "--lora-rank", str(lora_rank),
        "--lora-alpha", str(lora_rank),
        "--max-length", "8192",
        "--save-steps", "100",
        "--no-4bit",
    ]
    logger.info("SFT cmd: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return result.returncode


# ── Main loop ──────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=Path, required=True,
                    help="Path to the current LoRA adapter to roll out with.")
    ap.add_argument("--base-model-id", type=str, default="unsloth/gemma-4-26B-A4B-it",
                    help="Base model id (for logging; unsloth resolves adapter config).")
    ap.add_argument("--game", type=str, default="red")
    ap.add_argument("--rom", type=str, default="PokemonRed-GBC/pokered.gbc")
    ap.add_argument("--output", type=Path, default=Path("train_runs/dagger_prm_red"))
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--rollout-steps", type=int, default=200)
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--num-gpus", type=int, default=2)
    ap.add_argument("--rollout-device", type=int, default=0)
    ap.add_argument("--judge-model", type=str,
                    default=os.environ.get("OPENCLAW_JUDGE_MODEL", "gemini-3-flash-preview"))
    ap.add_argument("--teacher-model", type=str,
                    default=os.environ.get("DAGGER_TEACHER_MODEL", "gemini-3-flash-preview"))
    ap.add_argument("--relabel-threshold", type=float, default=0.3,
                    help="PRM reward below this triggers teacher relabel.")
    ap.add_argument("--max-teacher-calls", type=int, default=None,
                    help="Cap teacher API calls per iteration (None = unlimited).")
    ap.add_argument("--sft-epochs", type=int, default=1)
    ap.add_argument("--lora-rank", type=int, default=256)
    ap.add_argument("--socks-login-host", type=str, default=None,
                    help="Hostname of the SSH login node to tunnel through "
                         "for Gemini API access (e.g. della-gpu.princeton.edu). "
                         "Required on della compute which has no external net.")
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    data_root = str(args.data_root or REPO_ROOT)
    current_adapter = str(args.adapter)

    # Start SOCKS tunnel up-front so judge + teacher can reach Gemini.
    # Safe to run unconditionally; on hosts with direct internet the
    # proxy just adds a hop.
    if args.socks_login_host:
        start_socks_tunnel(args.socks_login_host)
    elif os.environ.get("HTTPS_PROXY"):
        logger.info("HTTPS_PROXY already set to %s; skipping tunnel",
                    os.environ["HTTPS_PROXY"])
    else:
        logger.warning("no --socks-login-host set and no HTTPS_PROXY in env; "
                       "Gemini API calls will fail on compute nodes with no "
                       "external DNS")

    for iteration in range(args.iterations):
        logger.info("======== ITERATION %d/%d ========",
                    iteration + 1, args.iterations)
        iter_dir = args.output / f"iter_{iteration + 1}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        # Phase 1
        logger.info("PHASE 1: rollout %d steps via autoevolve harness with %s",
                    args.rollout_steps, current_adapter)
        rollout = run_rollout_harness(
            adapter_path=current_adapter,
            base_model_id=args.base_model_id,
            game=args.game,
            n_steps=args.rollout_steps,
            output_dir=iter_dir,
            device_index=args.rollout_device,
            rom=args.rom,
        )
        with open(iter_dir / "rollout_summary.json", "w") as f:
            json.dump(rollout, f, indent=2)
        if rollout["steps"] == 0 or rollout.get("trajectory") is None:
            logger.error("rollout produced no data, stopping")
            return 1

        # Phase 2
        shard_path = iter_dir / "dagger_shard.jsonl"
        logger.info("PHASE 2: PRM score + teacher relabel (threshold=%.2f)",
                    args.relabel_threshold)
        stats = score_and_relabel(
            trajectory_path=Path(rollout["trajectory"]),
            run_dir=Path(rollout["run_dir"]),
            shard_path=shard_path,
            judge_model=args.judge_model,
            teacher_model=args.teacher_model,
            relabel_threshold=args.relabel_threshold,
            max_teacher_calls=args.max_teacher_calls,
        )
        with open(iter_dir / "dagger_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        if stats["relabeled"] + stats["kept"] == 0:
            logger.error("empty DAgger shard, stopping")
            return 1

        # Phase 3
        train_dir = iter_dir / "sft_checkpoint"
        logger.info("PHASE 3: SFT on %d records (%d relabeled, %d kept)",
                    stats["relabeled"] + stats["kept"],
                    stats["relabeled"], stats["kept"])
        rc = run_sft(
            shard_path=str(shard_path),
            adapter_path=current_adapter,
            base_model_id=args.base_model_id,
            output_dir=str(train_dir),
            data_root=data_root,
            epochs=args.sft_epochs,
            lora_rank=args.lora_rank,
            num_gpus=args.num_gpus,
        )
        if rc != 0:
            logger.error("SFT failed (exit %d)", rc)
            return rc

        # Next iteration uses the final checkpoint (or newest if multiple)
        ckpt_dirs = sorted(train_dir.glob("checkpoint-*"))
        if ckpt_dirs:
            current_adapter = str(ckpt_dirs[-1])
            logger.info("next iteration adapter: %s", current_adapter)
        else:
            # sft_run.py saves final adapter to output_dir directly
            if (train_dir / "adapter_config.json").exists():
                current_adapter = str(train_dir)
                logger.info("next iteration adapter (final): %s", current_adapter)
            else:
                logger.warning("no checkpoint produced; reusing %s", current_adapter)

        with open(args.output / "iteration_log.jsonl", "a") as f:
            f.write(json.dumps({
                "iteration": iteration + 1,
                "rollout_steps": rollout["steps"],
                "evolutions": rollout.get("evolutions", 0),
                "reward_mean": stats["reward_mean"],
                "relabeled": stats["relabeled"],
                "kept": stats["kept"],
                "teacher_calls": stats["teacher_calls"],
                "adapter": current_adapter,
            }) + "\n")

    logger.info("===== DAgger+PRM COMPLETE =====")
    return 0


if __name__ == "__main__":
    sys.exit(main())
