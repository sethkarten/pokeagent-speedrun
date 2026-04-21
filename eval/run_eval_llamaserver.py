"""Eval for vision-split-GGUF models via llama-server.

For each model config, launches llama-server on a local port, runs all samples,
then shuts down. Results are merged into the same JSON format as run_eval.py.

Usage:
    uv run eval/run_eval_llamaserver.py \\
        --models-config llama_server_models.json \\
        --dataset emerald_v3 --n-samples 20 \\
        --output data/eval_emerald_ls.json --md data/eval_emerald_ls.md

models-config JSON:
    {"gemma4-emerald:31b": {
        "gguf": "/path/to/main.q4.gguf",
        "mmproj": "/path/to/mmproj.gguf"}, ...}
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import signal
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.data_loader import Sample, load_samples  # noqa: E402
from eval.run_eval import aggregate, score_sample, write_markdown  # noqa: E402

logging.basicConfig(
    level=os.environ.get("EVAL_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("eval_ls")

LLAMA_SERVER = os.environ.get(
    "LLAMA_SERVER_BIN",
    "/media/milkkarten/data/llama.cpp/build2/bin/llama-server",
)


def _wait_ready(port: int, timeout: float = 240.0) -> bool:
    t0 = time.time()
    url = f"http://127.0.0.1:{port}/health"
    while time.time() - t0 < timeout:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200 and r.json().get("status") == "ok":
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def _launch(gguf: str, mmproj: str, port: int, gpu: int, ctx: int, log_path: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [
        LLAMA_SERVER,
        "-m", gguf,
        "--mmproj", mmproj,
        "--host", "127.0.0.1",
        "--port", str(port),
        "-ngl", "99",
        "-c", str(ctx),
        "--no-warmup",
    ]
    logger.info("launching: %s", " ".join(cmd))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = log_path.open("w")
    return subprocess.Popen(cmd, env=env, stdout=f, stderr=f)


def _stop(proc: subprocess.Popen) -> None:
    if proc is None:
        return
    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=30)
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception:
            pass


def _generate(port: int, prompt: str, image_b64: Optional[str],
              num_predict: int, temperature: float, timeout: float) -> tuple[str, float, float]:
    messages = [{"role": "user", "content": []}]
    if image_b64:
        messages[0]["content"].append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
        })
    messages[0]["content"].append({"type": "text", "text": prompt})
    payload = {
        "messages": messages,
        "max_tokens": num_predict,
        "temperature": temperature,
    }
    t0 = time.time()
    try:
        r = requests.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json=payload, timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return f"ERROR: {e}", time.time() - t0, 0.0
    elapsed = time.time() - t0
    choice = (data.get("choices") or [{}])[0]
    text = choice.get("message", {}).get("content", "") or ""
    timings = data.get("timings") or {}
    usage = data.get("usage") or {}
    comp = usage.get("completion_tokens") or len(text.split())
    pt_ms = timings.get("predicted_per_token_ms") or 0
    if pt_ms > 0:
        tok_s = 1000.0 / pt_ms
    elif elapsed > 0 and comp > 0:
        tok_s = comp / elapsed
    else:
        tok_s = 0.0
    return text, elapsed, tok_s


def run(args) -> int:
    models_cfg = json.loads(Path(args.models_config).read_text())
    models = list(models_cfg.keys())

    samples = load_samples(
        args.dataset, args.n_samples, seed=args.seed, require_image=True,
    )
    if not samples:
        logger.error("no samples loaded")
        return 2
    logger.info("loaded %d samples", len(samples))

    judge = None
    if not args.no_judge:
        try:
            from eval.judge import GeminiJudge
            judge = GeminiJudge(args.judge)
            logger.info("judge enabled: %s", args.judge)
        except Exception as e:
            logger.error("judge init failed (%s); --no-judge", e)

    output_path = Path(args.output).resolve()
    results_by_model: dict[str, list[dict]] = defaultdict(list)

    for display_name in models:
        cfg = models_cfg[display_name]
        gguf = cfg["gguf"]
        mmproj = cfg["mmproj"]
        logger.info("===== %s =====", display_name)
        log_path = Path(f"merge_logs/llama_server/{display_name.replace(':','_').replace('/','_')}.log")
        proc = _launch(gguf, mmproj, args.port, args.gpu, args.ctx, log_path)
        try:
            if not _wait_ready(args.port, timeout=args.load_timeout):
                logger.error("llama-server failed to become ready for %s; see %s", display_name, log_path)
                _stop(proc)
                continue

            for sample in samples:
                prompt_text = sample.prompt if args.prompt_mode == "real" else sample.simplified_prompt
                response, elapsed, tok_s = _generate(
                    args.port, prompt_text, sample.image_b64,
                    args.num_predict, args.temperature, args.req_timeout,
                )

                judge_scores = None
                if judge is not None and not response.startswith("ERROR:"):
                    try:
                        judge_scores = judge.score(
                            model_response=response,
                            teacher_response=sample.raw_response,
                            pre_state=sample.pre_state,
                            image_b64=sample.image_b64,
                        )
                    except Exception as e:
                        logger.warning("judge failed idx=%d: %s", sample.idx, e)

                scores = score_sample(
                    response=response,
                    sample=sample,
                    prompt_text=prompt_text,
                    judge_scores=judge_scores,
                )
                scores["tok_s"] = tok_s
                scores["elapsed"] = elapsed

                results_by_model[display_name].append({
                    "idx": sample.idx,
                    "state_type": sample.state_type,
                    "location": sample.location,
                    "source": sample.source,
                    "scores": scores,
                    "response_preview": (response or "")[:240],
                })
                logger.info(
                    "  [%s] idx=%d %s @ %s  tool=%.2f grnd=%.2f act=%.2f %.1ft/s  %.1fs",
                    display_name, sample.idx, sample.state_type,
                    sample.location[:24],
                    scores.get("tool_format", 0.0),
                    scores.get("grounding", 0.0),
                    scores.get("actionable", 0.0),
                    tok_s, elapsed,
                )
        finally:
            _stop(proc)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(dict(results_by_model), indent=2))
        logger.info("saved partial to %s", output_path)

    aggregates = aggregate(dict(results_by_model))
    final = dict(results_by_model)
    final["_aggregate"] = aggregates
    final["_meta"] = {
        "dataset": args.dataset,
        "n_samples": len(samples),
        "prompt_mode": args.prompt_mode,
        "backend": "llama-server",
        "judge": None if args.no_judge else args.judge,
    }
    output_path.write_text(json.dumps(final, indent=2))
    logger.info("saved final JSON to %s", output_path)

    if args.md:
        md_path = Path(args.md).resolve()
        write_markdown(
            md_path,
            dataset=args.dataset,
            prompt_mode=args.prompt_mode,
            models=models,
            samples=samples,
            results_by_model=dict(results_by_model),
            aggregates=aggregates,
            judge_name=None if args.no_judge else args.judge,
        )
        logger.info("saved markdown to %s", md_path)

    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-config", required=True)
    ap.add_argument("--dataset", default="emerald_v3")
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--prompt-mode", choices=["simplified", "real"], default="real")
    ap.add_argument("--judge", default="gemini-2.5-flash")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--output", required=True)
    ap.add_argument("--md", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-predict", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--port", type=int, default=11435)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--load-timeout", type=float, default=300.0)
    ap.add_argument("--req-timeout", type=float, default=300.0)
    args = ap.parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
