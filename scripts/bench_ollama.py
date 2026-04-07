#!/usr/bin/env python
"""Benchmark Gemma 4 (Ollama) on the harness orchestrator workload.

Uses a real captured prompt + screenshot fixture (from a recent browser
agent run) so the numbers are directly comparable to the Gemini API
durations recorded in llm_logs/. For each model the script runs one
warmup call (absorbs cold load) followed by N measured calls and reports
both per-call and median statistics.

Metrics (taken from Ollama's /api/generate response, which are far more
accurate than wall-clock):

  - load_duration       — model load time (zero on warm calls)
  - prompt_eval_count   — input tokens processed (prefill)
  - prompt_eval_dur     — prefill wall time
  - prefill_tok_s       — prompt_eval_count / prompt_eval_dur
  - eval_count          — output tokens generated (decode)
  - eval_duration       — decode wall time
  - decode_tok_s        — eval_count / eval_duration
  - ttft                — time to first non-empty stream chunk (separately measured)
  - total_wall          — full request wall time

Usage:
  .venv/bin/python scripts/bench_ollama.py \
      --models gemma4:26b gemma4:31b \
      --runs 3 \
      --num-ctx 32768 \
      --num-predict 5000

Defaults match the harness expectations. Set OLLAMA_HOST to point at a
non-default daemon.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
if not DEFAULT_HOST.startswith("http"):
    DEFAULT_HOST = f"http://{DEFAULT_HOST}"

REPO = Path(__file__).resolve().parent.parent


def gpu_vram_used_mib(gpu_index: int = 1) -> int:
    """Return MiB currently used on a specific GPU."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        return int(out.strip())
    except Exception:
        return -1


def call_generate(
    host: str,
    model: str,
    prompt: str,
    image_b64: str,
    num_ctx: int,
    num_predict: int,
    timeout: float = 600.0,
) -> dict[str, Any]:
    """Call /api/generate streaming and return timing metrics.

    Returns a dict with: ttft_s, total_wall_s, and the final-message
    timing fields from Ollama (load_duration, prompt_eval_count,
    prompt_eval_duration, eval_count, eval_duration, response_chars).
    """
    body = {
        "model": model,
        "prompt": prompt,
        "images": [image_b64],
        "stream": True,
        "options": {
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "temperature": 0.7,
        },
    }
    started = time.perf_counter()
    ttft_s: float | None = None
    response_chars = 0
    final: dict[str, Any] = {}

    with requests.post(
        f"{host}/api/generate",
        json=body,
        stream=True,
        timeout=timeout,
    ) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            try:
                chunk = json.loads(raw)
            except json.JSONDecodeError:
                continue
            text = chunk.get("response", "")
            if text and ttft_s is None:
                ttft_s = time.perf_counter() - started
            response_chars += len(text)
            if chunk.get("done"):
                final = chunk
                break

    total_wall = time.perf_counter() - started

    def _ns_to_s(ns: int | None) -> float:
        return (ns or 0) / 1e9

    return {
        "ttft_s": ttft_s if ttft_s is not None else float("nan"),
        "total_wall_s": total_wall,
        "load_s": _ns_to_s(final.get("load_duration")),
        "prompt_eval_count": final.get("prompt_eval_count", 0),
        "prompt_eval_s": _ns_to_s(final.get("prompt_eval_duration")),
        "eval_count": final.get("eval_count", 0),
        "eval_s": _ns_to_s(final.get("eval_duration")),
        "response_chars": response_chars,
    }


def derived_rates(rec: dict[str, Any]) -> dict[str, float]:
    pe_c = rec.get("prompt_eval_count") or 0
    pe_s = rec.get("prompt_eval_s") or 0
    ev_c = rec.get("eval_count") or 0
    ev_s = rec.get("eval_s") or 0
    return {
        "prefill_tok_s": pe_c / pe_s if pe_s > 0 else float("nan"),
        "decode_tok_s": ev_c / ev_s if ev_s > 0 else float("nan"),
    }


def fmt_row(label: str, recs: list[dict[str, Any]]) -> str:
    """Format a row showing median + min/max for the given metric across runs."""
    if not recs:
        return f"  {label:22}: (no data)"
    return f"  {label:22}: " + " ".join(f"{r:>10}" for r in recs)


def report_model(model: str, warmup: dict[str, Any], runs: list[dict[str, Any]],
                 vram_after_load: int) -> None:
    print(f"\n=== {model} ===")
    print(f"  vram_after_load_mib  : {vram_after_load}")
    print(f"  warmup load          : {warmup['load_s']:.2f}s")
    print(f"  warmup ttft          : {warmup['ttft_s']:.2f}s")
    print(f"  warmup total_wall    : {warmup['total_wall_s']:.2f}s")

    headers = [f"run{i+1}" for i in range(len(runs))]
    print(f"\n  {'metric':22}: " + " ".join(f"{h:>10}" for h in headers))
    print(f"  {'-'*22}: " + " ".join("-" * 10 for _ in runs))
    print(fmt_row("ttft_s",
                  [f"{r['ttft_s']:.2f}" for r in runs]))
    print(fmt_row("total_wall_s",
                  [f"{r['total_wall_s']:.2f}" for r in runs]))
    print(fmt_row("prompt_eval_count",
                  [f"{r['prompt_eval_count']}" for r in runs]))
    print(fmt_row("prompt_eval_s",
                  [f"{r['prompt_eval_s']:.2f}" for r in runs]))
    print(fmt_row("eval_count",
                  [f"{r['eval_count']}" for r in runs]))
    print(fmt_row("eval_s",
                  [f"{r['eval_s']:.2f}" for r in runs]))

    rates = [derived_rates(r) for r in runs]
    print(fmt_row("prefill_tok_s",
                  [f"{x['prefill_tok_s']:.0f}" for x in rates]))
    print(fmt_row("decode_tok_s",
                  [f"{x['decode_tok_s']:.1f}" for x in rates]))

    # Median summary
    def med(seq: list[float]) -> float:
        seq = [x for x in seq if x == x]  # drop NaN
        return statistics.median(seq) if seq else float("nan")

    print(f"\n  --- median across {len(runs)} runs ---")
    print(f"  ttft_s          : {med([r['ttft_s'] for r in runs]):.2f}")
    print(f"  total_wall_s    : {med([r['total_wall_s'] for r in runs]):.2f}")
    print(f"  prefill_tok_s   : {med([x['prefill_tok_s'] for x in rates]):.0f}")
    print(f"  decode_tok_s    : {med([x['decode_tok_s'] for x in rates]):.1f}")
    print(f"  output_tokens   : {int(med([float(r['eval_count']) for r in runs]))}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=DEFAULT_HOST,
                        help=f"Ollama host (default: {DEFAULT_HOST})")
    parser.add_argument("--models", nargs="+",
                        default=["gemma4:26b", "gemma4:31b"])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--num-ctx", type=int, default=32768)
    parser.add_argument("--num-predict", type=int, default=5000)
    parser.add_argument("--prompt-file", type=Path,
                        default=Path("/tmp/bench_prompt.txt"))
    parser.add_argument("--image-file", type=Path,
                        default=Path("/tmp/bench_screenshot.png"))
    parser.add_argument("--gpu-id", type=int, default=1,
                        help="GPU index for VRAM measurements (logical)")
    parser.add_argument("--results-out", type=Path,
                        default=REPO / "scripts" / "bench_ollama_results.json")
    args = parser.parse_args()

    if not args.prompt_file.exists():
        print(f"ERROR: prompt fixture missing: {args.prompt_file}",
              file=sys.stderr)
        return 1
    if not args.image_file.exists():
        print(f"ERROR: image fixture missing: {args.image_file}",
              file=sys.stderr)
        return 1

    prompt = args.prompt_file.read_text()
    image_b64 = base64.b64encode(args.image_file.read_bytes()).decode("ascii")

    print(f"host          : {args.host}")
    print(f"prompt        : {args.prompt_file} ({len(prompt):,} chars)")
    print(f"image         : {args.image_file} "
          f"({args.image_file.stat().st_size:,} bytes)")
    print(f"num_ctx       : {args.num_ctx}")
    print(f"num_predict   : {args.num_predict}")
    print(f"runs/model    : {args.runs} (+1 warmup)")

    # Verify Ollama is reachable
    try:
        v = requests.get(f"{args.host}/api/version", timeout=3).json()
        print(f"ollama        : {v.get('version', '?')}")
    except Exception as e:
        print(f"ERROR: cannot reach {args.host}: {e}", file=sys.stderr)
        return 1

    # GPU baseline
    vram_baseline = gpu_vram_used_mib(args.gpu_id)
    print(f"vram_baseline_mib: {vram_baseline} (gpu {args.gpu_id})")

    results: dict[str, Any] = {
        "host": args.host,
        "num_ctx": args.num_ctx,
        "num_predict": args.num_predict,
        "prompt_chars": len(prompt),
        "image_bytes": args.image_file.stat().st_size,
        "vram_baseline_mib": vram_baseline,
        "runs_per_model": args.runs,
        "models": {},
    }

    for model in args.models:
        print(f"\n{'='*60}")
        print(f" {model}")
        print(f"{'='*60}")

        try:
            # Warmup — absorbs cold load + first JIT
            print("[warmup]", end=" ", flush=True)
            warmup = call_generate(
                args.host, model, prompt, image_b64,
                args.num_ctx, args.num_predict,
            )
            print(
                f"load={warmup['load_s']:.1f}s "
                f"ttft={warmup['ttft_s']:.2f}s "
                f"total={warmup['total_wall_s']:.2f}s "
                f"out={warmup['eval_count']}tok"
            )

            vram_after_load = gpu_vram_used_mib(args.gpu_id)
            print(f"[vram] after-load={vram_after_load} mib "
                  f"(delta={vram_after_load - vram_baseline} mib)")

            measured: list[dict[str, Any]] = []
            for i in range(args.runs):
                print(f"[run {i+1}/{args.runs}]", end=" ", flush=True)
                rec = call_generate(
                    args.host, model, prompt, image_b64,
                    args.num_ctx, args.num_predict,
                )
                d = derived_rates(rec)
                print(
                    f"ttft={rec['ttft_s']:.2f}s "
                    f"prefill={d['prefill_tok_s']:.0f}t/s "
                    f"decode={d['decode_tok_s']:.1f}t/s "
                    f"total={rec['total_wall_s']:.2f}s "
                    f"out={rec['eval_count']}tok"
                )
                measured.append(rec)

            report_model(model, warmup, measured, vram_after_load)
            results["models"][model] = {
                "warmup": warmup,
                "runs": measured,
                "vram_after_load_mib": vram_after_load,
            }

        except Exception as e:
            print(f"\nERROR running {model}: {e}", file=sys.stderr)
            results["models"][model] = {"error": str(e)}

    # Compare to Gemini reference (from llm_logs analysis)
    GEMINI_MEDIAN_INPUT_TOK = 11614
    GEMINI_MEDIAN_OUTPUT_TOK = 1412
    GEMINI_MEDIAN_DURATION_S = 12.99
    print(f"\n{'='*60}")
    print(" Gemini reference (median orchestrator call from llm_logs)")
    print(f"{'='*60}")
    print(f"  input_tokens   : {GEMINI_MEDIAN_INPUT_TOK:,}")
    print(f"  output_tokens  : {GEMINI_MEDIAN_OUTPUT_TOK:,}")
    print(f"  total_duration : {GEMINI_MEDIAN_DURATION_S:.2f}s")

    print(f"\n{'='*60}")
    print(" Side-by-side (median across runs)")
    print(f"{'='*60}")
    header = f"  {'model':24}{'total_wall':>14}{'decode_t/s':>14}{'vs gemini':>14}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    print(f"  {'gemini-2.5-flash':24}{f'{GEMINI_MEDIAN_DURATION_S:.2f}s':>14}"
          f"{'?':>14}{'1.0x':>14}")
    for model, data in results["models"].items():
        if "error" in data:
            print(f"  {model:24}{'ERROR':>14}{'':>14}{'':>14}")
            continue
        runs = data["runs"]
        med_total = statistics.median([r["total_wall_s"] for r in runs])
        med_decode = statistics.median(
            [derived_rates(r)["decode_tok_s"] for r in runs]
        )
        ratio = med_total / GEMINI_MEDIAN_DURATION_S
        print(f"  {model:24}{f'{med_total:.2f}s':>14}"
              f"{f'{med_decode:.1f}':>14}{f'{ratio:.2f}x':>14}")

    args.results_out.parent.mkdir(parents=True, exist_ok=True)
    args.results_out.write_text(json.dumps(results, indent=2))
    print(f"\nresults saved: {args.results_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
