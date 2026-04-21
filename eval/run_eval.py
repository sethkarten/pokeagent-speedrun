#!/usr/bin/env python3
"""Pokemon Gemma4 agent eval pipeline.

Example::

    uv run eval/run_eval.py \\
        --models gemma4:26b,gemma4-emerald:26b \\
        --dataset emerald_v3 \\
        --n-samples 16 \\
        --prompt-mode real \\
        --judge gemini-2.5-flash \\
        --output data/eval_real_mine.json \\
        --md data/eval_real_mine.md

Outputs:
- ``--output`` JSON: raw per-sample scores per model (schema matches
  ``data/eval_real_prompts_comparison.json``).
- ``--md`` Markdown: aggregate tables + sample comparison excerpts.

Partial results are saved after each model finishes so crashes don't lose work.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import requests

# Make ``train.reward_functions`` importable when invoked as a script.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.data_loader import Sample, load_samples  # noqa: E402
from eval.scorers import (  # noqa: E402
    mean,
    score_actionable,
    score_degenerate,
    score_grounding,
    score_tool_format,
)


logging.basicConfig(
    level=os.environ.get("EVAL_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("eval")


OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "120"))


# ── Ollama inference ─────────────────────────────────────────────────


def _ollama_list_models() -> set[str]:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        return {m["name"] for m in r.json().get("models", [])}
    except Exception as e:
        logger.warning("could not list Ollama models (%s); skipping filter", e)
        return set()


def ollama_generate(
    model: str,
    prompt: str,
    image_b64: Optional[str],
    *,
    num_predict: int = 512,
    temperature: float = 0.0,
) -> tuple[str, float, float]:
    """Call Ollama /api/generate. Returns (response_text, elapsed_seconds, tok_per_s).

    Returns ("ERROR: ...", elapsed, 0) on failure rather than raising so the
    caller can still record the failed sample.
    """
    payload: dict = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }
    if image_b64:
        payload["images"] = [image_b64]

    start = time.time()
    try:
        r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=OLLAMA_TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        elapsed = time.time() - start
        return f"ERROR: {e}", elapsed, 0.0

    elapsed = time.time() - start
    text = data.get("response", "") or ""
    # Ollama reports eval_count (output tokens) and eval_duration (ns).
    eval_count = data.get("eval_count") or 0
    eval_duration_ns = data.get("eval_duration") or 0
    if eval_count and eval_duration_ns:
        tok_s = eval_count / (eval_duration_ns / 1e9)
    elif eval_count and elapsed > 0:
        tok_s = eval_count / elapsed
    else:
        tok_s = 0.0
    return text, elapsed, tok_s


# ── per-sample scoring ───────────────────────────────────────────────


def score_sample(
    *,
    response: str,
    sample: Sample,
    prompt_text: str,
    judge_scores,
) -> dict:
    """Compose heuristic + judge scores into the per-sample dict."""
    scores = {
        "tool_format": score_tool_format(response),
        "grounding": score_grounding(response, sample.pre_state),
        "degenerate": score_degenerate(response, prompt_text),
        "actionable": score_actionable(response),
    }
    if judge_scores is not None:
        if judge_scores.action_relevance is not None:
            scores["action_relevance"] = judge_scores.action_relevance
        if judge_scores.reasoning_similarity is not None:
            scores["reasoning_similarity"] = judge_scores.reasoning_similarity
        if judge_scores.hallucination is not None:
            scores["hallucination"] = judge_scores.hallucination
    return scores


# ── aggregation ──────────────────────────────────────────────────────


def _metric_mean(rows: list[dict], metric: str) -> Optional[float]:
    values = [r["scores"][metric] for r in rows if metric in r["scores"]]
    if not values:
        return None
    return sum(values) / len(values)


def aggregate(results_by_model: dict[str, list[dict]]) -> dict:
    """Compute overall and per-state_type means."""
    agg: dict = {"overall": {}, "by_state_type": {}}

    metrics = [
        "tool_format",
        "grounding",
        "action_relevance",
        "reasoning_similarity",
        "hallucination",
        "degenerate",
        "actionable",
        "tok_s",
    ]

    for model, rows in results_by_model.items():
        agg["overall"][model] = {m: _metric_mean(rows, m) for m in metrics}

    for state_type in ("overworld", "battle"):
        agg["by_state_type"][state_type] = {}
        for model, rows in results_by_model.items():
            filtered = [r for r in rows if r["state_type"] == state_type]
            agg["by_state_type"][state_type][model] = {
                m: _metric_mean(filtered, m) for m in metrics
            }
    return agg


# ── Markdown output ──────────────────────────────────────────────────


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def _write_table(lines: list[str], title: str, models: list[str], per_model_metrics: dict, metrics: list[str]) -> None:
    lines.append(f"### {title}\n" if title else "")
    header = ["Metric"] + models
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for m in metrics:
        row = [m]
        for model in models:
            v = per_model_metrics[model].get(m)
            if m == "tok_s" and v is not None:
                row.append(f"{v:.2f} t/s")
            else:
                row.append(_fmt(v))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")


def write_markdown(
    md_path: Path,
    *,
    dataset: str,
    prompt_mode: str,
    models: list[str],
    samples: list[Sample],
    results_by_model: dict[str, list[dict]],
    aggregates: dict,
    judge_name: Optional[str],
) -> None:
    lines: list[str] = []
    lines.append("# Pokemon Gemma4 agent eval\n")
    lines.append(f"- **Dataset**: {dataset} ({len(samples)} samples)")
    lines.append(f"- **Prompt mode**: {prompt_mode}")
    lines.append(f"- **Models**: {', '.join(models)}")
    lines.append(f"- **Judge**: {judge_name or 'disabled'}")
    lines.append("")

    # Decide which metrics to show based on what's actually populated.
    displayed_metrics = [
        "tool_format",
        "actionable",
        "grounding",
        "action_relevance",
        "reasoning_similarity",
        "hallucination",
        "degenerate",
        "tok_s",
    ]
    # Drop metrics that are None for every model overall (e.g. when --no-judge)
    overall = aggregates["overall"]
    displayed_metrics = [
        m for m in displayed_metrics
        if any(overall.get(mod, {}).get(m) is not None for mod in models)
    ]

    lines.append("## Overall Scores\n")
    _write_table(lines, "", models, overall, displayed_metrics)

    for state_type in ("overworld", "battle"):
        per = aggregates["by_state_type"][state_type]
        if all(all(v is None for v in d.values()) for d in per.values()):
            continue
        lines.append(f"## {state_type.capitalize()}\n")
        _write_table(lines, "", models, per, displayed_metrics)

    lines.append("## Sample Responses\n")
    # show up to 6 samples across the spread
    sample_indices = list(range(0, len(samples), max(1, len(samples) // 6)))[:6]
    for idx in sample_indices:
        sample = samples[idx]
        lines.append(f"### Example {idx + 1}: {sample.state_type} @ {sample.location}")
        teacher_preview = (sample.raw_response or "").replace("\n", " ")[:220]
        lines.append(f"- **Teacher**: `{teacher_preview}`")
        for model in models:
            rows = results_by_model.get(model, [])
            match = next((r for r in rows if r["idx"] == idx), None)
            if match is None:
                lines.append(f"- **{model}**: _(not run)_")
                continue
            preview = match["scores"].get("_response_full", match["scores"].get("response_preview", ""))
            preview = match.get("response_preview", "") or preview
            preview = (preview or "").replace("\n", " ")[:220]
            lines.append(f"- **{model}**: `{preview}`")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")


# ── main ─────────────────────────────────────────────────────────────


def _save_partial(output_path: Path, results_by_model: dict[str, list[dict]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results_by_model, indent=2), encoding="utf-8")


def run(args) -> int:
    dataset = args.dataset
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        logger.error("--models is required and must be a comma-separated list")
        return 2

    # Filter models not in Ollama
    available = _ollama_list_models()
    if available:
        missing = [m for m in models if m not in available]
        if missing:
            logger.warning("Ollama does not have: %s — will attempt anyway (Ollama may auto-pull)", missing)

    samples = load_samples(
        dataset,
        args.n_samples,
        seed=args.seed,
        require_image=True,
    )
    if not samples:
        logger.error("no samples loaded (dataset %s; missing screenshots?)", dataset)
        return 2
    logger.info("loaded %d samples (target %d)", len(samples), args.n_samples)

    judge = None
    if not args.no_judge:
        try:
            from eval.judge import GeminiJudge

            judge = GeminiJudge(args.judge)
            logger.info("judge enabled: %s", args.judge)
        except Exception as e:
            logger.error("failed to init Gemini judge (%s); continuing with --no-judge", e)
            judge = None

    output_path = Path(args.output).resolve()
    md_path = Path(args.md).resolve() if args.md else None

    results_by_model: dict[str, list[dict]] = defaultdict(list)

    for model in models:
        logger.info("===== %s =====", model)
        for sample in samples:
            prompt_text = sample.prompt if args.prompt_mode == "real" else sample.simplified_prompt
            t0 = time.time()
            response, elapsed, tok_s = ollama_generate(
                model,
                prompt_text,
                sample.image_b64,
                num_predict=args.num_predict,
                temperature=args.temperature,
            )
            # Judge
            judge_scores = None
            if judge is not None and not response.startswith("ERROR:"):
                judge_scores = judge.score(
                    model_response=response,
                    teacher_response=sample.raw_response,
                    pre_state=sample.pre_state,
                    image_b64=sample.image_b64,
                )
            # Score
            scores = score_sample(
                response=response,
                sample=sample,
                prompt_text=prompt_text,
                judge_scores=judge_scores,
            )
            scores["tok_s"] = tok_s
            scores["elapsed"] = elapsed

            results_by_model[model].append({
                "idx": sample.idx,
                "state_type": sample.state_type,
                "location": sample.location,
                "source": sample.source,
                "scores": scores,
                "response_preview": (response or "")[:240],
            })
            logger.info(
                "  [%s] idx=%d %s @ %s  tool=%.2f grnd=%.2f act=%.2f %.1ft/s  %.1fs",
                model,
                sample.idx,
                sample.state_type,
                sample.location[:24],
                scores.get("tool_format", 0.0),
                scores.get("grounding", 0.0),
                scores.get("actionable", 0.0),
                tok_s,
                elapsed,
            )
        # Save partial after each model
        _save_partial(output_path, dict(results_by_model))
        logger.info("saved partial results to %s", output_path)

    aggregates = aggregate(dict(results_by_model))

    # Save final JSON (raw per-sample) and aggregates alongside it
    final = dict(results_by_model)
    final["_aggregate"] = aggregates
    final["_meta"] = {
        "dataset": dataset,
        "n_samples": len(samples),
        "prompt_mode": args.prompt_mode,
        "models": models,
        "judge": None if args.no_judge else args.judge,
        "seed": args.seed,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(final, indent=2), encoding="utf-8")
    logger.info("wrote %s", output_path)

    if md_path:
        write_markdown(
            md_path,
            dataset=dataset,
            prompt_mode=args.prompt_mode,
            models=models,
            samples=samples,
            results_by_model=dict(results_by_model),
            aggregates=aggregates,
            judge_name=None if args.no_judge else args.judge,
        )
        logger.info("wrote %s", md_path)

    # Print summary to stdout
    print("\n=== AGGREGATE SCORES ===")
    for model in models:
        print(f"\n{model}:")
        for metric, v in aggregates["overall"][model].items():
            if v is None:
                continue
            if metric == "tok_s":
                print(f"  {metric}: {v:.2f} t/s")
            else:
                print(f"  {metric}: {v:.3f}")
    return 0


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pokemon Gemma4 agent eval")
    p.add_argument("--models", required=True, help="comma-separated Ollama model names")
    p.add_argument("--dataset", default="emerald_v3", help="sft_dataset subdir (emerald_v3 | red_v1)")
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--prompt-mode", choices=("simplified", "real"), default="real")
    p.add_argument("--judge", default="gemini-2.5-flash")
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--output", required=True, help="path to JSON output")
    p.add_argument("--md", default=None, help="optional Markdown output path")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-predict", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    return p


def main() -> int:
    args = build_argparser().parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
