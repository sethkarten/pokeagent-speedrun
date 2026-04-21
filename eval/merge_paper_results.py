"""Merge Ollama + llama-server eval JSONs into one paper-ready per-game output.

Produces:
- data/eval_<game>_paper.json — union of all models with aggregate
- data/eval_<game>_paper.md — markdown tables
- {out_dir}/eval_<game>_paper.json + .tex — for paper agent

Usage:
    uv run eval/merge_paper_results.py \\
        --game emerald \\
        --inputs data/eval_emerald_paper_v1.json,data/eval_emerald_ls.json \\
        --out-dir /media/milkkarten/data/pokeagent-speedrun/autoevolve-latex/analysis/data
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.run_eval import aggregate, write_markdown  # noqa: E402
from eval.data_loader import load_samples  # noqa: E402


# Preferred display order for markdown/tex tables
DEFAULT_ORDER = [
    "gemma4:26b", "gemma4:31b", "gemma4:e4b", "gemma4:e2b",
    "gemma4-emerald:26b", "gemma4-emerald:31b", "gemma4-emerald:e4b", "gemma4-emerald:e2b",
    "gemma4-emerald-grpo:26b",
    "gemma4-red:26b", "gemma4-red:31b", "gemma4-red:e4b", "gemma4-red:e2b",
    "gemma4-red-grpo:26b",
]


def _looks_errored(rows: list[dict]) -> bool:
    """Rows are 'errored' when every response starts with 'ERROR:'."""
    if not rows:
        return True
    return all((r.get("response_preview") or "").startswith("ERROR:") for r in rows)


def merge_inputs(paths: list[Path]) -> dict:
    merged: dict[str, list[dict]] = {}
    for p in paths:
        data = json.loads(p.read_text())
        for model, rows in data.items():
            if model.startswith("_"):
                continue
            if not isinstance(rows, list):
                continue
            # Prefer later-arriving, non-errored data if conflict
            if model in merged:
                if _looks_errored(merged[model]) and not _looks_errored(rows):
                    merged[model] = rows
                elif _looks_errored(rows):
                    continue  # keep existing
                else:
                    continue  # first-in wins for valid data
            else:
                merged[model] = rows
    return merged


def order_models(models: list[str]) -> list[str]:
    seen = set()
    out = []
    for m in DEFAULT_ORDER:
        if m in models and m not in seen:
            out.append(m); seen.add(m)
    for m in models:
        if m not in seen:
            out.append(m); seen.add(m)
    return out


def write_tex_table(tex_path: Path, models: list[str], aggregates: dict, metrics: list[str]) -> None:
    """Produce a LaTeX booktabs table with models as columns."""
    def esc(s: str) -> str:
        return s.replace("_", r"\_").replace("&", r"\&").replace("#", r"\#")

    def fmt(v):
        if v is None:
            return "--"
        if isinstance(v, float):
            return f"{v:.2f}"
        return str(v)

    ncols = 1 + len(models)
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering\small")
    lines.append(r"\begin{tabular}{l" + "c" * len(models) + "}")
    lines.append(r"\toprule")
    header = ["Metric"] + [esc(m) for m in models]
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")
    overall = aggregates["overall"]
    for metric in metrics:
        row = [esc(metric)]
        for m in models:
            v = overall.get(m, {}).get(metric)
            if metric == "tok_s" and v is not None:
                row.append(f"{v:.1f} t/s")
            else:
                row.append(fmt(v))
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Overall scores across models.}")
    lines.append(r"\end{table}")
    tex_path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True, choices=["emerald", "red"])
    ap.add_argument("--inputs", required=True, help="Comma-separated list of JSON input paths")
    ap.add_argument("--dataset", default=None, help="Dataset name for sample loading (for markdown; default derived from game)")
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--out-local", default=None, help="Optional local output basename (default data/eval_<game>_paper)")
    ap.add_argument("--out-dir", default=None, help="Copy outputs to this dir (e.g., autoevolve-latex/analysis/data)")
    args = ap.parse_args()

    inputs = [Path(p.strip()) for p in args.inputs.split(",") if p.strip()]
    merged_rows = merge_inputs(inputs)

    # Drop models that are fully errored
    merged_rows = {m: rows for m, rows in merged_rows.items() if not _looks_errored(rows)}

    models = order_models(list(merged_rows.keys()))

    aggregates = aggregate(merged_rows)

    # Load samples for markdown (dataset name, snippet rendering)
    dataset = args.dataset or ("emerald_v3" if args.game == "emerald" else "red_v1")
    samples = load_samples(dataset, args.n_samples, require_image=True)

    out_local_base = args.out_local or f"data/eval_{args.game}_paper"
    json_path = Path(out_local_base + ".json").resolve()
    md_path = Path(out_local_base + ".md").resolve()
    tex_path = Path(out_local_base + ".tex").resolve()

    # Write JSON + MD + TeX locally
    out = dict(merged_rows)
    out["_aggregate"] = aggregates
    out["_meta"] = {
        "game": args.game,
        "dataset": dataset,
        "n_samples": args.n_samples,
        "inputs": [str(p) for p in inputs],
        "models": models,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {json_path}")

    write_markdown(
        md_path,
        dataset=dataset,
        prompt_mode="real",
        models=models,
        samples=samples,
        results_by_model=merged_rows,
        aggregates=aggregates,
        judge_name="gemini-2.5-flash",
    )
    print(f"wrote {md_path}")

    metrics = ["tool_format", "actionable", "grounding", "action_relevance",
               "reasoning_similarity", "hallucination", "degenerate", "tok_s"]
    write_tex_table(tex_path, models, aggregates, metrics)
    print(f"wrote {tex_path}")

    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in (json_path, md_path, tex_path):
            dst = out_dir / p.name
            dst.write_bytes(p.read_bytes())
            print(f"copied → {dst}")


if __name__ == "__main__":
    main()
