"""Transformers-backed eval for models that can't be served via Ollama (vision-split GGUFs).

Loads the merged safetensors directly, runs greedy/argmax decoding with the Gemma4
processor, and writes results in the same per-sample JSON format as run_eval.py so
results can be merged into the main comparison output.

Usage:
    uv run eval/run_eval_transformers.py \
        --models-config eval_models.json \
        --dataset emerald_v3 --n-samples 20 \
        --output data/eval_emerald_tf.json --md data/eval_emerald_tf.md

``models-config`` is JSON mapping display_name -> merged_model_path, e.g.:
    {"gemma4-emerald:31b": "/path/to/merged/31b_emerald_v3", ...}
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.data_loader import Sample, load_samples  # noqa: E402
from eval.scorers import (  # noqa: E402
    score_actionable,
    score_degenerate,
    score_grounding,
    score_tool_format,
)
from eval.run_eval import (  # noqa: E402
    aggregate,
    score_sample,
    write_markdown,
)

for _c in [
    os.environ.get("HF_HOME"),
    "/mnt/storage/models/huggingface",
    "/data1/milkkarten/.cache/huggingface",
]:
    if _c and os.path.isdir(_c):
        os.environ["HF_HOME"] = _c
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        break

logging.basicConfig(
    level=os.environ.get("EVAL_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("eval_tf")


def _decode_image(image_b64: Optional[str]) -> Optional[Image.Image]:
    if not image_b64:
        return None
    return Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")


def _load_model(model_path: str):
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

    logger.info("loading processor + model from %s (bnb 4-bit)", model_path)
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(model_path)
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        quantization_config=quant_cfg,
        device_map="auto",
        max_memory={0: "28GiB", 1: "28GiB", "cpu": "80GiB"},
        attn_implementation="sdpa",
    )
    model.eval()
    logger.info("model loaded in %.1fs", time.time() - t0)
    return processor, model


def _free_model(model):
    del model
    torch.cuda.empty_cache()


def _generate(processor, model, prompt: str, image: Optional[Image.Image],
              num_predict: int, temperature: float) -> tuple[str, float, float]:
    messages = [{"role": "user", "content": []}]
    if image is not None:
        messages[0]["content"].append({"type": "image", "image": image})
    messages[0]["content"].append({"type": "text", "text": prompt})

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]
    t0 = time.time()
    with torch.inference_mode():
        gen_kwargs = dict(
            max_new_tokens=num_predict,
            do_sample=temperature > 0,
        )
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
        out = model.generate(**inputs, **gen_kwargs)
    elapsed = time.time() - t0
    new_tokens = out[0][input_len:]
    text = processor.decode(new_tokens, skip_special_tokens=True)
    tok_s = len(new_tokens) / elapsed if elapsed > 0 else 0.0
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
        model_path = models_cfg[display_name]
        logger.info("===== %s (%s) =====", display_name, model_path)
        try:
            processor, model = _load_model(model_path)
        except Exception as e:
            logger.error("failed to load %s: %s", display_name, e)
            continue

        try:
            for sample in samples:
                prompt_text = sample.prompt if args.prompt_mode == "real" else sample.simplified_prompt
                image = _decode_image(sample.image_b64)
                try:
                    response, elapsed, tok_s = _generate(
                        processor, model, prompt_text, image,
                        args.num_predict, args.temperature,
                    )
                except Exception as e:
                    response = f"ERROR: {e}"
                    elapsed = 0.0
                    tok_s = 0.0

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
            _free_model(model)
            del processor
            torch.cuda.empty_cache()

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
        "backend": "transformers",
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
    ap.add_argument("--models-config", required=True, help="JSON file mapping display_name -> model_path")
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
    args = ap.parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
