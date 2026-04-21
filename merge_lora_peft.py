"""Merge a LoRA adapter into the base Gemma4 model using raw PEFT.

Fallback for models not in Unsloth's supported list (e.g. gemma-4-26B-A4B-it).
Uses PeftModel.from_pretrained + merge_and_unload.

Usage:
    python merge_lora_peft.py \
        --base unsloth/gemma-4-26B-A4B-it \
        --adapter /path/to/adapter \
        --output /path/to/merged \
        --device cpu
"""

import argparse
import os
import torch
import time

# Auto-discover HF cache across machines (della/local/Cynthia)
_hf_candidates = [
    os.environ.get("HF_HOME"),
    "/mnt/storage/models/huggingface",
    "/scratch/gpfs/CHIJ/milkkarten/huggingface",
    "/data1/milkkarten/.cache/huggingface",
]
for _c in _hf_candidates:
    if _c and os.path.isdir(_c):
        os.environ["HF_HOME"] = _c
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base model ID (e.g. unsloth/gemma-4-26B-A4B-it)")
    ap.add_argument("--adapter", required=True, help="Path to LoRA adapter checkpoint")
    ap.add_argument("--output", required=True, help="Output path for merged model")
    ap.add_argument("--device", default="cpu", help="Device to load on (cpu or cuda:N)")
    args = ap.parse_args()

    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import PeftModel

    print(f"Loading base model {args.base} on {args.device}...")
    t0 = time.time()

    load_kwargs = {"dtype": torch.bfloat16}
    if args.device == "cpu":
        load_kwargs["device_map"] = "cpu"
    else:
        load_kwargs["device_map"] = {"": args.device}

    model = AutoModelForImageTextToText.from_pretrained(args.base, **load_kwargs)
    print(f"Base model loaded in {time.time()-t0:.0f}s")

    print(f"Loading adapter from {args.adapter}...")
    t1 = time.time()
    model = PeftModel.from_pretrained(model, args.adapter)
    print(f"Adapter loaded in {time.time()-t1:.0f}s")

    print("Merging LoRA into base weights...")
    t2 = time.time()
    model = model.merge_and_unload()
    print(f"Merged in {time.time()-t2:.0f}s")

    print(f"Saving merged model to {args.output}...")
    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True)

    # Save processor/tokenizer - prefer adapter's (may have customizations), fallback to base
    try:
        processor = AutoProcessor.from_pretrained(args.adapter)
        print("Saving processor from adapter dir...")
    except Exception as e:
        print(f"Adapter has no processor ({e}); loading from base {args.base}")
        processor = AutoProcessor.from_pretrained(args.base)
    processor.save_pretrained(args.output)

    # Copy chat_template.jinja if adapter has one but processor didn't pick it up
    adapter_ct = os.path.join(args.adapter, "chat_template.jinja")
    if os.path.exists(adapter_ct):
        import shutil
        shutil.copy(adapter_ct, os.path.join(args.output, "chat_template.jinja"))
        print(f"Copied chat_template.jinja from adapter")

    print(f"Done! Merged model at {args.output}")
    total_size = sum(
        os.path.getsize(os.path.join(args.output, f))
        for f in os.listdir(args.output)
        if f.endswith('.safetensors')
    )
    print(f"Model size: {total_size/1e9:.1f} GB")


if __name__ == "__main__":
    main()
