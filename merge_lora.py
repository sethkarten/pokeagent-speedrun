"""Merge a LoRA adapter into the base Gemma4 model and save.

For models too large for a single GPU (26B = 52 GB bf16), loads on CPU.
For smaller models (E4B = 16 GB), can load on GPU for speed.

Usage:
    python merge_lora.py \
        --base unsloth/gemma-4-26B-A4B-it \
        --adapter /data1/milkkarten/Research/adapters/26b_emerald_v3 \
        --output /data1/milkkarten/Research/merged/26b_emerald_v3 \
        --device cpu
"""

import argparse
import os
import torch
import time

os.environ.setdefault("HF_HOME", "/data1/milkkarten/.cache/huggingface")

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

    load_kwargs = {"torch_dtype": torch.bfloat16}
    if args.device == "cpu":
        load_kwargs["device_map"] = "cpu"
    else:
        load_kwargs["device_map"] = {"": args.device}

    model = AutoModelForImageTextToText.from_pretrained(args.base, **load_kwargs)
    print(f"Base model loaded in {time.time()-t0:.0f}s")

    print(f"Loading adapter from {args.adapter}...")
    model = PeftModel.from_pretrained(model, args.adapter)
    print(f"Adapter loaded")

    print("Merging LoRA into base weights...")
    t1 = time.time()
    model = model.merge_and_unload()
    print(f"Merged in {time.time()-t1:.0f}s")

    print(f"Saving merged model to {args.output}...")
    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True)

    # Also save processor/tokenizer
    print("Saving processor...")
    processor = AutoProcessor.from_pretrained(args.base)
    processor.save_pretrained(args.output)

    print(f"Done! Merged model at {args.output}")
    total_size = sum(
        os.path.getsize(os.path.join(args.output, f))
        for f in os.listdir(args.output)
        if f.endswith('.safetensors')
    )
    print(f"Model size: {total_size/1e9:.1f} GB")

if __name__ == "__main__":
    main()
