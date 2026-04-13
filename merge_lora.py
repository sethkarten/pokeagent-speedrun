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

    # Use Unsloth's loader — raw PeftModel.from_pretrained fails because
    # Gemma4 has ClippableLinear wrappers in the audio tower that PEFT
    # can't inject LoRA into. Unsloth handles this automatically.
    from unsloth import FastVisionModel

    print(f"Loading adapter {args.adapter} via Unsloth...")
    t0 = time.time()

    model, processor = FastVisionModel.from_pretrained(
        args.adapter,
        load_in_4bit=False,    # bf16 for clean merge
        use_gradient_checkpointing=False,
        max_seq_length=8192,
    )
    print(f"Model + adapter loaded in {time.time()-t0:.0f}s")

    # Merge LoRA weights into base and save as a standard HF model
    print("Merging LoRA into base weights...")
    t1 = time.time()
    model.save_pretrained_merged(
        args.output,
        processor,
        save_method="merged_16bit",
    )
    print(f"Merged and saved in {time.time()-t1:.0f}s")
    print(f"Output: {args.output}")

    print(f"Done! Merged model at {args.output}")
    total_size = sum(
        os.path.getsize(os.path.join(args.output, f))
        for f in os.listdir(args.output)
        if f.endswith('.safetensors')
    )
    print(f"Model size: {total_size/1e9:.1f} GB")

if __name__ == "__main__":
    main()
