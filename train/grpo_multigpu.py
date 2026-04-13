"""Wrapper to launch grpo_offline.py with accelerate for multi-GPU DDP.

The GPU manager runs scripts with `python script.py`, but TRL's
GRPOTrainer needs `accelerate launch` for multi-GPU. This wrapper
detects available GPUs and re-execs via accelerate.

Usage (GPU manager submits this with gpu_count=2):
    python train/grpo_multigpu.py [all grpo_offline.py args...]
"""
import subprocess
import sys

import torch


def main() -> int:
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        print(f"Only {num_gpus} GPU(s) found — running single-GPU directly")
        from train.grpo_offline import main as grpo_main
        return grpo_main()

    print(f"Detected {num_gpus} GPUs — launching via accelerate DDP")
    cmd = [
        sys.executable, "-m", "accelerate.commands.launch",
        "--num_processes", str(num_gpus),
        "--mixed_precision", "bf16",
        "-m", "train.grpo_offline",
        *sys.argv[1:],
    ]
    print(f"CMD: {' '.join(cmd)}")
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
