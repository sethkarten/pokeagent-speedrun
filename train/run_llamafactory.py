"""Wrapper to launch a LlamaFactory training run from gpu-manager.

gpu-manager only knows how to run scripts inside our pokeagent-speedrun
repo's .venv. LlamaFactory is installed in a separate venv (because its
transformers version pin conflicts with Unsloth's). This wrapper just
shells out to the LF CLI from that other venv.

Usage (under gpu-manager):
    train/run_llamafactory.py /path/to/lf_config.yaml

The wrapper:
1. Locates the LF venv at /scratch/gpfs/CHIJ/milkkarten/llamafactory/.venv
2. Sets HF_HOME / HF_HUB_OFFLINE for the offline-only compute node
3. Execs llamafactory-cli train with the config path
"""

from __future__ import annotations

import os
import shutil
import sys

LF_VENV = "/scratch/gpfs/CHIJ/milkkarten/llamafactory/.venv"
LF_CLI = f"{LF_VENV}/bin/llamafactory-cli"


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: run_llamafactory.py <lf_config.yaml> [extra args]", file=sys.stderr)
        return 2

    if not os.path.exists(LF_CLI):
        print(f"FATAL: LlamaFactory CLI not found at {LF_CLI}", file=sys.stderr)
        return 1

    cfg_path = sys.argv[1]
    if not os.path.exists(cfg_path):
        print(f"FATAL: config file not found: {cfg_path}", file=sys.stderr)
        return 1

    extra_args = sys.argv[2:]

    # gpu-manager activates the pokeagent-speedrun venv before invoking
    # this wrapper. That sets VIRTUAL_ENV / PYTHONPATH / PYTHONHOME
    # which then leak into the LF CLI subprocess and break its
    # site-packages discovery (LF venv is python 3.12 in a different
    # path). Clear them so the LF venv's python self-discovers cleanly.
    env = {k: v for k, v in os.environ.items()
           if k not in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME",
                        "PYTHONNOUSERSITE")}
    # Drop the pokeagent venv's bin from PATH and prepend the LF venv's.
    lf_bin = f"{LF_VENV}/bin"
    path_parts = [p for p in env.get("PATH", "").split(":")
                  if "pokeagent-speedrun/.venv" not in p]
    env["PATH"] = lf_bin + ":" + ":".join(path_parts)
    env["VIRTUAL_ENV"] = LF_VENV

    # Compute nodes have no internet — point HF cache at the pre-staged
    # local copy and force offline mode.
    env.setdefault("HF_HOME", "/scratch/gpfs/CHIJ/milkkarten/huggingface")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    # LF pyproject pins transformers<=5.2.0 (stale, predates gemma4
    # support) and the runtime check raises ImportError otherwise.
    # We need transformers 5.5.x for the gemma4 model_type, so skip
    # LF's check. Hint comes from the error message itself.
    env.setdefault("DISABLE_VERSION_CHECK", "1")

    cmd = [LF_CLI, "train", cfg_path, *extra_args]
    print(f"=== launching: {' '.join(cmd)} ===", flush=True)
    print(f"=== HF_HOME={env['HF_HOME']} ===", flush=True)

    # exec replaces this process; the LF run becomes the SLURM job's
    # main process so signals/return codes propagate cleanly.
    os.execvpe(cmd[0], cmd, env)


if __name__ == "__main__":
    sys.exit(main())
