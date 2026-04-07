#!/bin/bash
# Pin the system Ollama service to GPU 1.
#
# Default config (what an empirical bisection turned out to be fastest
# for our patched gemma4:26b at 1120 image tokens):
#   - CUDA_VISIBLE_DEVICES=1     pins to the second 5090
#   - OLLAMA_KEEP_ALIVE=30m      model stays resident between agent steps
#   - Flash Attention OFF        (it's a SLOWDOWN, not a speedup, for the
#                                1120-token vision encoder — turning it
#                                on dropped prefill from 4084 → 450 tok/s
#                                in our bench. Likely related to ollama
#                                issue #15350 which reports FA hangs on
#                                gemma4 + large prompts. Re-test with
#                                future Ollama versions.)
#   - KV cache f16 (default)     same story — q8_0 is bundled with the
#                                FA slowdown when both are enabled, and
#                                the few GB of VRAM saved isn't worth it
#                                on a 32 GB card.
#
# If you ever want to re-test the speedups (e.g. after upgrading Ollama
# or trying a different model), pass --enable-fa / --enable-kv-q
# explicitly. Then re-run scripts/bisect_speedups.sh to verify.
#
# Idempotent — re-running just rewrites the drop-in and bounces the daemon.
#
# Usage:
#   sudo bash scripts/pin_ollama_gpu1.sh                  # default (no FA, no KV-q)
#   sudo bash scripts/pin_ollama_gpu1.sh --enable-fa      # turn FA on (probably slower!)
#   sudo bash scripts/pin_ollama_gpu1.sh --enable-kv-q    # turn KV-q on (probably slower!)
#   sudo bash scripts/pin_ollama_gpu1.sh --revert         # remove drop-in entirely
#
set -euo pipefail

DROPIN_DIR=/etc/systemd/system/ollama.service.d
DROPIN_FILE="$DROPIN_DIR/gpu1.conf"

# Defaults are now OFF for both — the bisection showed they make our
# patched gemma4:26b SLOWER, not faster.
ENABLE_FA=0
ENABLE_KV_Q=0

for arg in "$@"; do
  case "$arg" in
    --revert)
      if [[ -f "$DROPIN_FILE" ]]; then
        rm -v "$DROPIN_FILE"
        rmdir --ignore-fail-on-non-empty "$DROPIN_DIR" 2>/dev/null || true
        systemctl daemon-reload
        systemctl restart ollama
        echo "ollama drop-in removed; service restarted with default GPU visibility"
      else
        echo "no drop-in at $DROPIN_FILE — nothing to revert"
      fi
      exit 0
      ;;
    --enable-fa)
      ENABLE_FA=1
      ;;
    --enable-kv-q)
      ENABLE_KV_Q=1
      ;;
    # legacy aliases — kept so existing notes / muscle memory don't break
    --no-fa)    ENABLE_FA=0 ;;
    --no-kv-q)  ENABLE_KV_Q=0 ;;
    *)
      echo "unknown arg: $arg" >&2
      exit 1
      ;;
  esac
done

if [[ $EUID -ne 0 ]]; then
  echo "ERROR: must be run as root (use: sudo bash $0)" >&2
  exit 1
fi

mkdir -p "$DROPIN_DIR"
{
  echo "[Service]"
  echo "Environment=\"CUDA_VISIBLE_DEVICES=1\""
  echo "Environment=\"OLLAMA_KEEP_ALIVE=30m\""
  if [[ "$ENABLE_FA" == "1" ]]; then
    echo "Environment=\"OLLAMA_FLASH_ATTENTION=1\""
  fi
  if [[ "$ENABLE_KV_Q" == "1" ]]; then
    echo "Environment=\"OLLAMA_KV_CACHE_TYPE=q8_0\""
  fi
} > "$DROPIN_FILE"
echo "wrote $DROPIN_FILE:"
cat "$DROPIN_FILE"
echo

systemctl daemon-reload
systemctl restart ollama
sleep 2

echo
echo "--- ollama service environment ---"
systemctl show ollama -p Environment --no-pager

echo
echo "--- ollama version ---"
curl -s --max-time 3 http://127.0.0.1:11434/api/version || echo "(daemon not reachable)"
echo

echo
echo "Pin + speedups applied:"
echo "  GPU 1 only (CUDA_VISIBLE_DEVICES=1)"
echo "  keep-alive 30m"
echo "  flash attention: $([[ "$ENABLE_FA" == "1" ]] && echo ON || echo OFF)"
echo "  KV cache q8_0:   $([[ "$ENABLE_KV_Q" == "1" ]] && echo ON || echo OFF)"
echo
echo "Verify with:  bash scripts/bench_ollama_quick.sh"
echo "Disable FA:   sudo bash scripts/pin_ollama_gpu1.sh --no-fa"
echo "Revert all:   sudo bash scripts/pin_ollama_gpu1.sh --revert"
