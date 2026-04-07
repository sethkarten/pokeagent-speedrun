#!/bin/bash
#
# Start a second Ollama daemon pinned to GPU 0, port 11435.
#
# The system Ollama service runs on GPU 1, port 11434 (see
# scripts/pin_ollama_gpu1.sh). This script launches a second instance
# on GPU 0, port 11435, so we can drive two harnesses in parallel —
# one per 5090. Each harness sets OLLAMA_HOST to point at its own
# instance.
#
# Both instances share the same ~/.ollama/models directory (read-only
# mmap of model files works fine concurrently), so we don't need to
# pull the model twice.
#
# Idempotent: if port 11435 is already serving, this script is a
# no-op apart from health verification.
#
# Usage:
#   bash scripts/start_second_ollama_gpu0.sh             # start in background
#   bash scripts/start_second_ollama_gpu0.sh --stop      # kill it
#   bash scripts/start_second_ollama_gpu0.sh --logs      # tail the daemon log
#
set -euo pipefail

PORT=11435
HOST="127.0.0.1:${PORT}"
LOG_FILE="/tmp/ollama-gpu0.log"
PID_FILE="/tmp/ollama-gpu0.pid"
KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-30m}"

if [[ "${1:-}" == "--stop" ]]; then
  if [[ -f "$PID_FILE" ]]; then
    pid=$(cat "$PID_FILE")
    if kill -0 "$pid" 2>/dev/null; then
      echo "stopping ollama-gpu0 (pid $pid)..."
      kill "$pid"
      sleep 1
      if kill -0 "$pid" 2>/dev/null; then
        kill -9 "$pid"
      fi
      echo "stopped"
    else
      echo "stale pid file (process $pid is gone)"
    fi
    rm -f "$PID_FILE"
  else
    echo "no pid file at $PID_FILE — looking for stray ollama on port $PORT"
    pids=$(ss -tlnp 2>/dev/null | grep ":${PORT} " | grep -oP 'pid=\K[0-9]+' || true)
    if [[ -n "$pids" ]]; then
      echo "found stray pid(s): $pids"
      for p in $pids; do kill "$p" 2>/dev/null || true; done
    fi
  fi
  exit 0
fi

if [[ "${1:-}" == "--logs" ]]; then
  exec tail -f "$LOG_FILE"
fi

# Already running?
if curl -s --max-time 2 "http://${HOST}/api/version" >/dev/null 2>&1; then
  echo "ollama already running on $HOST"
  if [[ -f "$PID_FILE" ]]; then
    echo "  pid: $(cat $PID_FILE)"
  fi
  echo "  models loaded:"
  curl -s "http://${HOST}/api/ps" | head -c 500
  echo
  exit 0
fi

echo "Starting second ollama daemon on GPU 0, port $PORT..."
echo "  log: $LOG_FILE"
echo "  pid file: $PID_FILE"
echo "  num_batch: ${OLLAMA_NUM_BATCH:-2048} (matches our patched 1120-image-token build)"
echo

# CUDA_VISIBLE_DEVICES=0 pins to the first 5090. OLLAMA_HOST sets both
# the bind address and the port. The other env vars match what
# scripts/pin_ollama_gpu1.sh applies to the system service: keep-alive
# for 30m, FA off, KV-q off (the bisection-proven fast config for our
# patched build).
# Share the system service's model store so we don't have to pull
# gemma4:26b a second time. The system service runs as the `ollama`
# user and stores models in /usr/share/ollama/.ollama/models. Group
# perms make those files readable by everyone in the `ollama` group
# (this user is, via the ollama package install). mmap-based reads
# are concurrent-safe across processes.
OLLAMA_MODELS_DEFAULT=/usr/share/ollama/.ollama/models
OLLAMA_MODELS_DIR="${OLLAMA_MODELS:-$OLLAMA_MODELS_DEFAULT}"
echo "  using model store: $OLLAMA_MODELS_DIR"

# Match the system service config from scripts/pin_ollama_gpu1.sh:
# Flash Attention OFF and KV cache f16 (default). The bisection
# showed FA + q8_0 is a 16x slowdown, not a speedup, on our patched
# 1120-image-token gemma4:26b. Without these explicit "off" flags
# Ollama may default FA on, which causes layer spillover to CPU
# (only 8/31 layers fit on GPU when FA is on with 32k ctx).
CUDA_VISIBLE_DEVICES=0 \
OLLAMA_HOST="$HOST" \
OLLAMA_MODELS="$OLLAMA_MODELS_DIR" \
OLLAMA_KEEP_ALIVE="$KEEP_ALIVE" \
OLLAMA_NUM_PARALLEL=1 \
OLLAMA_MAX_LOADED_MODELS=1 \
OLLAMA_FLASH_ATTENTION=0 \
OLLAMA_KV_CACHE_TYPE=f16 \
nohup /usr/local/bin/ollama serve >"$LOG_FILE" 2>&1 &

pid=$!
echo "$pid" > "$PID_FILE"
echo "started ollama-gpu0 with pid $pid"

# Wait for /api/version to respond.
echo "waiting for daemon to be ready..."
for i in $(seq 1 30); do
  sleep 1
  if curl -s --max-time 2 "http://${HOST}/api/version" >/dev/null 2>&1; then
    echo "ready after ${i}s"
    break
  fi
done

if ! curl -s --max-time 2 "http://${HOST}/api/version" >/dev/null 2>&1; then
  echo "ERROR: ollama on $HOST did not become ready in 30s — see $LOG_FILE" >&2
  exit 1
fi

echo
echo "ollama-gpu0 ready:"
echo "  HOST=http://${HOST}"
echo "  pid: $pid"
echo "  GPU: 0"
echo
echo "Drive a harness against this instance with:"
echo "  OLLAMA_HOST=http://${HOST} BACKEND=ollama MODEL=gemma4:26b ./run_browser.sh"
echo
echo "Stop with:  bash scripts/start_second_ollama_gpu0.sh --stop"
echo "Tail logs:  bash scripts/start_second_ollama_gpu0.sh --logs"
