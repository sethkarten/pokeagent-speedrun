#!/bin/bash
# Pin the system Ollama service to GPU 1 only.
#
# Adds a systemd drop-in (without modifying the original unit file) that
# sets CUDA_VISIBLE_DEVICES=1 and OLLAMA_KEEP_ALIVE=30m, then reloads and
# restarts the ollama service. Idempotent — re-running just rewrites the
# drop-in and bounces the daemon.
#
# Usage:  sudo bash scripts/pin_ollama_gpu1.sh
# Revert: sudo bash scripts/pin_ollama_gpu1.sh --revert
#
set -euo pipefail

DROPIN_DIR=/etc/systemd/system/ollama.service.d
DROPIN_FILE="$DROPIN_DIR/gpu1.conf"

if [[ "${1:-}" == "--revert" ]]; then
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
fi

if [[ $EUID -ne 0 ]]; then
  echo "ERROR: must be run as root (use: sudo bash $0)" >&2
  exit 1
fi

mkdir -p "$DROPIN_DIR"
cat > "$DROPIN_FILE" <<'EOF'
[Service]
Environment="CUDA_VISIBLE_DEVICES=1"
Environment="OLLAMA_KEEP_ALIVE=30m"
EOF
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
echo "--- gpu visibility check ---"
echo "Loading a tiny model briefly to confirm GPU 1 is the only one used..."
echo "(Skipping — run scripts/bench_ollama.py to actually exercise it.)"

echo
echo "Done. Ollama is now pinned to GPU 1 (CUDA_VISIBLE_DEVICES=1)."
echo "To revert: sudo bash scripts/pin_ollama_gpu1.sh --revert"
