#!/usr/bin/env bash
# Test the patched Ollama with FA OFF and KV-q OFF, isolating whether
# the perf hit is from the patch itself or from the speedup flags.
#
# Steps:
#   1. Rewrite the systemd drop-in with FA=off, KV=default (f16)
#   2. Restart ollama
#   3. Run bench_ollama_quick.sh and report
#   4. ALSO run a test with maxTokens=280 (we'll need to revert)... actually
#      no, we keep the patched binary; we just toggle the env vars.
#
# Usage: bash scripts/test_patched_no_speedups.sh
#        (will sudo for the systemd bits)
set -euo pipefail

DROPIN=/etc/systemd/system/ollama.service.d/gpu1.conf

step() { printf '\n\033[1;36m==>\033[0m %s\n' "$*"; }
info() { printf '    %s\n' "$*"; }

step "Caching sudo"
sudo -v

step "Rewriting drop-in: FA off, KV cache f16 (default)"
sudo tee "$DROPIN" >/dev/null <<'EOF'
[Service]
Environment="CUDA_VISIBLE_DEVICES=1"
Environment="OLLAMA_KEEP_ALIVE=30m"
EOF
sudo systemctl daemon-reload
sudo systemctl restart ollama

# Wait for daemon
for _ in 1 2 3 4 5 6 7 8; do
  if curl -s --max-time 3 http://127.0.0.1:11434/api/version >/dev/null; then break; fi
  sleep 2
done
info "daemon: $(curl -s http://127.0.0.1:11434/api/version)"

step "Running bench (patched binary, no FA, no KV-q)"
bash scripts/bench_ollama_quick.sh

step "Restoring drop-in with FA + KV-q for normal operation"
sudo bash scripts/pin_ollama_gpu1.sh
