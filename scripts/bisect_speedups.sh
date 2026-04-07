#!/usr/bin/env bash
# Bisect FA vs KV-q on the patched Ollama build.
#
# We already know:
#   stock (no FA, no KV-q):    prefill ~7000 tok/s   total ~6.4s   ← official baseline
#   patched (no FA, no KV-q):  prefill ~4000 tok/s   total ~5.5s   ← FAST, our build
#   patched (FA + KV-q):       prefill ~450  tok/s   total ~25s    ← SLOW, current state
#
# So one or both speedup flags is interacting badly with the 1120-token
# vision encoder. This script tries the four patched configs in turn and
# reports a side-by-side table so we can pick the right combination.
#
# Each config is just an env var change — no binary swap, no rebuild.
# The whole bisection takes ~3-5 minutes.
#
# Usage:  bash scripts/bisect_speedups.sh

set -euo pipefail

DROPIN=/etc/systemd/system/ollama.service.d/gpu1.conf
RESULTS_FILE=/tmp/bisect_speedups.tsv

step() { printf '\n\033[1;36m==>\033[0m %s\n' "$*"; }
info() { printf '    %s\n' "$*"; }

step "Caching sudo"
sudo -v

# Save current dropin so we can restore at the end no matter what
SAVED_DROPIN=$(mktemp /tmp/dropin.XXXXXX)
sudo cp "$DROPIN" "$SAVED_DROPIN" 2>/dev/null || true
trap 'echo "==> Restoring drop-in"; sudo cp "$SAVED_DROPIN" "$DROPIN" 2>/dev/null || true; sudo systemctl daemon-reload; sudo systemctl restart ollama; rm -f "$SAVED_DROPIN"' EXIT

> "$RESULTS_FILE"

write_dropin() {
  local fa="$1" kvq="$2"
  sudo tee "$DROPIN" >/dev/null <<EOF
[Service]
Environment="CUDA_VISIBLE_DEVICES=1"
Environment="OLLAMA_KEEP_ALIVE=30m"
$([[ "$fa"  == "on" ]] && echo "Environment=\"OLLAMA_FLASH_ATTENTION=1\"")
$([[ "$kvq" == "on" ]] && echo "Environment=\"OLLAMA_KV_CACHE_TYPE=q8_0\"")
EOF
}

# Each entry: label   fa   kvq
declare -a CONFIGS=(
  "no_fa_no_kv|off|off"
  "kv_only|off|on"
  "fa_only|on|off"
  "fa_and_kv|on|on"
)

for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r label fa kvq <<< "$cfg"
  echo
  echo "================================================================"
  echo "==> CONFIG: $label   (fa=$fa  kv=$kvq)"
  echo "================================================================"

  set +e
  echo "    [1/4] writing drop-in"
  write_dropin "$fa" "$kvq"
  echo "    [2/4] reloading + restarting daemon"
  sudo systemctl daemon-reload
  sudo systemctl restart ollama
  ready=0
  for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
    sleep 2
    if curl -s --max-time 3 http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
      ready=1
      break
    fi
  done
  if [[ $ready -ne 1 ]]; then
    echo "    !! daemon down — skipping"
    printf '%s\t%s\t%s\t%s\t%s\n' "$label" "DOWN" "0" "0" "0" >> "$RESULTS_FILE"
    set -e
    continue
  fi
  echo "    [3/4] waiting for keep-alive eviction (5s) so model reloads with new env"
  sleep 5
  echo "    [4/4] running bench"
  bench_out=$(bash scripts/bench_ollama_quick.sh 2>&1 || true)
  echo "$bench_out" | tail -15

  total=$(echo "$bench_out" | grep -oE 'total_wall[[:space:]]*: *[0-9.]+s' | head -1 | grep -oE '[0-9.]+' || echo 0)
  prefill_rate=$(echo "$bench_out" | grep -oE 'prompt_eval.*\([[:space:]]*[0-9]+ tok/s\)' | head -1 | grep -oE '[0-9]+ tok/s' | grep -oE '[0-9]+' || echo 0)
  decode_rate=$(echo "$bench_out" | grep -oE 'eval \(decode\).*\([[:space:]]*[0-9]+ tok/s\)' | head -1 | grep -oE '[0-9]+ tok/s' | grep -oE '[0-9]+' || echo 0)
  prompt_tok=$(echo "$bench_out" | grep -oE 'prompt_eval[[:space:]]*: *[0-9]+ tokens' | head -1 | grep -oE '[0-9]+' || echo 0)

  if echo "$bench_out" | grep -q "60s WALL-CLOCK TIMEOUT"; then
    status="HANG"
  else
    status="OK"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$label" "$status" "$prompt_tok" "$prefill_rate" "$decode_rate" "$total" >> "$RESULTS_FILE"
  set -e
done

echo
echo "================================================================"
echo "==> RESULTS (patched binary, varying FA + KV-q only)"
echo "================================================================"
printf '%-18s %6s %12s %14s %14s %10s\n' \
  "config" "status" "prompt_tok" "prefill_t/s" "decode_t/s" "total_s"
echo "----------------------------------------------------------------------------------"
while IFS=$'\t' read -r label status pt pr dr tot; do
  printf '%-18s %6s %12s %14s %14s %10s\n' "$label" "$status" "$pt" "$pr" "$dr" "$tot"
done < "$RESULTS_FILE"
echo
echo "Reference: stock binary (no FA, no KV-q): prefill ~7050 tok/s, decode ~167 tok/s, total ~6.43s"
echo "Raw TSV at: $RESULTS_FILE"
