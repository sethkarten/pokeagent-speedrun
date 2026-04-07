#!/usr/bin/env bash
# Bisect why the patched Ollama is slower than the stock one.
#
# Runs scripts/bench_ollama_quick.sh under five different configs and
# prints a side-by-side table of prefill/decode/total. Lets you see
# whether the slowdown is from:
#   - the source-built binary (vs the official release we backed up)
#   - the 1120-token image patch
#   - flash attention
#   - q8_0 KV cache quantization
#
# Each config requires a service restart, which we do via systemctl.
# The whole thing takes ~5 minutes (5 configs × ~30s call + restart).
#
# Usage:  sudo bash scripts/bisect_ollama_perf.sh
#         (sudo needed for systemctl + binary swap)
#
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "ERROR: must be run as root (use: sudo bash $0)" >&2
  exit 1
fi

DROPIN=/etc/systemd/system/ollama.service.d/gpu1.conf
INSTALL_PATH=/usr/local/bin/ollama       # where the daemon is loaded from
ORIG_BIN_SRC=/usr/local/bin/ollama.orig  # backup of stock 0.20.2 binary
PATCHED_BIN_SRC="${PATCHED_BIN_SRC:-$HOME/ollama-src/ollama}"  # our home build
BENCH=scripts/bench_ollama_quick.sh
PROMPT_FILE=/tmp/bench_prompt.txt
IMAGE_FILE=/tmp/bench_screenshot.png
RESULTS_FILE=/tmp/bisect_results.tsv

[[ -f "$BENCH" ]] || { echo "missing $BENCH" >&2; exit 1; }
[[ -f "$PROMPT_FILE" ]] || { echo "missing $PROMPT_FILE" >&2; exit 1; }
[[ -f "$IMAGE_FILE" ]]  || { echo "missing $IMAGE_FILE" >&2;  exit 1; }
[[ -f "$ORIG_BIN_SRC" ]] || { echo "missing $ORIG_BIN_SRC (run sudo cp /usr/local/bin/ollama /usr/local/bin/ollama.orig first)" >&2; exit 1; }
[[ -f "$PATCHED_BIN_SRC" ]] || { echo "missing $PATCHED_BIN_SRC (run scripts/build_ollama_high_image_tokens.sh first, or set PATCHED_BIN_SRC env var)" >&2; exit 1; }

# Save the current binary so we can restore at the end no matter what.
SAVED_CURRENT=$(mktemp /tmp/ollama-current.XXXXXX)
cp "$INSTALL_PATH" "$SAVED_CURRENT"
SAVED_DROPIN=$(mktemp /tmp/dropin.XXXXXX)
cp "$DROPIN" "$SAVED_DROPIN" 2>/dev/null || true

# Atomically swap a binary into /usr/local/bin/ollama. We use install
# instead of cp because cp gets ETXTBSY ("text file busy") if the
# kernel still has the old binary mmap'd from a not-yet-fully-exited
# daemon process — install creates a tempfile and rename(2)s it, which
# works regardless of what's holding the old inode. Returns nonzero on
# failure so the caller can decide what to do.
swap_binary() {
  local src="$1" dst="$2"
  install -m 0755 "$src" "$dst"
}

# Stop the daemon and wait for the process to be fully gone — even
# after systemctl stop returns, the runner subprocess can linger for
# a moment. Without this wait, the next binary swap can race.
stop_and_wait() {
  systemctl stop ollama 2>/dev/null || true
  for _ in 1 2 3 4 5 6 7 8; do
    if ! pgrep -x ollama >/dev/null 2>&1; then return 0; fi
    sleep 0.5
  done
  # Last resort
  pkill -9 -x ollama 2>/dev/null || true
  sleep 0.5
}

cleanup() {
  echo
  echo "==> Restoring original binary + drop-in"
  stop_and_wait
  if ! swap_binary "$SAVED_CURRENT" "$INSTALL_PATH"; then
    echo "    !! cleanup: failed to restore $INSTALL_PATH from $SAVED_CURRENT" >&2
  fi
  cp "$SAVED_DROPIN" "$DROPIN" 2>/dev/null || true
  systemctl daemon-reload
  systemctl start ollama || true
  rm -f "$SAVED_CURRENT" "$SAVED_DROPIN"
}
trap cleanup EXIT

> "$RESULTS_FILE"

write_dropin() {
  local fa="$1" kvq="$2"
  {
    echo "[Service]"
    echo "Environment=\"CUDA_VISIBLE_DEVICES=1\""
    echo "Environment=\"OLLAMA_KEEP_ALIVE=30m\""
    [[ "$fa"  == "on" ]] && echo "Environment=\"OLLAMA_FLASH_ATTENTION=1\""
    [[ "$kvq" == "on" ]] && echo "Environment=\"OLLAMA_KV_CACHE_TYPE=q8_0\""
  } > "$DROPIN"
}

# Each entry: label   binary_source_path   fa   kvq
# binary_source_path is where to copy FROM. The destination is always
# $INSTALL_PATH (/usr/local/bin/ollama). Sources have to live somewhere
# OTHER than the install path so we can swap them in/out without
# clobbering each other.
declare -a CONFIGS=(
  "stock_orig|$ORIG_BIN_SRC|off|off"
  "patched_no_fa_no_kv|$PATCHED_BIN_SRC|off|off"
  "patched_kv_only|$PATCHED_BIN_SRC|off|on"
  "patched_fa_only|$PATCHED_BIN_SRC|on|off"
  "patched_fa_and_kv|$PATCHED_BIN_SRC|on|on"
)

echo "==> Bisection plan: 5 configs (~30s each)"
for c in "${CONFIGS[@]}"; do echo "    $c"; done

for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r label bin fa kvq <<< "$cfg"
  echo
  echo "================================================================"
  echo "==> CONFIG: $label   (binary=$(basename $bin)  fa=$fa  kv=$kvq)"
  echo "================================================================"

  if [[ ! -f "$bin" ]]; then
    echo "    SKIP — binary not found at $bin"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$label" "SKIP" "0" "0" "0" "binary not found" >> "$RESULTS_FILE"
    continue
  fi

  # All failures inside this iteration are non-fatal so the bisection
  # can keep going. set +e for the body, set -e back at the end.
  set +e

  echo "    [1/5] stopping daemon"
  stop_and_wait
  echo "    [2/5] swapping binary $(basename $bin) -> /usr/local/bin/ollama"
  if ! swap_binary "$bin" "$INSTALL_PATH"; then
    echo "    !! swap_binary failed — skipping"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$label" "SWAPFAIL" "0" "0" "0" "binary swap failed" >> "$RESULTS_FILE"
    set -e
    continue
  fi
  echo "    [3/5] writing drop-in (fa=$fa kv=$kvq)"
  write_dropin "$fa" "$kvq"
  systemctl daemon-reload
  echo "    [4/5] starting daemon"
  systemctl start ollama
  # Wait for /api/version to respond — up to 30s.
  ready=0
  for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
    sleep 2
    if curl -s --max-time 3 http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
      ready=1
      break
    fi
  done
  if [[ $ready -ne 1 ]]; then
    echo "    !! daemon not reachable after 30s — skipping"
    systemctl status ollama --no-pager 2>&1 | tail -10
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$label" "DOWN" "0" "0" "0" "daemon down" >> "$RESULTS_FILE"
    set -e
    continue
  fi

  echo "    [5/5] running bench"
  bench_out=$(bash "$BENCH" 2>&1 || true)
  echo "$bench_out" | tail -20

  # Parse out the numbers (or mark FAIL on hang/timeout)
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
echo "==> RESULTS"
echo "================================================================"
printf '%-22s %6s %12s %12s %12s %10s\n' \
  "config" "status" "prompt_tok" "prefill_t/s" "decode_t/s" "total_s"
echo "----------------------------------------------------------------------------------"
while IFS=$'\t' read -r label status pt pr dr tot; do
  printf '%-22s %6s %12s %12s %12s %10s\n' "$label" "$status" "$pt" "$pr" "$dr" "$tot"
done < "$RESULTS_FILE"
echo
echo "Raw TSV at: $RESULTS_FILE"
