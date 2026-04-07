#!/usr/bin/env bash
# Quick Ollama smoke + benchmark using the captured real harness fixture.
#
# Sends one ~12K-token prompt + a real Folder Dungeon screenshot to
# gemma4:26b through the running daemon, with a hard 60s wall-clock
# timeout. Reports prefill / decode rates, total wall, current
# environment, and current VRAM. Designed to:
#   - Confirm Ollama is reachable + the model loads
#   - Detect Flash Attention hangs (issue #15350) early — if the
#     request takes >60s the script bails so you don't wait forever
#   - Give a single-number perf check after toggling
#     OLLAMA_FLASH_ATTENTION / OLLAMA_KV_CACHE_TYPE / image-token patches
#
# Usage:
#   bash scripts/bench_ollama_quick.sh                  # default model gemma4:26b
#   MODEL=gemma4:31b bash scripts/bench_ollama_quick.sh
#   HOST=http://127.0.0.1:11434 bash scripts/bench_ollama_quick.sh
#
set -euo pipefail

MODEL="${MODEL:-gemma4:26b}"
HOST="${HOST:-http://127.0.0.1:11434}"
PROMPT_FILE="${PROMPT_FILE:-/tmp/bench_prompt.txt}"
IMAGE_FILE="${IMAGE_FILE:-/tmp/bench_screenshot.png}"

step()  { printf '\n\033[1;36m==>\033[0m %s\n' "$*"; }
info()  { printf '    %s\n' "$*"; }
fail()  { printf '\033[1;31mxx\033[0m  %s\n' "$*" >&2; exit 1; }

# ---- 0. Sanity ------------------------------------------------------------
step "Environment"
info "host:        $HOST"
info "model:       $MODEL"
info "prompt:      $PROMPT_FILE"
info "image:       $IMAGE_FILE"
[[ -f "$PROMPT_FILE" ]] || fail "missing prompt fixture: $PROMPT_FILE (run scripts/bench_ollama.py once to recreate, or have the agent run a step)"
[[ -f "$IMAGE_FILE" ]]  || fail "missing image fixture: $IMAGE_FILE"

step "Ollama daemon"
ver_json="$(curl -s --max-time 3 "$HOST/api/version" || true)"
if [[ -z "$ver_json" ]]; then
  fail "daemon not reachable at $HOST"
fi
info "version: $ver_json"

env_line="$(systemctl show ollama -p Environment --no-pager 2>/dev/null || true)"
if [[ -n "$env_line" ]]; then
  info "service env (filtered):"
  echo "$env_line" | tr ' ' '\n' | grep -E 'CUDA_VISIBLE|OLLAMA_' | sed 's/^/      /' || true
fi

step "GPU baseline"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits | sed 's/^/    /'

# ---- 1. Send the call ----------------------------------------------------
step "Sending one orchestrator-sized request (60s wall-clock cap)"
.venv/bin/python - <<PYEOF
import base64, json, sys, time
import requests

HOST = "${HOST}"
MODEL = "${MODEL}"
PROMPT_FILE = "${PROMPT_FILE}"
IMAGE_FILE = "${IMAGE_FILE}"

prompt = open(PROMPT_FILE).read()
image_b64 = base64.b64encode(open(IMAGE_FILE, "rb").read()).decode()
print(f"  prompt chars : {len(prompt):,}")
print(f"  image bytes  : {len(image_b64):,} base64")

# Send a system message + a couple of tool declarations matching what
# the real OllamaBackend sends for the browser harness — without these,
# gemma4 sees the prompt as raw structured text and goes into a
# degenerate completion mode where the parser strips everything as
# thinking and content comes back empty. With them, it cleanly returns
# a tool_call which is what the agent actually consumes.
TOOLS = [
    {"type": "function", "function": {
        "name": "mouse_click",
        "description": "Click at (x, y) on the game canvas.",
        "parameters": {"type": "object", "properties": {
            "x": {"type": "integer"},
            "y": {"type": "integer"},
            "reasoning": {"type": "string"},
        }, "required": ["x", "y", "reasoning"]}}},
    {"type": "function", "function": {
        "name": "press_keys",
        "description": "Press one or more keys.",
        "parameters": {"type": "object", "properties": {
            "keys": {"type": "array", "items": {"type": "string"}},
            "reasoning": {"type": "string"},
        }, "required": ["keys", "reasoning"]}}},
]
SYSTEM_MSG = (
    "You are playing a browser game. Look at the screenshot and call ONE "
    "tool to take an action. Do not output plain text — only call a tool."
)
body = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": prompt, "images": [image_b64]},
    ],
    "tools": TOOLS,
    "think": False,
    "stream": False,
    "options": {
        "num_ctx": 32768,
        "num_predict": 500,
        "temperature": 0.7,
    },
}

t0 = time.perf_counter()
try:
    r = requests.post(f"{HOST}/api/chat", json=body, timeout=60)
except requests.exceptions.Timeout:
    print("\n  !! 60s WALL-CLOCK TIMEOUT — likely a Flash Attention hang.")
    print("  !! revert with: sudo bash scripts/pin_ollama_gpu1.sh --no-fa")
    sys.exit(2)
except Exception as e:
    print(f"\n  !! request error: {e}")
    sys.exit(3)
wall = time.perf_counter() - t0

if r.status_code != 200:
    print(f"  !! HTTP {r.status_code}: {r.text[:300]}")
    sys.exit(4)

d = r.json()
inp = d.get("prompt_eval_count", 0)
out = d.get("eval_count", 0)
inp_dur = (d.get("prompt_eval_duration") or 0) / 1e9
out_dur = (d.get("eval_duration") or 0) / 1e9
load_dur = (d.get("load_duration") or 0) / 1e9

prefill_rate = (inp / inp_dur) if inp_dur > 0 else 0
decode_rate  = (out / out_dur) if out_dur > 0 else 0

print()
print(f"  total_wall      : {wall:.2f}s")
print(f"  load_duration   : {load_dur:.2f}s")
print(f"  prompt_eval     : {inp:5d} tokens in {inp_dur:6.2f}s  ({prefill_rate:7.0f} tok/s)")
print(f"  eval (decode)   : {out:5d} tokens in {out_dur:6.2f}s  ({decode_rate:7.0f} tok/s)")
print()
msg = d.get("message", {})
content = msg.get("content", "")
tool_calls = msg.get("tool_calls", [])
print(f"  response chars  : {len(content)}")
if content:
    print(f"  response sample : {content[:200]!r}")
print(f"  tool_calls      : {len(tool_calls)}")
for tc in tool_calls[:3]:
    fn = tc.get("function", {})
    args = json.dumps(fn.get("arguments", {}))
    print(f"    - {fn.get('name')}({args[:200]})")
if not content and not tool_calls:
    print("  !! WARNING: model returned neither content nor tool_calls.")
    print("  !! gemma4 may be eating output as thinking — re-test the agent.")
PYEOF
rc=$?

step "GPU after"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits | sed 's/^/    /'

if [[ $rc -ne 0 ]]; then
  exit $rc
fi

step "OK — Ollama is healthy at the current settings"
