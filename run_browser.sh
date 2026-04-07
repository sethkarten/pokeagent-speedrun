#!/bin/bash
set -a && source .env && set +a

# Create timestamped log file
LOG_DIR="run_data/browser_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(date +%Y%m%d_%H%M%S).log"

echo "Logging to $LOG_FILE"
PORT=${PORT:-8002}
echo "Watch live at http://localhost:$PORT/stream"

# Backend / model defaults — override with BACKEND= and MODEL= env vars.
# Examples:
#   BACKEND=gemini MODEL=gemini-3-flash-preview ./run_browser.sh   # cloud (default)
#   BACKEND=ollama MODEL=gemma4:26b ./run_browser.sh               # local Ollama on GPU 1
BACKEND=${BACKEND:-gemini}
MODEL=${MODEL:-gemini-3-flash-preview}
MAX_STEPS=${MAX_STEPS:-5000}
echo "Backend: $BACKEND   Model: $MODEL   Max steps: $MAX_STEPS"

# WebGL2 (required by Unity 2021+) doesn't work in Playwright's headless Chromium
# even with all the GPU flags — the headless build strips real GPU support.
# Solution: launch a virtual X display via Xvfb and run Playwright in headed mode.
# `xvfb-run -a` picks an unused display number and exports DISPLAY for the child.
if ! command -v xvfb-run >/dev/null 2>&1; then
  echo "ERROR: xvfb-run not found. Install with: sudo apt install xvfb" >&2
  exit 1
fi

xvfb-run -a -s "-screen 0 1280x800x24" \
  .venv/bin/python run.py \
    --game browser \
    --game-url "https://ravernt.itch.io/folder-dungeon" \
    --max-steps "$MAX_STEPS" \
    --port "$PORT" \
    --headed \
    --backend "$BACKEND" \
    --model-name "$MODEL" \
    2>&1 | tee "$LOG_FILE"
