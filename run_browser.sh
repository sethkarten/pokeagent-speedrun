#!/bin/bash
set -a && source .env && set +a

# Create timestamped log file
LOG_DIR="run_data/browser_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(date +%Y%m%d_%H%M%S).log"

echo "Logging to $LOG_FILE"
echo "Watch live at http://localhost:8000/stream"

.venv/bin/python run.py \
  --game browser \
  --game-url "https://ravernt.itch.io/folder-dungeon" \
  --max-steps 50 \
  2>&1 | tee "$LOG_FILE"
