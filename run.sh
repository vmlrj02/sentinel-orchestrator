#!/usr/bin/env bash
# Starts the scoring API and the Telegram bot together. Ctrl+C stops both.
set -euo pipefail
cd "$(dirname "$0")"
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"  # launchd starts with a bare PATH

uv run --frozen uvicorn scoring.app:app --host 127.0.0.1 --port 8000 --log-level warning &
SCORER_PID=$!
trap 'kill $SCORER_PID 2>/dev/null || true' EXIT INT TERM

for _ in $(seq 1 30); do
  curl -sf http://127.0.0.1:8000/health >/dev/null && break
  sleep 1
done
curl -sf http://127.0.0.1:8000/health >/dev/null || { echo "Scoring API failed to start" >&2; exit 1; }
echo "Scoring API up on :8000"

uv run --frozen python -m src.main
