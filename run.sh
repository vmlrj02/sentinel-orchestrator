#!/usr/bin/env bash
# Local run: API + web UI on :8000, Telegram via long polling. Ctrl+C stops it.
set -euo pipefail
cd "$(dirname "$0")"
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"  # launchd starts with a bare PATH
exec uv run --frozen uvicorn sentinel.api.app:create_app --factory --host 127.0.0.1 --port "${PORT:-8000}"
