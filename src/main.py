import logging

from config.settings import TELEGRAM_TOKEN, OPENAI_API_KEY
from src.bot.telegram_bot import start_bot

logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)  # silence per-poll request logs

if __name__ == "__main__":
    missing = [name for name, val in [("TELEGRAM_TOKEN", TELEGRAM_TOKEN), ("OPENAI_API_KEY", OPENAI_API_KEY)] if not val]
    if missing:
        raise SystemExit(f"Missing in .env: {', '.join(missing)} (see .env.example)")
    try:
        start_bot()
    except KeyboardInterrupt:
        print("\n--- Sentinel Orchestrator Offline ---")
