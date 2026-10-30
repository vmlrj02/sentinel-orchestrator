"""Point Telegram at the deployed API (webhook mode). The local polling bot must be stopped first:
Telegram delivers updates to a webhook or to getUpdates, never both.

    AWS_PROFILE=default uv run python scripts/set_webhook.py
"""
import json
from pathlib import Path

import boto3
import httpx

outputs = json.loads((Path(__file__).resolve().parents[1] / "cdk-outputs.json").read_text())["Sentinel"]
secret = json.loads(boto3.client("secretsmanager", region_name="us-east-1")
                    .get_secret_value(SecretId=outputs["SecretArn"])["SecretString"])
api = f"https://api.telegram.org/bot{secret['TELEGRAM_TOKEN']}"
url = f"{outputs['Url'].rstrip('/')}/telegram/webhook"
result = httpx.post(f"{api}/setWebhook", json={
    "url": url, "secret_token": secret["TELEGRAM_WEBHOOK_SECRET"], "drop_pending_updates": True,
    "allowed_updates": ["message", "callback_query"], "max_connections": 10}).json()
print("setWebhook:", result.get("description"))
info = httpx.get(f"{api}/getWebhookInfo").json()["result"]
print("webhook:", info.get("url"), "| pending:", info.get("pending_update_count"), "| last error:",
      info.get("last_error_message", "none"))
