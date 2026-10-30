"""Copy API keys from .env into the deployed stack's Secrets Manager secret (values are never printed).

    AWS_PROFILE=default uv run python scripts/put_secrets.py
"""
import json
import secrets
from pathlib import Path

import boto3
from dotenv import dotenv_values

KEYS = ["OPENAI_API_KEY", "TELEGRAM_TOKEN", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL",
        "ALLOWED_CHAT_IDS"]

root = Path(__file__).resolve().parents[1]
arn = json.loads((root / "cdk-outputs.json").read_text())["Sentinel"]["SecretArn"]
env = {k.strip(): (v or "").strip() for k, v in dotenv_values(root / ".env").items()}
client = boto3.client("secretsmanager", region_name="us-east-1")

try:
    current = json.loads(client.get_secret_value(SecretId=arn)["SecretString"])
except Exception:
    current = {}
values = {k: env[k] for k in KEYS if env.get(k)}
# Keep the webhook secret stable across runs, so the registered Telegram webhook keeps working
values["TELEGRAM_WEBHOOK_SECRET"] = current.get("TELEGRAM_WEBHOOK_SECRET") or secrets.token_urlsafe(32)
values["ADMIN_TOKEN"] = current.get("ADMIN_TOKEN") or secrets.token_urlsafe(32)  # opens /admin#<token>
client.put_secret_value(SecretId=arn, SecretString=json.dumps(values))
print("Stored:", ", ".join(sorted(values)))
