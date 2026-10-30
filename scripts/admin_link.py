"""Print the private admin link (chat URL + /admin#<token>). Bookmark it; don't share it.

    AWS_PROFILE=default uv run python scripts/admin_link.py
"""
import json
from pathlib import Path

import boto3

outputs = json.loads((Path(__file__).resolve().parents[1] / "cdk-outputs.json").read_text())["Sentinel"]
secret = boto3.client("secretsmanager", region_name="us-east-1").get_secret_value(SecretId=outputs["SecretArn"])
print(f"{outputs['Url'].rstrip('/')}/admin#{json.loads(secret['SecretString'])['ADMIN_TOKEN']}")
