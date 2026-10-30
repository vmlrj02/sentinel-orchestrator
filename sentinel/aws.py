"""AWS adapters, used when deployed on Lambda (CHECKPOINTER=dynamodb).

- load_secrets: API keys from Secrets Manager at cold start, never in Lambda env vars
- DynamoApplicationStore: audit records + review queue
- DynamoLocks: per-conversation lease lock shared by every Lambda instance
- emit_emf / CloudWatchMetrics: metrics via CloudWatch Embedded Metric Format, read back for the UI

boto3 is synchronous, so calls run in a worker thread to keep the event loop free.
"""
import asyncio
import json
import logging
import os
import sys
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from functools import cache
from typing import Any
from uuid import uuid4

import boto3
from botocore.exceptions import ClientError

from .store import ApplicationRecord, Status

logger = logging.getLogger(__name__)
NAMESPACE = "Sentinel"
FUNNEL_STAGES = ["started", "summary_shown", "confirmed", "handoff_requested"]
SLOT_EVENTS = ["filled_fast", "filled_llm", "reask", "invalid", "skipped", "declined"]
COMMAND_NAMES = ["start_over", "help", "summary", "human", "skip", "apply", "change", "sensitive_data_removed"]
SLOTS = ["person_age", "person_income", "person_emp_length", "person_home_ownership", "loan_amnt",
         "loan_int_rate", "loan_intent", "loan_grade", "cb_person_cred_hist_length", "cb_person_default_on_file",
         "confirm"]


def load_secrets() -> None:
    """Copy the JSON secret named by SECRETS_ARN into the environment (existing variables win)."""
    arn = os.getenv("SECRETS_ARN")
    if not arn:
        return
    secret = boto3.client("secretsmanager").get_secret_value(SecretId=arn)["SecretString"]
    for key, value in json.loads(secret).items():
        if value:
            os.environ.setdefault(key, str(value))


class DynamoApplicationStore:
    """One item per application. GSIs: `by_status` (status, created_at) for the review queue and
    `recent` (kind, created_at) for the audit log, so neither needs a table scan."""

    def __init__(self, table_name: str) -> None:
        self._table = boto3.resource("dynamodb").Table(table_name)

    async def upsert(self, record: ApplicationRecord) -> None:
        item = {"application_id": record.application_id, "tenant_id": record.tenant_id, "status": record.status,
                "created_at": record.created_at, "kind": "application", "record": json.dumps(asdict(record))}
        await asyncio.to_thread(self._table.put_item, Item=item)

    async def get(self, application_id: str) -> ApplicationRecord | None:
        response = await asyncio.to_thread(self._table.get_item, Key={"application_id": application_id})
        item = response.get("Item")
        return ApplicationRecord(**json.loads(item["record"])) if item else None

    async def list(self, status: Status | None = None, tenant_id: str | None = None,
                   limit: int = 50) -> list[ApplicationRecord]:
        if status:
            query = {"IndexName": "by_status", "KeyConditionExpression": "#s = :s",
                     "ExpressionAttributeNames": {"#s": "status"}, "ExpressionAttributeValues": {":s": status}}
        else:
            query = {"IndexName": "recent", "KeyConditionExpression": "kind = :k",
                     "ExpressionAttributeValues": {":k": "application"}}
        response = await asyncio.to_thread(self._table.query, ScanIndexForward=False, Limit=limit * 2, **query)
        records = [ApplicationRecord(**json.loads(i["record"])) for i in response["Items"]]
        return [r for r in records if tenant_id is None or r.tenant_id == tenant_id][:limit]


class LockTimeout(TimeoutError):
    pass


class DynamoLocks:
    """Lease lock per conversation: a conditional put that succeeds only if nobody holds an unexpired
    lease. The lease expires on its own, so a crashed instance cannot block a chat forever."""

    def __init__(self, table_name: str, lease_s: float = 90, wait_s: float = 30) -> None:
        self._table = boto3.resource("dynamodb").Table(table_name)
        self.lease_s, self.wait_s = lease_s, wait_s

    def _acquire(self, lock_id: str, owner: str) -> bool:
        now = time.time()
        try:
            self._table.put_item(
                Item={"lock_id": lock_id, "owner": owner, "expires_at": int(now + self.lease_s)},
                ConditionExpression="attribute_not_exists(lock_id) OR expires_at < :now",
                ExpressionAttributeValues={":now": int(now)},
            )
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                return False
            raise

    def _release(self, lock_id: str, owner: str) -> None:
        try:
            self._table.delete_item(Key={"lock_id": lock_id}, ConditionExpression="#o = :o",
                                    ExpressionAttributeNames={"#o": "owner"}, ExpressionAttributeValues={":o": owner})
        except ClientError as e:
            if e.response["Error"]["Code"] != "ConditionalCheckFailedException":  # lease expired, someone else holds it
                raise

    @asynccontextmanager
    async def hold(self, conversation_id: str) -> AsyncIterator[None]:
        lock_id, owner = f"conversation#{conversation_id}", uuid4().hex
        deadline, delay = time.monotonic() + self.wait_s, 0.05
        while not await asyncio.to_thread(self._acquire, lock_id, owner):
            if time.monotonic() > deadline:
                raise LockTimeout(f"conversation {conversation_id} is busy")
            await asyncio.sleep(delay)
            delay = min(delay * 2, 1.0)
        try:
            yield
        finally:
            await asyncio.to_thread(self._release, lock_id, owner)


def emit_emf(metrics: dict[str, tuple[float, str]], dimensions: dict[str, str] | None = None) -> None:
    """Write one CloudWatch Embedded Metric Format line to stdout; Lambda turns it into metrics with no
    API call on the request path. metrics = {name: (value, unit)}."""
    dimensions = dimensions or {}
    print(json.dumps({
        "_aws": {"Timestamp": int(time.time() * 1000), "CloudWatchMetrics": [{
            "Namespace": NAMESPACE,
            "Dimensions": [list(dimensions)] if dimensions else [[]],
            "Metrics": [{"Name": name, "Unit": unit} for name, (_, unit) in metrics.items()],
        }]},
        **dimensions, **{name: value for name, (value, _) in metrics.items()},
    }), file=sys.stdout, flush=True)


class CloudWatchMetrics:
    """Reads the EMF metrics back, aggregated across all Lambda instances, for the System panel."""

    def __init__(self, ttl_s: float = 30) -> None:
        self._cw = boto3.client("cloudwatch")
        self._ttl_s = ttl_s
        self._cached: tuple[float, dict[str, Any]] | None = None

    def snapshot(self, hours: int = 24) -> dict[str, Any]:
        if self._cached and time.monotonic() - self._cached[0] < self._ttl_s:
            return self._cached[1]
        end = datetime.now(UTC) + timedelta(minutes=1)  # end is exclusive: include the latest points
        period = hours * 3600

        def q(qid: str, name: str, stat: str, dims: dict[str, str] | None = None) -> dict[str, Any]:
            return {"Id": qid, "MetricStat": {"Metric": {
                "Namespace": NAMESPACE, "MetricName": name,
                "Dimensions": [{"Name": k, "Value": v} for k, v in (dims or {}).items()]},
                "Period": period, "Stat": stat}}

        queries = [
            q("turns_ok", "Turns", "Sum", {"Outcome": "ok"}), q("turns_error", "Turns", "Sum", {"Outcome": "error"}),
            q("turns_timeout", "Turns", "Sum", {"Outcome": "timeout"}),
            q("turns_limited", "Turns", "Sum", {"Outcome": "rate_limited"}),
            q("p50", "TurnLatency", "p50", {"Outcome": "ok"}), q("p95", "TurnLatency", "p95", {"Outcome": "ok"}),
            q("tokens", "LLMTokens", "Sum"), q("llm_calls", "LLMCalls", "Sum"), q("llm_errors", "LLMErrors", "Sum"),
            *(q(f"d_{d}", "Decisions", "Sum", {"Decision": d})
              for d in ("approved", "declined", "referred", "underwriter_approved", "underwriter_declined")),
            q("fb_up", "Feedback", "Sum", {"Rating": "up"}), q("fb_down", "Feedback", "Sum", {"Rating": "down"}),
            *(q(f"f_{s}", "Funnel", "Sum", {"Stage": s}) for s in FUNNEL_STAGES),
            *(q(f"c_{c}", "Commands", "Sum", {"Command": c}) for c in COMMAND_NAMES),
            *(q(f"s_{i}_{e}", "SlotEvents", "Sum", {"Slot": slot, "Event": e})
              for i, slot in enumerate(SLOTS) for e in SLOT_EVENTS),
        ]
        response = self._cw.get_metric_data(MetricDataQueries=queries, StartTime=end - timedelta(hours=hours),
                                            EndTime=end, ScanBy="TimestampDescending")
        v = {r["Id"]: (r["Values"][0] if r["Values"] else 0.0) for r in response["MetricDataResults"]}
        total = v["turns_ok"] + v["turns_error"] + v["turns_timeout"] + v["turns_limited"]
        snapshot = {
            "window": f"last {hours}h, all instances (CloudWatch)",
            "turns": {"total": total, "ok": v["turns_ok"], "error": v["turns_error"], "timeout": v["turns_timeout"]},
            "latency_s": {"p50": round(v["p50"] / 1000, 2) if v["p50"] else None,
                          "p95": round(v["p95"] / 1000, 2) if v["p95"] else None},
            "in_flight": None, "rate_limited": v["turns_limited"],
            "llm": {"calls": v["llm_calls"], "errors": v["llm_errors"], "tokens": {"total": v["tokens"]},
                    "tokens_per_turn": round(v["tokens"] / total) if total else None},
            "decisions": {k[2:]: n for k, n in v.items() if k.startswith("d_") and n},
            "feedback": {"up": v["fb_up"], "down": v["fb_down"]},
            "funnel": {s: v[f"f_{s}"] for s in FUNNEL_STAGES if v[f"f_{s}"]},
            "commands": {c: v[f"c_{c}"] for c in COMMAND_NAMES if v[f"c_{c}"]},
            "slots": {slot: {e: v[f"s_{i}_{e}"] for e in SLOT_EVENTS if v[f"s_{i}_{e}"]}
                      for i, slot in enumerate(SLOTS) if any(v[f"s_{i}_{e}"] for e in SLOT_EVENTS)},
        }
        self._cached = (time.monotonic(), snapshot)
        return snapshot


@cache
def cloudwatch_metrics() -> CloudWatchMetrics:
    return CloudWatchMetrics()
