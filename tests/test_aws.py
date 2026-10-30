"""AWS adapters against moto's in-process AWS, so CI covers the Lambda backend without real AWS."""
import asyncio
import dataclasses
import json
import time

import boto3
import pytest
from moto import mock_aws

from sentinel.agent.llm import FakeAgentLLM
from sentinel.aws import DynamoApplicationStore, DynamoLocks, LockTimeout, emit_emf, load_secrets
from sentinel.bootstrap import build_runtime
from sentinel.store import ApplicationRecord
from tests.conftest import REFERRED_TURNS, retriever

REGION = "us-east-1"


@pytest.fixture
def aws(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", REGION)
    for var in ("AWS_PROFILE", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    with mock_aws():
        ddb = boto3.client("dynamodb", region_name=REGION)
        ddb.create_table(
            TableName="apps", BillingMode="PAY_PER_REQUEST",
            KeySchema=[{"AttributeName": "application_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": n, "AttributeType": "S"}
                                  for n in ("application_id", "status", "created_at", "kind")],
            GlobalSecondaryIndexes=[
                {"IndexName": "by_status", "Projection": {"ProjectionType": "ALL"},
                 "KeySchema": [{"AttributeName": "status", "KeyType": "HASH"},
                               {"AttributeName": "created_at", "KeyType": "RANGE"}]},
                {"IndexName": "recent", "Projection": {"ProjectionType": "ALL"},
                 "KeySchema": [{"AttributeName": "kind", "KeyType": "HASH"},
                               {"AttributeName": "created_at", "KeyType": "RANGE"}]}])
        ddb.create_table(TableName="locks", BillingMode="PAY_PER_REQUEST",
                         KeySchema=[{"AttributeName": "lock_id", "KeyType": "HASH"}],
                         AttributeDefinitions=[{"AttributeName": "lock_id", "AttributeType": "S"}])
        ddb.create_table(TableName="checkpoints", BillingMode="PAY_PER_REQUEST",
                         KeySchema=[{"AttributeName": "PK", "KeyType": "HASH"},
                                    {"AttributeName": "SK", "KeyType": "RANGE"}],
                         AttributeDefinitions=[{"AttributeName": "PK", "AttributeType": "S"},
                                               {"AttributeName": "SK", "AttributeType": "S"}])
        yield


def record(app_id: str, status: str, tenant: str, created: str) -> ApplicationRecord:
    return ApplicationRecord(application_id=app_id, conversation_id=f"c-{app_id}", tenant_id=tenant,
                             status=status, decided_by="policy", application={"loan_amnt": 1000.5},  # type: ignore[arg-type]
                             model_result={"probability": 0.4}, policy_reason="r", model_version="m",
                             prompt_versions={}, created_at=created)


async def test_application_store_queue_and_audit(aws):
    store = DynamoApplicationStore("apps")
    await store.upsert(record("a1", "pending_review", "aster", "2026-10-30T01:00:00"))
    await store.upsert(record("a2", "approved", "aster", "2026-10-30T02:00:00"))
    await store.upsert(record("a3", "pending_review", "kestrel", "2026-10-30T03:00:00"))

    assert [r.application_id for r in await store.list(status="pending_review")] == ["a3", "a1"]
    assert [r.application_id for r in await store.list(status="pending_review", tenant_id="aster")] == ["a1"]
    assert [r.application_id for r in await store.list()] == ["a3", "a2", "a1"]  # newest first
    fetched = await store.get("a2")
    assert fetched and fetched.application == {"loan_amnt": 1000.5}  # floats survive (stored as JSON)


async def test_lock_is_exclusive_across_holders(aws):
    a, b = DynamoLocks("locks", wait_s=0.3), DynamoLocks("locks", wait_s=0.3)  # two "instances"
    async with a.hold("chat-1"):
        with pytest.raises(LockTimeout):
            async with b.hold("chat-1"):
                pass
        async with b.hold("chat-2"):  # other conversations are unaffected
            pass
    async with b.hold("chat-1"):  # released
        pass


async def test_expired_lease_can_be_taken_over(aws):
    crashed = DynamoLocks("locks", lease_s=-1)  # a holder whose lease has already run out
    await asyncio.to_thread(crashed._acquire, "conversation#c", "dead-instance")
    async with DynamoLocks("locks", wait_s=0.3).hold("c"):
        pass


async def test_turns_in_one_chat_are_serialised_across_instances(aws):
    a, b = DynamoLocks("locks"), DynamoLocks("locks")
    events: list[str] = []

    async def work(locks: DynamoLocks, name: str) -> None:
        async with locks.hold("chat"):
            events.append(f"{name}-start")
            await asyncio.sleep(0.05)
            events.append(f"{name}-end")

    await asyncio.gather(work(a, "a"), work(b, "b"))
    assert events in (["a-start", "a-end", "b-start", "b-end"], ["b-start", "b-end", "a-start", "a-end"])


def test_secrets_are_loaded_without_overriding_env(aws, monkeypatch):
    arn = boto3.client("secretsmanager", region_name=REGION).create_secret(
        Name="sentinel", SecretString=json.dumps({"OPENAI_API_KEY": "from-secret", "TELEGRAM_TOKEN": "tg"}))["ARN"]
    monkeypatch.setenv("SECRETS_ARN", arn)
    monkeypatch.setenv("TELEGRAM_TOKEN", "already-set")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    load_secrets()
    import os
    assert os.environ["OPENAI_API_KEY"] == "from-secret" and os.environ["TELEGRAM_TOKEN"] == "already-set"


def test_emf_line_format(capsys):
    emit_emf({"Turns": (1, "Count"), "TurnLatency": (1234, "Milliseconds")}, {"Outcome": "ok"})
    line = json.loads(capsys.readouterr().out)
    assert line["_aws"]["CloudWatchMetrics"][0]["Namespace"] == "Sentinel"
    assert line["_aws"]["CloudWatchMetrics"][0]["Dimensions"] == [["Outcome"]]
    assert line["Turns"] == 1 and line["TurnLatency"] == 1234 and line["Outcome"] == "ok"
    assert abs(line["_aws"]["Timestamp"] / 1000 - time.time()) < 5


async def test_full_review_flow_on_dynamodb_backend(aws, settings):
    """The Lambda configuration end to end: DynamoDB checkpoints, store and locks."""
    s = dataclasses.replace(settings, checkpointer="dynamodb", checkpoint_table="checkpoints",
                            applications_table="apps", locks_table="locks")
    async with build_runtime(s, llm=FakeAgentLLM(), retriever=retriever()) as rt:
        for text in [*REFERRED_TURNS, "yes"]:
            result = await rt.service.run_turn("c", "u", text, "test", tenant_id="aster")
        assert result.assessment["policy_decision"] == "referred"
        assert (await rt.service.get_state("c"))["status"] == "under_review"

    # A different "instance" (fresh runtime) resumes the paused conversation from DynamoDB
    async with build_runtime(s, llm=FakeAgentLLM(), retriever=retriever()) as rt:
        [pending] = await rt.service.list_reviews("aster")
        decided = await rt.service.submit_review(pending.application_id, "approved", "maya")
        assert decided.explained
        state = await rt.service.get_state("c")
        assert state["status"] == "decided" and state["final_decision"]["decided_by"] == "underwriter:maya"
