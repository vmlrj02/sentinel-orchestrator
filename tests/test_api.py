import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient

from sentinel.agent.llm import FakeAgentLLM
from sentinel.api.app import create_app
from sentinel.bootstrap import build_runtime
from tests.conftest import APPLICATION_TURNS, REFERRED_TURNS, retriever


@pytest.fixture
def client(settings):
    @asynccontextmanager
    async def runtime(s) -> AsyncIterator:
        async with build_runtime(s, llm=FakeAgentLLM(), retriever=retriever()) as rt:
            yield rt

    with TestClient(create_app(settings, runtime_factory=runtime)) as c:
        yield c


def sse_events(text: str) -> list[dict]:
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]


def test_health_and_metrics(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").json()["status"] == "ready"
    assert "sentinel_turns_total" in client.get("/metrics").text


def test_web_conversation_over_sse(client):
    cid = client.post("/v1/conversations").json()["conversation_id"]
    for text in [*APPLICATION_TURNS, "yes"]:
        response = client.post(f"/v1/conversations/{cid}/messages", json={"text": text, "user_id": "t"})
        assert response.headers["content-type"].startswith("text/event-stream")
    events = sse_events(response.text)
    assert events[-1]["type"] == "done"
    assert any(e["type"] == "assessment" for e in events)
    assert client.get(f"/v1/conversations/{cid}").json()["assessment"] is not None


def test_json_mode_and_idempotency(client):
    body = {"text": "person_age=30", "client_message_id": "m-1"}
    first = client.post("/v1/conversations/c/messages", json=body, headers={"accept": "application/json"})
    assert first.status_code == 200 and first.json()["trace_id"]
    assert client.post("/v1/conversations/c/messages", json=body).status_code == 409


def test_direct_score_validates_input(client):
    good = {"person_age": 30, "person_income": 60000, "person_emp_length": 5, "person_home_ownership": "RENT",
            "loan_amnt": 10000, "loan_int_rate": 11.5, "loan_intent": "EDUCATION", "loan_grade": "B",
            "cb_person_cred_hist_length": 6, "cb_person_default_on_file": "N"}
    assert client.post("/v1/score", json=good).json()["decision"] in ("approved", "declined")
    assert client.post("/v1/score", json={**good, "loan_grade": "Z"}).status_code == 422


def test_feedback_and_webhook_auth(client):
    assert client.post("/v1/feedback", json={"trace_id": "a" * 32, "rating": 1}).status_code == 202
    assert client.post("/telegram/webhook", json={}).status_code == 403  # no secret / telegram off


def test_underwriter_review_over_http(client):
    cid = client.post("/v1/conversations").json()["conversation_id"]
    for text in [*REFERRED_TURNS, "yes"]:
        client.post(f"/v1/conversations/{cid}/messages", json={"text": text, "lender": "aster"})
    assert client.get(f"/v1/conversations/{cid}").json()["status"] == "under_review"

    pending = client.get("/v1/reviews", params={"lender": "aster"}).json()
    app_id = next(r["application_id"] for r in pending if r["conversation_id"] == cid)
    decided = client.post(f"/v1/reviews/{app_id}", json={"decision": "approved", "reviewer": "maya"})
    assert decided.status_code == 200 and decided.json()["explained"]
    assert client.post(f"/v1/reviews/{app_id}", json={"decision": "approved", "reviewer": "maya"}).status_code == 409

    state = client.get(f"/v1/conversations/{cid}").json()
    assert state["status"] == "decided" and state["final_decision"]["decided_by"] == "underwriter:maya"
    audit = next(r for r in client.get("/v1/applications").json() if r["application_id"] == app_id)
    assert audit["status"] == "approved" and audit["model_version"].startswith("logreg-")


def test_lenders_and_system_panel(client):
    assert {t["id"] for t in client.get("/v1/lenders").json()} == {"aster", "kestrel"}
    client.post("/v1/conversations/c/messages", json={"text": "person_age=30"})
    system = client.get("/v1/system").json()
    assert system["turns"]["total"] >= 1 and "p95" in system["latency_s"]
    assert client.post("/v1/conversations/c2/messages", json={"text": "hi", "lender": "nope"},
                       headers={"accept": "application/json"}).json()["error"] == "unknown_lender"
