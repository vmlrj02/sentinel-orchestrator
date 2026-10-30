import dataclasses
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient

from sentinel.agent.llm import FakeAgentLLM
from sentinel.api.app import create_app
from sentinel.bootstrap import build_runtime
from tests.conftest import APPLICATION_TURNS, REFERRED_TURNS, retriever

ADMIN = {"authorization": "Bearer test-admin-token"}


@pytest.fixture
def client(settings):
    @asynccontextmanager
    async def runtime(s) -> AsyncIterator:
        async with build_runtime(s, llm=FakeAgentLLM(), retriever=retriever()) as rt:
            yield rt

    app = create_app(dataclasses.replace(settings, admin_token="test-admin-token"), runtime_factory=runtime)
    with TestClient(app) as c:
        yield c


def sse_events(text: str) -> list[dict]:
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]


def new_cid(client) -> str:
    return client.post("/v1/conversations").json()["conversation_id"]


def test_health_and_metrics(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz", headers=ADMIN).json()["status"] == "ready"
    assert "sentinel_turns_total" in client.get("/metrics", headers=ADMIN).text


def test_internal_endpoints_are_hidden_from_the_public(client):
    for method, path in [("get", "/readyz"), ("get", "/metrics"), ("get", "/v1/system"), ("get", "/v1/lenders"),
                         ("get", "/v1/reviews"), ("get", "/v1/applications"), ("get", "/v1/evals"),
                         ("post", "/v1/score"), ("post", "/v1/reviews/app-1")]:
        for headers in ({}, {"authorization": "Bearer wrong"}):
            assert getattr(client, method)(path, headers=headers).status_code == 404, path


def test_admin_api_is_disabled_without_a_configured_token(settings):
    app = create_app(dataclasses.replace(settings, admin_token=None))
    with TestClient(app) as c:
        assert c.get("/v1/system", headers={"authorization": "Bearer "}).status_code == 404


def test_public_chat_cannot_reach_other_channels_conversations(client):
    assert client.get("/v1/conversations/tg-12345").status_code == 404
    assert client.post("/v1/conversations/tg-12345/messages", json={"text": "hi"}).status_code == 404
    assert client.get("/v1/conversations/web-short").status_code == 404


def test_web_conversation_over_sse_hides_internal_model_data(client):
    cid = new_cid(client)
    for text in [*APPLICATION_TURNS, "yes"]:
        response = client.post(f"/v1/conversations/{cid}/messages", json={"text": text, "user_id": "t"})
        assert response.headers["content-type"].startswith("text/event-stream")
    events = sse_events(response.text)
    assert events[-1]["type"] == "done"
    assert {"type": "assessment", "policy_decision": "approved"} in events  # outcome only
    assert "probability" not in response.text and "explanations" not in response.text
    state = client.get(f"/v1/conversations/{cid}").json()
    assert set(state) == {"conversation_id", "status", "messages"} and state["status"] == "decided"


def test_json_mode_and_idempotency(client):
    cid = new_cid(client)
    body = {"text": "person_age=30", "client_message_id": "m-1"}
    first = client.post(f"/v1/conversations/{cid}/messages", json=body, headers={"accept": "application/json"})
    assert first.status_code == 200 and first.json()["trace_id"]
    assert client.post(f"/v1/conversations/{cid}/messages", json=body).status_code == 409


def test_direct_score_validates_input(client):
    good = {"person_age": 30, "person_income": 60000, "person_emp_length": 5, "person_home_ownership": "RENT",
            "loan_amnt": 10000, "loan_int_rate": 11.5, "loan_intent": "EDUCATION", "loan_grade": "B",
            "cb_person_cred_hist_length": 6, "cb_person_default_on_file": "N"}
    assert client.post("/v1/score", json=good, headers=ADMIN).json()["decision"] in ("approved", "declined")
    assert client.post("/v1/score", json={**good, "loan_grade": "Z"}, headers=ADMIN).status_code == 422


def test_feedback_and_webhook_auth(client):
    assert client.post("/v1/feedback", json={"trace_id": "a" * 32, "rating": 1}).status_code == 202
    assert client.post("/telegram/webhook", json={}).status_code == 403  # no secret / telegram off


def test_underwriter_review_over_http(client):
    cid = new_cid(client)
    for text in [*REFERRED_TURNS, "yes"]:
        client.post(f"/v1/conversations/{cid}/messages", json={"text": text, "lender": "aster"})
    assert client.get(f"/v1/conversations/{cid}").json()["status"] == "under_review"

    pending = client.get("/v1/reviews", params={"lender": "aster"}, headers=ADMIN).json()
    app_id = next(r["application_id"] for r in pending if r["conversation_id"] == cid)
    review = {"decision": "approved", "reviewer": "maya"}
    decided = client.post(f"/v1/reviews/{app_id}", json=review, headers=ADMIN)
    assert decided.status_code == 200 and decided.json()["explained"]
    assert client.post(f"/v1/reviews/{app_id}", json=review, headers=ADMIN).status_code == 409

    assert client.get(f"/v1/conversations/{cid}").json()["status"] == "decided"
    audit = next(r for r in client.get("/v1/applications", headers=ADMIN).json() if r["application_id"] == app_id)
    assert audit["status"] == "approved" and audit["decided_by"] == "underwriter:maya"
    assert audit["model_version"].startswith("logreg-")


def test_admin_page_and_panels(client):
    page = client.get("/admin")
    assert page.status_code == 200 and "noindex" in page.headers["x-robots-tag"]
    assert {t["id"] for t in client.get("/v1/lenders", headers=ADMIN).json()} == {"aster", "kestrel"}
    client.post(f"/v1/conversations/{new_cid(client)}/messages", json={"text": "person_age=30"})
    system = client.get("/v1/system", headers=ADMIN).json()
    assert system["turns"]["total"] >= 1 and "p95" in system["latency_s"]
    assert "conversations" in client.get("/v1/evals", headers=ADMIN).json()
    bad = client.post(f"/v1/conversations/{new_cid(client)}/messages", json={"text": "hi", "lender": "nope"},
                      headers={"accept": "application/json"})
    assert bad.json()["error"] == "unknown_lender"
