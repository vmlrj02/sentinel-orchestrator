"""End-to-end smoke test against a deployed URL: policy question, referral, underwriter decision.

    uv run python scripts/smoke_deployed.py https://<function-url>
"""
import json
import sys
import time

import httpx

url = sys.argv[1].rstrip("/")
client = httpx.Client(timeout=120)


def turn(cid: str, text: str, lender: str = "aster") -> list[dict]:
    started, events = time.perf_counter(), []
    with client.stream("POST", f"{url}/v1/conversations/{cid}/messages",
                       json={"text": text, "lender": lender, "user_id": "smoke"}) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    kinds = sorted({e["type"] for e in events})
    reply = next((e["text"] for e in reversed(events) if e["type"] == "message"), "")
    print(f"  [{time.perf_counter() - started:.1f}s {kinds}] {text[:60]!r}\n    -> {reply[:160]!r}")
    return events


cid = client.post(f"{url}/v1/conversations").json()["conversation_id"]
turn(cid, "Before I apply, is there an origination fee?")
turn(cid, "Asha Rao, 29, 55000 a year, 4 years in my job, renting. 18000 for medical treatment at 13%, grade C, "
          "5 years of credit history, no defaults.")
events = turn(cid, "yes")
assessment = next(e for e in events if e["type"] == "assessment")
print("  decision:", assessment["policy_decision"], assessment["model"]["probability"])

pending = [r for r in client.get(f"{url}/v1/reviews").json() if r["conversation_id"] == cid]
decided = client.post(f"{url}/v1/reviews/{pending[0]['application_id']}",
                      json={"decision": "approved", "reviewer": "smoke-test", "note": "payslips verified"}).json()
print("  underwriter ->", decided["messages"][-1][:160])
state = client.get(f"{url}/v1/conversations/{cid}").json()
print("  final state:", state["status"], state["final_decision"]["decided_by"])
time.sleep(5)
print("  system:", json.dumps({k: v for k, v in client.get(f"{url}/v1/system").json().items()
                                if k in ("window", "turns", "latency_s", "decisions")}))
