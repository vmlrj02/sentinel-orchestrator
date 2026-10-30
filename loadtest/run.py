"""Concurrent-user load test over the real HTTP/SSE API.

Each virtual user completes a whole application (4 detail messages, "yes", one follow-up question),
so every turn exercises locking, checkpointing, scoring and streaming. Point it at a server started
with the fake LLM to measure the system itself rather than the LLM provider:

    LLM_PROVIDER=fake FAKE_LLM_LATENCY_MS=800 RATE_LIMIT_PER_MINUTE=100000 TELEGRAM_MODE=off \\
        uv run uvicorn sentinel.api.app:create_app --factory --port 8011
    uv run python -m loadtest.run --url http://127.0.0.1:8011 --users 50 200 500
"""
import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

TURNS = [
    "person_name=Load person_age=30",
    "person_income=60000 person_emp_length=5 person_home_ownership=RENT",
    "loan_amnt=10000 loan_int_rate=11.5 loan_intent=EDUCATION",
    "loan_grade=B cb_person_cred_hist_length=6 cb_person_default_on_file=N",
    "yes",
    "why was I approved?",
]


@dataclass
class Stats:
    turn_s: list[float] = field(default_factory=list)
    first_event_s: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    decisions: int = 0


async def turn(client: httpx.AsyncClient, url: str, cid: str, user: str, text: str, stats: Stats) -> None:
    started = time.perf_counter()
    first = None
    async with client.stream("POST", f"{url}/v1/conversations/{cid}/messages",
                             json={"text": text, "user_id": user, "lender": "kestrel"}) as response:
        if response.status_code != 200:
            stats.errors.append(f"http {response.status_code}")
            return
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            first = first or time.perf_counter()
            event = json.loads(line[6:])
            if event["type"] == "error":
                stats.errors.append(event["code"])
            elif event["type"] == "assessment":
                stats.decisions += 1
    stats.turn_s.append(time.perf_counter() - started)
    if first:
        stats.first_event_s.append(first - started)


async def user(client: httpx.AsyncClient, url: str, n: int, run: str, stats: Stats) -> None:
    cid = (await client.post(f"{url}/v1/conversations")).json()["conversation_id"]
    for text in TURNS:
        try:
            await turn(client, url, cid, f"load-{run}-{n}", text, stats)
        except Exception as e:  # count, don't crash the run
            stats.errors.append(type(e).__name__)


def q(values: list[float], quantile: float) -> float:
    return round(statistics.quantiles(values, n=100)[int(quantile * 100) - 1], 2) if len(values) > 1 else 0.0


async def run_level(url: str, users: int) -> dict[str, Any]:
    stats = Stats()
    limits = httpx.Limits(max_connections=users + 10, max_keepalive_connections=users + 10)
    async with httpx.AsyncClient(timeout=120, limits=limits) as client:
        started = time.perf_counter()
        await asyncio.gather(*(user(client, url, i, f"{users}-{int(started)}", stats) for i in range(users)))
        wall = time.perf_counter() - started
    turns = len(stats.turn_s)
    return {
        "users": users, "turns": turns, "wall_s": round(wall, 1), "turns_per_s": round(turns / wall, 1),
        "turn_p50_s": q(stats.turn_s, 0.5), "turn_p95_s": q(stats.turn_s, 0.95), "turn_p99_s": q(stats.turn_s, 0.99),
        "first_event_p95_s": q(stats.first_event_s, 0.95),
        "applications_scored": stats.decisions, "errors": len(stats.errors),
        "error_kinds": sorted(set(stats.errors)),
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8011")
    parser.add_argument("--users", type=int, nargs="+", default=[50, 200, 500])
    args = parser.parse_args()
    async with httpx.AsyncClient() as client:
        ready = (await client.get(f"{args.url}/readyz")).json()
    print(f"Target: {args.url} ({ready['models']}, checkpointer {ready['checkpointer']})")
    print("| users | turns | wall s | turns/s | p50 s | p95 s | p99 s | first event p95 s | scored | errors |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for users in args.users:
        r = await run_level(args.url, users)
        print(f"| {r['users']} | {r['turns']} | {r['wall_s']} | {r['turns_per_s']} | {r['turn_p50_s']} | "
              f"{r['turn_p95_s']} | {r['turn_p99_s']} | {r['first_event_p95_s']} | {r['applications_scored']} | "
              f"{r['errors']} {r['error_kinds'] or ''} |")


if __name__ == "__main__":
    asyncio.run(main())
