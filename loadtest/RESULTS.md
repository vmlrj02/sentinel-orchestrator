# Load test results

`python -m loadtest.run` drives N concurrent virtual applicants through the real HTTP + SSE API. Each
completes a full application (4 detail messages, "yes", one follow-up), so every turn exercises the
per-conversation lock, the checkpointer, the credit model, policy rules and streaming.

The server uses the **fake LLM with 800 ms latency per call**, so the numbers measure Sentinel itself,
not OpenAI's rate limits. One uvicorn process on an Apple-silicon laptop, SQLite checkpointer.

## Results (2026-10-30)

| users | turns | wall s | turns/s | p50 s | p95 s | p99 s | first event p95 s | scored | errors |
|---|---|---|---|---|---|---|---|---|---|
| 50 | 300 | 6.3 | 47.6 | 0.85 | 1.70 | 1.76 | 1.14 | 50 | 0 |
| 200 | 1200 | 28.6 | 41.9 | 3.48 | 7.09 | 8.06 | 7.08 | 200 | 0 |
| 500 | 3000 | 65.7 | 45.7 | 8.57 | 15.09 | 16.57 | 14.86 | 500 | 0 |

- At 50 users the median turn takes 0.85 s, of which 0.80 s is the simulated LLM: the service adds
  about 50 ms (locking, checkpointing, scoring, streaming).
- No errors and every application scored at 500 concurrent users: under overload the service queues
  and slows down; it does not fail or corrupt state.

## Where the ceiling comes from

Throughput flattens at about 45 turns/s per process. Investigating it:

| hypothesis | experiment | result |
|---|---|---|
| The concurrency cap (`MAX_CONCURRENT_TURNS=50`) | rerun 500 users with the cap at 500 | 45 to 54 turns/s: not the main limit |
| SQLite checkpoint writes on one connection | rerun with the in-memory checkpointer | unchanged: not the limit |
| The load generator | sample CPU during the run | client 8%, **server 99.8%** |
| Server CPU | server-side latency from `/metrics`; queue wait | server p50 2 s / p95 3 s vs client 5 s; queue wait ~0 |

**The server process is CPU-bound on one core**: about 10 ms of CPU per turn inside the agent
(LangGraph runtime and checkpoint serialisation, spread across many small calls; profiled with
cProfile, no single hotspot) plus about the same again in the HTTP/SSE layer. Python runs one core per
process, so the answer is to scale **out**, not up:

- State lives outside the process (checkpointer + application store), so any instance can serve any
  turn. Locally that means `uvicorn --workers N` behind a shared database; on AWS, Lambda runs one
  request per execution environment and adds environments as traffic grows.
- The per-conversation lock is in-process. With several instances, two messages in the same chat can
  land on different instances, so the deployed version needs a cross-instance guard (a DynamoDB
  conditional write per conversation); see `docs/decisions/005-scaling.md`.
- In real traffic the LLM provider's rate limits arrive long before 45 turns/s per instance; that is
  what `MAX_CONCURRENT_TURNS` protects.

Cheap wins not yet taken: replace the `@app.middleware("http")` metrics middleware (Starlette's
`BaseHTTPMiddleware`, known to add overhead to streaming responses) with a pure ASGI middleware, and
trim stored messages so checkpoints stay small in long conversations.

## Reproduce

```bash
LLM_PROVIDER=fake FAKE_LLM_LATENCY_MS=800 RATE_LIMIT_PER_MINUTE=100000 TELEGRAM_MODE=off \
  uv run uvicorn sentinel.api.app:create_app --factory --port 8011
ulimit -n 4096
uv run python -m loadtest.run --url http://127.0.0.1:8011 --users 50 200 500
```
