"""HTTP API: web chat (SSE streaming), underwriter review, Telegram webhook, health, metrics."""
import asyncio
import hmac
import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from .. import metrics
from ..bootstrap import Runtime, build_runtime
from ..config import Settings, load_settings
from ..logging_setup import configure_logging
from ..scoring.model import get_credit_model
from ..service import ReviewError
from ..telegram import TelegramAdapter
from ..tenants import TENANTS

logger = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"

HTTP_REQUESTS = Counter("sentinel_http_requests_total", "HTTP requests", ["method", "route", "status"])
HTTP_LATENCY = Histogram("sentinel_http_request_seconds", "HTTP time to response headers", ["method", "route"])


class MessageIn(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    lender: str | None = Field(None, description="Lender id; fixed on the conversation's first message")
    user_id: str | None = Field(None, max_length=100, description="Stable anonymous id from the client")
    client_message_id: str | None = Field(None, max_length=100, description="Idempotency key for retries")


class ReviewIn(BaseModel):
    decision: Literal["approved", "declined"]
    reviewer: str = Field(min_length=1, max_length=60)
    note: str | None = Field(None, max_length=500)


class FeedbackIn(BaseModel):
    trace_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    rating: Literal[1, -1]
    comment: str | None = Field(None, max_length=500)


class ScoreIn(BaseModel):
    person_age: int = Field(ge=18, le=100)
    person_income: float = Field(gt=0)
    person_emp_length: float = Field(ge=0)
    person_home_ownership: Literal["RENT", "OWN", "MORTGAGE", "OTHER"]
    loan_amnt: float = Field(gt=0)
    loan_int_rate: float = Field(ge=0, le=100)
    loan_intent: Literal["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
    loan_grade: Literal["A", "B", "C", "D", "E", "F", "G"]
    cb_person_cred_hist_length: int = Field(ge=0)
    cb_person_default_on_file: Literal["Y", "N"]


def create_app(settings: Settings | None = None, runtime_factory: Any = build_runtime) -> FastAPI:
    settings = settings or load_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        get_credit_model()  # load the model at startup, not on the first request
        async with runtime_factory(settings) as runtime:
            app.state.runtime = runtime
            app.state.telegram = None
            poller = None
            if settings.telegram_token and settings.telegram_mode != "off":
                app.state.telegram = TelegramAdapter(
                    settings.telegram_token, runtime.service, settings.allowed_chat_ids)
                if settings.telegram_mode == "polling":
                    poller = asyncio.create_task(app.state.telegram.poll_forever())
            logger.info("Sentinel API started", extra={"fields": {
                "models": settings.model_label, "checkpointer": settings.checkpointer,
                "telegram": settings.telegram_mode if settings.telegram_token else "off",
                "langfuse": settings.langfuse_enabled}})
            yield
            if poller:
                poller.cancel()
                with suppress(asyncio.CancelledError):
                    await poller
            if app.state.telegram:
                await app.state.telegram.close()

    app = FastAPI(title="Sentinel Orchestrator", version="1.0.0", lifespan=lifespan,
                  description="Conversational loan underwriting: LangGraph agent, RAG, human review.")
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    def runtime(request: Request) -> Runtime:
        return request.app.state.runtime

    @app.middleware("http")
    async def observe(request: Request, call_next: Any) -> Response:
        started = time.perf_counter()
        request_id = request.headers.get("x-request-id") or uuid4().hex
        response = await call_next(request)
        route = getattr(request.scope.get("route"), "path", "unmatched")
        HTTP_REQUESTS.labels(request.method, route, response.status_code).inc()
        HTTP_LATENCY.labels(request.method, route).observe(time.perf_counter() - started)
        response.headers["x-request-id"] = request_id
        return response

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    # --- Operations ---------------------------------------------------------------------------------

    @app.get("/healthz", tags=["ops"])
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz", tags=["ops"])
    async def readyz(request: Request) -> dict[str, Any]:
        rt = runtime(request)
        await rt.service.graph.aget_state({"configurable": {"thread_id": "readiness-probe"}})
        return {"status": "ready", "models": settings.model_label, "checkpointer": settings.checkpointer,
                "langfuse": rt.tracer.enabled, "policy_chunks": len(rt.retriever.chunks)}

    @app.get("/metrics", include_in_schema=False)
    async def prometheus() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/system", tags=["ops"])
    async def system(request: Request) -> dict[str, Any]:
        """Live service numbers for the UI's System panel."""
        rt = runtime(request)
        if settings.metrics_backend == "cloudwatch":
            from ..aws import cloudwatch_metrics
            numbers = await asyncio.to_thread(cloudwatch_metrics().snapshot)
        else:
            numbers = {**metrics.snapshot(), "window": "this process, since start"}
        return {**numbers, "models": rt.llm.model_names, "environment": settings.environment,
                "langfuse": rt.tracer.enabled, "max_concurrent_turns": settings.max_concurrent_turns}

    # --- Chat ---------------------------------------------------------------------------------------

    @app.get("/v1/lenders", tags=["chat"])
    async def lenders() -> list[dict[str, Any]]:
        return [{"id": t.id, "name": t.name, "approve_below": t.approve_below, "decline_above": t.decline_above,
                 "auto_limit": t.auto_limit, "max_loan_to_income": t.max_loan_to_income} for t in TENANTS.values()]

    @app.post("/v1/conversations", status_code=201, tags=["chat"])
    async def create_conversation() -> dict[str, str]:
        return {"conversation_id": f"web-{uuid4().hex}"}

    @app.get("/v1/conversations/{conversation_id}", tags=["chat"])
    async def get_conversation(conversation_id: str, request: Request) -> dict[str, Any]:
        return await runtime(request).service.get_state(conversation_id)

    @app.post("/v1/conversations/{conversation_id}/messages", tags=["chat"])
    async def send_message(conversation_id: str, body: MessageIn, request: Request,
                           accept: str = Header("text/event-stream")) -> Any:
        """Streams the reply as server-sent events, or returns JSON with `Accept: application/json`."""
        service = runtime(request).service
        if body.client_message_id and not service.seen_messages.check_and_add(f"web-{body.client_message_id}"):
            raise HTTPException(409, "Duplicate client_message_id")
        user_id = f"web-{body.user_id or (request.client.host if request.client else 'anon')}"

        if "application/json" in accept:
            result = await service.run_turn(conversation_id, user_id, body.text, "web", body.lender)
            return asdict(result)

        async def sse() -> AsyncIterator[str]:
            async for event in service.stream_turn(conversation_id, user_id, body.text, "web", body.lender):
                yield f"event: {event['type']}\ndata: {json.dumps(event, default=str)}\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/v1/feedback", status_code=202, tags=["chat"])
    async def feedback(body: FeedbackIn, request: Request) -> dict[str, str]:
        runtime(request).service.record_feedback(body.trace_id, body.rating, body.comment)
        return {"status": "recorded"}

    # --- Underwriting -------------------------------------------------------------------------------

    @app.get("/v1/reviews", tags=["underwriting"])
    async def reviews(request: Request, lender: str | None = None) -> list[dict[str, Any]]:
        return [asdict(r) for r in await runtime(request).service.list_reviews(lender)]

    @app.post("/v1/reviews/{application_id}", tags=["underwriting"])
    async def submit_review(application_id: str, body: ReviewIn, request: Request) -> dict[str, Any]:
        try:
            result = await runtime(request).service.submit_review(application_id, body.decision, body.reviewer,
                                                                  body.note)
        except ReviewError as e:
            raise HTTPException(409, str(e)) from e
        return asdict(result)

    @app.get("/v1/applications", tags=["underwriting"])
    async def applications(request: Request, lender: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        """Audit log of decisions: who decided, with which model and prompt versions, and the trace."""
        rows = await runtime(request).service.store.list(tenant_id=lender, limit=min(limit, 100))
        return [asdict(r) for r in rows]

    @app.post("/v1/score", tags=["underwriting"])
    async def score(body: ScoreIn) -> dict[str, Any]:
        """Direct access to the credit model, bypassing the conversation."""
        application = body.model_dump()
        application["loan_percent_income"] = round(body.loan_amnt / body.person_income, 2)
        return await asyncio.to_thread(get_credit_model().predict, application)

    # --- Telegram -----------------------------------------------------------------------------------

    @app.post("/telegram/webhook", include_in_schema=False)
    async def telegram_webhook(request: Request,
                               x_telegram_bot_api_secret_token: str = Header("")) -> dict[str, bool]:
        adapter: TelegramAdapter | None = request.app.state.telegram
        secret = settings.telegram_webhook_secret
        if adapter is None or not secret or not hmac.compare_digest(x_telegram_bot_api_secret_token, secret):
            raise HTTPException(403)
        try:
            await adapter.handle_update(await request.json())
        except Exception:
            # Still 200: a non-2xx makes Telegram retry the same update over and over
            logger.exception("Telegram webhook update failed")
        return {"ok": True}

    return app
