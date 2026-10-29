"""HTTP API: web chat (SSE streaming), Telegram webhook, health, metrics."""
import asyncio
import hmac
import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from ..bootstrap import Runtime, build_runtime
from ..config import Settings, load_settings
from ..logging_setup import configure_logging
from ..scoring.model import get_credit_model
from ..telegram import TelegramAdapter

logger = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"

HTTP_REQUESTS = Counter("sentinel_http_requests_total", "HTTP requests", ["method", "route", "status"])
HTTP_LATENCY = Histogram("sentinel_http_request_seconds", "HTTP time to response headers", ["method", "route"])


class MessageIn(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    user_id: str | None = Field(None, max_length=100, description="Stable anonymous id from the client")
    client_message_id: str | None = Field(None, max_length=100, description="Idempotency key for retries")


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
                "model": settings.model_label, "checkpointer": settings.checkpointer,
                "telegram": settings.telegram_mode if settings.telegram_token else "off",
                "langfuse": settings.langfuse_enabled}})
            yield
            if poller:
                poller.cancel()
                with suppress(asyncio.CancelledError):
                    await poller
            if app.state.telegram:
                await app.state.telegram.close()

    app = FastAPI(title="Sentinel Orchestrator", version="1.0.0", lifespan=lifespan)

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

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(request: Request) -> dict[str, Any]:
        rt = runtime(request)
        await rt.service.graph.aget_state({"configurable": {"thread_id": "readiness-probe"}})
        return {"status": "ready", "model": settings.model_label, "checkpointer": settings.checkpointer,
                "langfuse": rt.tracer.enabled}

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/v1/conversations", status_code=201)
    async def create_conversation() -> dict[str, str]:
        return {"conversation_id": f"web-{uuid4().hex}"}

    @app.get("/v1/conversations/{conversation_id}")
    async def get_conversation(conversation_id: str, request: Request) -> dict[str, Any]:
        return await runtime(request).service.get_state(conversation_id)

    @app.post("/v1/conversations/{conversation_id}/messages")
    async def send_message(conversation_id: str, body: MessageIn, request: Request,
                           accept: str = Header("text/event-stream")) -> Any:
        service = runtime(request).service
        if body.client_message_id and not service.seen_messages.check_and_add(f"web-{body.client_message_id}"):
            raise HTTPException(409, "Duplicate client_message_id")
        user_id = f"web-{body.user_id or (request.client.host if request.client else 'anon')}"

        if "application/json" in accept:
            result = await service.run_turn(conversation_id, user_id, body.text, channel="web")
            return result.__dict__

        async def sse() -> AsyncIterator[str]:
            async for event in service.stream_turn(conversation_id, user_id, body.text, channel="web"):
                yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/v1/feedback", status_code=202)
    async def feedback(body: FeedbackIn, request: Request) -> dict[str, str]:
        runtime(request).service.record_feedback(body.trace_id, body.rating, body.comment)
        return {"status": "recorded"}

    @app.post("/v1/score")
    async def score(body: ScoreIn) -> dict[str, Any]:
        """Direct access to the credit model, bypassing the conversation."""
        application = body.model_dump()
        application["loan_percent_income"] = round(body.loan_amnt / body.person_income, 2)
        return await asyncio.to_thread(get_credit_model().predict, application)

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
