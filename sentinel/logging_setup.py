"""JSON logs with request context, so every line can be joined to its Langfuse trace."""
import contextvars
import json
import logging
import sys
from datetime import UTC, datetime

conversation_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("conversation_id", default=None)
trace_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("trace_id", default=None)
channel_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("channel", default=None)

_CONTEXT = {"conversation_id": conversation_id_var, "trace_id": trace_id_var, "channel": channel_var}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, var in _CONTEXT.items():
            if (value := var.get()) is not None:
                payload[key] = value
        # logger.info("...", extra={"fields": {...}}) adds structured fields
        payload.update(getattr(record, "fields", {}))
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    for noisy in ("httpx", "httpcore", "openai", "uvicorn.access", "aiosqlite"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
