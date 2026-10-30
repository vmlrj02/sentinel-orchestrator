"""All runtime configuration, read once from the environment (.env locally, Lambda env vars in AWS)."""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _int(name: str, default: int) -> int:
    return int(_env(name, str(default)))  # type: ignore[arg-type]


@dataclass(frozen=True)
class Settings:
    environment: str
    log_level: str

    # LLM
    llm_provider: str  # "openai" or "fake" (deterministic, no API calls; for tests and load tests)
    openai_api_key: str | None
    interviewer_model: str  # model routing: cheap and fast for extraction...
    advisor_model: str      # ...and the one that reasons with tools for explanations
    fallback_model: str | None
    llm_timeout_s: float
    fake_llm_latency_ms: int

    retrieval_mode: str  # dense | hybrid | bm25, chosen by the retrieval eval

    # Conversation state
    checkpointer: str  # "memory", "sqlite" (local) or "dynamodb" (Lambda)
    sqlite_path: str
    checkpoint_table: str | None
    applications_table: str | None
    locks_table: str | None
    metrics_backend: str  # "prometheus" (one process) or "cloudwatch" (EMF, many instances)

    # Guards
    max_concurrent_turns: int
    rate_limit_per_minute: int
    turn_timeout_s: float
    max_message_chars: int
    max_history_messages: int

    # Internal endpoints (review queue, audit log, metrics, docs) need this bearer token
    admin_token: str | None

    # Telegram
    telegram_token: str | None
    telegram_mode: str  # "polling" (local), "webhook" (Lambda) or "off"
    telegram_webhook_secret: str | None
    allowed_chat_ids: frozenset[int]

    # Observability
    langfuse_enabled: bool
    public_traces: bool  # demo: every trace viewable by link, no login (PII is masked)
    flush_traces_each_turn: bool  # needed on Lambda, where the process freezes after a response

    @property
    def model_label(self) -> str:
        return "fake" if self.llm_provider == "fake" else f"{self.interviewer_model}+{self.advisor_model}"


def load_settings() -> Settings:
    from .aws import load_secrets
    load_secrets()  # no-op unless SECRETS_ARN is set (Lambda)
    chat_ids = (_env("ALLOWED_CHAT_IDS", "") or "").replace(" ", "")
    return Settings(
        environment=_env("ENVIRONMENT", "local") or "local",
        log_level=_env("LOG_LEVEL", "INFO") or "INFO",
        llm_provider=_env("LLM_PROVIDER", "openai") or "openai",
        openai_api_key=_env("OPENAI_API_KEY"),
        interviewer_model=_env("INTERVIEWER_MODEL", "gpt-5.4-mini") or "gpt-5.4-mini",
        advisor_model=_env("ADVISOR_MODEL", "gpt-5.4-mini") or "gpt-5.4-mini",
        fallback_model=_env("FALLBACK_MODEL", "gpt-5-mini"),
        llm_timeout_s=float(_env("LLM_TIMEOUT_S", "30") or 30),
        fake_llm_latency_ms=_int("FAKE_LLM_LATENCY_MS", 300),
        retrieval_mode=_env("RETRIEVAL_MODE", "dense") or "dense",
        checkpointer=_env("CHECKPOINTER", "sqlite") or "sqlite",
        sqlite_path=_env("SQLITE_PATH", "data/sentinel.db") or "data/sentinel.db",
        checkpoint_table=_env("CHECKPOINT_TABLE"),
        applications_table=_env("APPLICATIONS_TABLE"),
        locks_table=_env("LOCKS_TABLE"),
        metrics_backend=_env("METRICS_BACKEND", "prometheus") or "prometheus",
        max_concurrent_turns=_int("MAX_CONCURRENT_TURNS", 50),
        rate_limit_per_minute=_int("RATE_LIMIT_PER_MINUTE", 20),
        turn_timeout_s=float(_env("TURN_TIMEOUT_S", "60") or 60),
        max_message_chars=_int("MAX_MESSAGE_CHARS", 1000),
        max_history_messages=_int("MAX_HISTORY_MESSAGES", 30),
        admin_token=_env("ADMIN_TOKEN"),
        telegram_token=_env("TELEGRAM_TOKEN"),
        telegram_mode=_env("TELEGRAM_MODE", "polling") or "polling",
        telegram_webhook_secret=_env("TELEGRAM_WEBHOOK_SECRET"),
        allowed_chat_ids=frozenset(int(x) for x in chat_ids.split(",") if x),
        langfuse_enabled=bool(_env("LANGFUSE_PUBLIC_KEY") and _env("LANGFUSE_SECRET_KEY")),
        public_traces=(_env("LANGFUSE_PUBLIC_TRACES", "false") or "false").lower() == "true",
        flush_traces_each_turn=(_env("FLUSH_TRACES_EACH_TURN", "false") or "false").lower() == "true",
    )
