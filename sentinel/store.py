"""Applications: the audit record of every credit decision and the underwriter review queue.

Conversation state lives in the LangGraph checkpointer; this is the business record. Each row says
what was decided, by whom (policy or a named underwriter), with which model and prompt versions, and
links to the Langfuse trace, so any decision can be explained later.
"""
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

import aiosqlite

Status = Literal["approved", "declined", "pending_review"]


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class ApplicationRecord:
    application_id: str
    conversation_id: str
    tenant_id: str
    status: Status
    decided_by: str                     # "policy" or "underwriter:<name>"
    application: dict[str, Any]
    model_result: dict[str, Any]
    policy_reason: str
    model_version: str
    prompt_versions: dict[str, str]
    trace_id: str | None = None
    review_note: str | None = None
    created_at: str = field(default_factory=now)
    decided_at: str | None = None


class ApplicationStore(Protocol):
    async def upsert(self, record: ApplicationRecord) -> None: ...
    async def get(self, application_id: str) -> ApplicationRecord | None: ...
    async def list(self, status: Status | None = None, tenant_id: str | None = None,
                   limit: int = 50) -> list[ApplicationRecord]: ...


class MemoryApplicationStore:
    def __init__(self) -> None:
        self._rows: dict[str, ApplicationRecord] = {}

    async def upsert(self, record: ApplicationRecord) -> None:
        self._rows[record.application_id] = record

    async def get(self, application_id: str) -> ApplicationRecord | None:
        return self._rows.get(application_id)

    async def list(self, status: Status | None = None, tenant_id: str | None = None,
                   limit: int = 50) -> list[ApplicationRecord]:
        rows = [r for r in self._rows.values()
                if (status is None or r.status == status) and (tenant_id is None or r.tenant_id == tenant_id)]
        return sorted(rows, key=lambda r: r.created_at, reverse=True)[:limit]


_SCHEMA = """
CREATE TABLE IF NOT EXISTS applications (
    application_id  TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    tenant_id       TEXT NOT NULL,
    status          TEXT NOT NULL CHECK (status IN ('approved', 'declined', 'pending_review')),
    decided_by      TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    decided_at      TEXT,
    record          TEXT NOT NULL  -- full ApplicationRecord as JSON
);
CREATE INDEX IF NOT EXISTS applications_queue ON applications (tenant_id, status, created_at);
"""


class SqliteApplicationStore:
    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    @classmethod
    async def open(cls, path: str) -> "SqliteApplicationStore":
        conn = await aiosqlite.connect(path)
        await conn.executescript(_SCHEMA)
        await conn.commit()
        return cls(conn)

    async def close(self) -> None:
        await self._conn.close()

    async def upsert(self, record: ApplicationRecord) -> None:
        await self._conn.execute(
            "INSERT INTO applications VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(application_id) DO UPDATE SET "
            "status=excluded.status, decided_by=excluded.decided_by, decided_at=excluded.decided_at, "
            "record=excluded.record",
            (record.application_id, record.conversation_id, record.tenant_id, record.status, record.decided_by,
             record.created_at, record.decided_at, json.dumps(asdict(record))),
        )
        await self._conn.commit()

    async def get(self, application_id: str) -> ApplicationRecord | None:
        async with self._conn.execute("SELECT record FROM applications WHERE application_id = ?",
                                      (application_id,)) as cursor:
            row = await cursor.fetchone()
        return ApplicationRecord(**json.loads(row[0])) if row else None

    async def list(self, status: Status | None = None, tenant_id: str | None = None,
                   limit: int = 50) -> list[ApplicationRecord]:
        sql = "SELECT record FROM applications WHERE (? IS NULL OR status = ?) AND (? IS NULL OR tenant_id = ?) " \
              "ORDER BY created_at DESC LIMIT ?"
        async with self._conn.execute(sql, (status, status, tenant_id, tenant_id, limit)) as cursor:
            return [ApplicationRecord(**json.loads(row[0])) async for row in cursor]
