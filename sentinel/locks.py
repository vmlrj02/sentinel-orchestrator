"""Per-conversation locks: two messages in one chat must be processed one after the other.

In one process an asyncio.Lock is enough. With several instances (Lambda), messages from the same chat
can land on different instances, so the lock must live in shared storage: see DynamoLocks in aws.py.
"""
import asyncio
import weakref
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Protocol


class ConversationLocks(Protocol):
    def hold(self, conversation_id: str) -> AbstractAsyncContextManager[None]: ...


class InProcessLocks:
    def __init__(self) -> None:
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()

    @asynccontextmanager
    async def hold(self, conversation_id: str) -> AsyncIterator[None]:
        lock = self._locks.get(conversation_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[conversation_id] = lock
        async with lock:
            yield
