"""Telegram channel. One handler for both delivery modes:
- webhook: Telegram POSTs updates to /telegram/webhook (Lambda)
- polling: a background task long-polls getUpdates (local, no public URL needed)
"""
import asyncio
import logging
from typing import Any

import httpx

from .service import ChatService

logger = logging.getLogger(__name__)

WELCOME = (
    "Welcome to Sentinel, your automated loan underwriter.\n"
    "I'll ask a few questions, run your application through our credit model, and explain the result.\n"
    "Send /start any time to begin a fresh application. Let's begin: what is your name?"
)


class TelegramAdapter:
    def __init__(self, token: str, service: ChatService, allowed_chat_ids: frozenset[int] = frozenset()) -> None:
        self.service = service
        self.allowed_chat_ids = allowed_chat_ids
        self._api = f"https://api.telegram.org/bot{token}"
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(15, read=40))
        self._tasks: set[asyncio.Task[None]] = set()  # strong refs so running handlers aren't garbage-collected

    async def close(self) -> None:
        await self._http.aclose()

    async def call(self, method: str, **params: Any) -> Any:
        response = await self._http.post(f"{self._api}/{method}", json=params)
        body = response.json()
        if not body.get("ok"):
            raise RuntimeError(f"Telegram {method} failed: {body.get('description')}")
        return body["result"]

    async def handle_update(self, update: dict[str, Any]) -> None:
        # Telegram re-delivers an update if we were slow or failed; process each one once
        if not self.service.seen_messages.check_and_add(f"tg-update-{update.get('update_id')}"):
            return
        if "callback_query" in update:
            await self._handle_feedback(update["callback_query"])
            return

        message = update.get("message") or {}
        text, chat_id = message.get("text"), (message.get("chat") or {}).get("id")
        if not text or chat_id is None:
            return
        if self.allowed_chat_ids and chat_id not in self.allowed_chat_ids:
            logger.warning("Ignoring chat not in ALLOWED_CHAT_IDS", extra={"fields": {"chat_id": chat_id}})
            return

        conversation_id = f"tg-{chat_id}"
        if text.startswith("/start"):
            await self.service.reset(conversation_id)
            await self.send(chat_id, WELCOME)
            return

        typing = asyncio.create_task(self.call("sendChatAction", chat_id=chat_id, action="typing"))
        result = await self.service.run_turn(conversation_id, user_id=f"tg-{message['from']['id']}",
                                             text=text, channel="telegram")
        typing.cancel()
        for i, reply in enumerate(result.messages):
            # Thumbs up/down under the analyst's answers; the rating is scored on that turn's trace
            is_last = i == len(result.messages) - 1
            await self.send(chat_id, reply, feedback_for=result.trace_id if is_last and result.explained else None)

    async def send(self, chat_id: int, text: str, feedback_for: str | None = None) -> None:
        params: dict[str, Any] = {"chat_id": chat_id, "text": text[:4096]}
        if feedback_for:
            params["reply_markup"] = {"inline_keyboard": [[
                {"text": "👍 Helpful", "callback_data": f"fb:{feedback_for}:1"},
                {"text": "👎 Not helpful", "callback_data": f"fb:{feedback_for}:-1"},
            ]]}
        await self.call("sendMessage", **params)

    async def _handle_feedback(self, query: dict[str, Any]) -> None:
        _, trace_id, rating = (query.get("data") or "::").split(":")
        if trace_id:
            self.service.record_feedback(trace_id, int(rating or 0))
        await self.call("answerCallbackQuery", callback_query_id=query["id"], text="Thanks for the feedback!")

    async def poll_forever(self) -> None:
        """Local mode. Each update is handled in its own task, so one slow chat never blocks the others."""
        await self.call("deleteWebhook", drop_pending_updates=True)
        me = await self.call("getMe")
        logger.info("Telegram polling started", extra={"fields": {"bot": me.get("username")}})
        offset = 0
        while True:
            try:
                updates = await self.call("getUpdates", offset=offset, timeout=30,
                                          allowed_updates=["message", "callback_query"])
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("getUpdates failed; retrying in 5s")
                await asyncio.sleep(5)
                continue
            for update in updates:
                offset = update["update_id"] + 1
                task = asyncio.create_task(self._safe_handle(update))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)

    async def _safe_handle(self, update: dict[str, Any]) -> None:
        try:
            await self.handle_update(update)
        except Exception:
            logger.exception("Failed to handle Telegram update")
