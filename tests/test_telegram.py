from sentinel.telegram import WELCOME, TelegramAdapter
from tests.conftest import APPLICATION_TURNS


def make_adapter(service):
    adapter = TelegramAdapter("TOKEN", service)
    sent: list[tuple[str, dict]] = []

    async def fake_call(method, **params):
        sent.append((method, params))
        return {}

    adapter.call = fake_call  # type: ignore[method-assign]
    return adapter, sent


def update(update_id: int, text: str, chat_id: int = 42) -> dict:
    return {"update_id": update_id, "message": {"text": text, "chat": {"id": chat_id}, "from": {"id": 7}}}


async def test_redelivered_update_is_processed_once(service, fake_llm):
    adapter, sent = make_adapter(service)
    await adapter.handle_update(update(1, "person_age=30"))
    await adapter.handle_update(update(1, "person_age=30"))
    assert fake_llm.calls == 1
    assert sum(1 for m, _ in sent if m == "sendMessage") == 1


async def test_start_resets_and_welcomes(service):
    adapter, sent = make_adapter(service)
    await adapter.handle_update(update(1, "person_age=30"))
    await adapter.handle_update(update(2, "/start"))
    assert sent[-1] == ("sendMessage", {"chat_id": 42, "text": WELCOME})
    assert (await service.get_state("tg-42"))["application"] == {}


async def test_explanation_gets_feedback_buttons(service):
    adapter, sent = make_adapter(service)
    for i, text in enumerate([*APPLICATION_TURNS, "yes"]):
        await adapter.handle_update(update(i, text))
    last = [p for m, p in sent if m == "sendMessage"][-1]
    buttons = last["reply_markup"]["inline_keyboard"][0]
    assert buttons[0]["callback_data"].startswith("fb:") and buttons[0]["callback_data"].endswith(":1")

    await adapter.handle_update({"update_id": 99, "callback_query": {"id": "q", "data": buttons[1]["callback_data"]}})
    assert sent[-1][0] == "answerCallbackQuery"


async def test_answer_options_become_a_tap_keyboard(service):
    adapter, sent = make_adapter(service)
    await adapter.handle_update(update(1, "person_age=30 person_income=50000 person_emp_length=3"))
    keyboard = [p for m, p in sent if m == "sendMessage"][-1]["reply_markup"]["keyboard"]
    assert [b["text"] for row in keyboard for b in row] == ["Rent", "Own", "Mortgage", "Other"]
