import logging

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from src.graph.workflow import create_sentinel_graph
from config.settings import TELEGRAM_TOKEN, ALLOWED_CHAT_IDS

logger = logging.getLogger(__name__)

# One graph for all chats; the checkpointer keeps each chat's application separate by thread_id
sentinel_app = create_sentinel_graph(MemorySaver())

WELCOME = (
    "Welcome to Sentinel, your automated loan underwriter.\n"
    "I'll ask a few questions, run your application through our credit model, and explain the result.\n"
    "Send /start any time to begin a fresh application. Let's begin: what is your name?"
)


def _thread_id(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    # /start bumps the session number, giving the chat a clean application state
    return f"{update.effective_chat.id}:{context.chat_data.get('session', 0)}"


def _allowed(update: Update) -> bool:
    return not ALLOWED_CHAT_IDS or update.effective_chat.id in ALLOWED_CHAT_IDS


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _allowed(update):
        return
    context.chat_data["session"] = context.chat_data.get("session", 0) + 1
    await update.message.reply_text(WELCOME)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _allowed(update):
        logger.warning("Ignoring message from chat %s (not in ALLOWED_CHAT_IDS)", update.effective_chat.id)
        return

    await context.bot.send_chat_action(update.effective_chat.id, ChatAction.TYPING)
    config = {"configurable": {"thread_id": _thread_id(update, context)}}

    try:
        output = await sentinel_app.ainvoke(
            {"messages": [HumanMessage(content=update.message.text)]}, config=config
        )
        # Replies are sent as plain text, so drop markdown bold the model sometimes adds
        reply = output["messages"][-1].content.replace("**", "")
    except Exception:
        logger.exception("Graph run failed")
        reply = "Sorry, something went wrong on my side. Please try again in a moment."

    await update.message.reply_text(reply)


def start_bot():
    """Starts the bot using the library's internal loop management."""
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    logger.info("Sentinel bot is live")
    application.run_polling(drop_pending_updates=True)
