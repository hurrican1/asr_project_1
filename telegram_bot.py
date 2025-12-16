from __future__ import annotations

import logging
import os

from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

import config
from handlers import callbacks, commands, messages


def _setup_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # IMPORTANT: prevent leaking bot token in logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.WARNING)


logger = logging.getLogger("asr-bot")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Global error handler: logs full exception stack trace.
    If BOT_DEBUG=1, sends short error message to the chat.
    """
    logger.exception("Unhandled exception while processing update", exc_info=context.error)

    if os.getenv("BOT_DEBUG", "0").strip().lower() in ("1", "true", "yes", "y", "on"):
        try:
            chat = getattr(update, "effective_chat", None)
            if chat is not None:
                await context.bot.send_message(
                    chat_id=chat.id,
                    text=f"DEBUG: ошибка в обработчике: {type(context.error).__name__}: {context.error}",
                )
        except Exception:
            pass


def main() -> None:
    _setup_logging()

    token = config.require_tg_bot_token()
    app = ApplicationBuilder().token(token).build()

    # Commands
    app.add_handler(CommandHandler("start", commands.cmd_start))
    app.add_handler(CommandHandler("help", commands.cmd_help))
    app.add_handler(CommandHandler("speakers", commands.cmd_speakers))
    app.add_handler(CommandHandler("label", commands.cmd_label))
    app.add_handler(CommandHandler("protocol", commands.cmd_protocol))
    app.add_handler(CommandHandler("transcript", commands.cmd_transcript))
    app.add_handler(CommandHandler("files", commands.cmd_files))
    app.add_handler(CommandHandler("cancel", commands.cmd_cancel))

    # Inline callbacks
    app.add_handler(CallbackQueryHandler(callbacks.on_callback))

    # Text menu and normal text
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, messages.handle_menu_text))

    # Audio / documents
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO | filters.Document.ALL, messages.handle_audio))

    # Global error handler
    app.add_error_handler(error_handler)

    logger.info("Telegram ASR bot started (polling).")
    app.run_polling()


if __name__ == "__main__":
    main()
