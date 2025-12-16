from __future__ import annotations

import logging

from telegram.ext import ApplicationBuilder, CallbackQueryHandler, CommandHandler, MessageHandler, filters

import config
from handlers import callbacks, commands, messages


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("asr-bot")


def main() -> None:
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

    logger.info("Telegram ASR bot started (polling).")
    app.run_polling()


if __name__ == "__main__":
    main()
