from __future__ import annotations

from pathlib import Path
from typing import Optional

from telegram import InputFile, InlineKeyboardMarkup


async def send_document(msg, path: Path, *, caption: str = "", reply_markup: Optional[InlineKeyboardMarkup] = None) -> None:
    with path.open("rb") as f:
        await msg.reply_document(
            document=InputFile(f, filename=path.name),
            caption=(caption or "")[:1000],
            reply_markup=reply_markup,
        )
