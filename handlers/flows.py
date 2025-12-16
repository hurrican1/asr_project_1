from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

import config
import ui
from handlers.common import send_document
from services import export_service, protocol_service, voice_service
from services.docx_service import build_protocol_docx


logger = logging.getLogger("asr-bot")


def get_job_dir(job_id: str) -> Path:
    return config.JOBS_DIR / job_id


async def send_transcript(msg, job_id: str) -> None:
    job_dir = get_job_dir(job_id)
    path = job_dir / "result.txt"
    if not path.exists():
        await msg.reply_text("Расшифровка не найдена.")
        return
    await send_document(msg, path, caption=f"Расшифровка (job_id: {job_id})")


async def send_job_zip(msg, job_id: str) -> None:
    try:
        zip_path = export_service.create_job_zip(job_id)
    except Exception as e:
        logger.exception("create_job_zip failed: job_id=%s", job_id)
        await msg.reply_text(f"Не удалось собрать архив: {e}")
        return

    await send_document(msg, zip_path, caption=f"Файлы встречи (job_id: {job_id})")

    try:
        zip_path.unlink()
    except Exception:
        pass


async def _try_edit_message(message, *, text: str, reply_markup=None) -> None:
    """
    Robust edit:
    - try to edit the existing status message
    - if it fails, log error and send a NEW message with the same text/buttons
    """
    try:
        if message is not None:
            await message.edit_text(text, reply_markup=reply_markup)
            return
    except Exception as e:
        logger.exception("Failed to edit status message: %s", e)

        if config.env_bool("BOT_DEBUG", False):
            try:
                await message.reply_text(f"DEBUG: edit_text failed: {type(e).__name__}: {e}")
            except Exception:
                pass

    # fallback: send a new message if editing failed
    try:
        if message is not None:
            await message.reply_text(text, reply_markup=reply_markup)
    except Exception as e2:
        logger.exception("Fallback reply_text also failed: %s", e2)


async def _try_delete_message(message) -> None:
    try:
        if message is not None:
            await message.delete()
    except Exception as e:
        logger.exception("Failed to delete message: %s", e)


async def send_post_asr_choice(
    msg,
    job_id: str,
    *,
    status_msg=None,
) -> None:
    """
    Show user choice after ASR, before protocol:
    - Word now
    - Label unknowns then Word (if unknown exists)
    - Transcript / Zip
    """
    job_dir = get_job_dir(job_id)
    unknowns = voice_service.get_unknowns(job_dir)
    ids = [str(u.get("id")) for u in unknowns if isinstance(u, dict) and u.get("id")]
    has_unknown = len(ids) > 0

    kb = ui.post_asr_keyboard(job_id, has_unknown=has_unknown)

    lines = [f"Расшифровка готова.", f"job_id: {job_id}"]
    if has_unknown:
        short = ", ".join(ids[:6])
        more = f" +{len(ids)-6}" if len(ids) > 6 else ""
        lines.append(f"Найдены неизвестные спикеры: {len(ids)} ({short}{more})")
        lines.append("Выберите: получить протокол сейчас или сначала подписать UNKNOWN (и получить финальный Word).")
    else:
        lines.append("UNKNOWN-спикеры не найдены.")
        lines.append("Нажмите «Протокол (Word) сейчас», чтобы получить документ.")

    text = "\n".join(lines)

    logger.info("Post-ASR choice: job_id=%s has_unknown=%s", job_id, has_unknown)

    if status_msg is not None:
        await _try_edit_message(status_msg, text=text, reply_markup=kb)
    else:
        await msg.reply_text(text, reply_markup=kb)


async def generate_and_send_protocol(
    msg,
    job_id: str,
    *,
    topic: Optional[str] = None,
    announce: bool = True,
    status_msg=None,
    delete_status: bool = False,
) -> None:
    """
    Generates protocol.docx and sends it with inline keyboard.
    If status_msg provided, tries to edit/delete it to keep chat clean.
    """
    job_dir = get_job_dir(job_id)
    result_txt = job_dir / "result.txt"
    if not result_txt.exists():
        await msg.reply_text("Ошибка: result.txt не найден для этого job.")
        return

    transcript = result_txt.read_text(encoding="utf-8").strip()

    unknowns = voice_service.get_unknowns(job_dir)
    unknown_count = len(unknowns)
    keyboard = ui.job_keyboard(job_id, has_unknown=unknown_count > 0)

    logger.info("Protocol generation started: job_id=%s unknown_count=%s", job_id, unknown_count)

    if status_msg is not None:
        await _try_edit_message(status_msg, text=ui.STAGE2_TEXT, reply_markup=None)
    elif announce:
        await msg.reply_text(ui.STAGE2_TEXT)

    try:
        async with config.PROTOCOL_LOCK:
            protocol_data, debug_payload = await asyncio.to_thread(
                protocol_service.generate_protocol_structured,
                transcript,
                participants=voice_service.extract_participants(job_dir),
                topic=topic,
            )
    except Exception as e:
        logger.exception("Protocol generation failed: job_id=%s", job_id)
        if status_msg is not None:
            await _try_edit_message(status_msg, text=f"Ошибка формирования протокола: {e}", reply_markup=None)
        await msg.reply_text(
            f"Протокол не сформировался: {e}\n"
            f"Отправляю расшифровку. Вы можете нажать «Пересобрать протокол» позже."
        )
        await send_document(msg, result_txt, caption=f"Расшифровка (job_id: {job_id})", reply_markup=keyboard)
        return

    protocol_docx = job_dir / "protocol.docx"
    try:
        build_protocol_docx(protocol_data, protocol_docx)
    except Exception as e:
        logger.exception("Word build failed: job_id=%s", job_id)
        if status_msg is not None:
            await _try_edit_message(status_msg, text=f"Ошибка формирования Word: {e}", reply_markup=None)
        await msg.reply_text(f"Ошибка формирования Word-файла: {e}")
        await send_document(msg, result_txt, caption=f"Расшифровка (job_id: {job_id})", reply_markup=keyboard)
        return

    if config.PROTOCOL_SAVE_DEBUG:
        (job_dir / "protocol_debug.json").write_text(
            json.dumps(debug_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    decisions_count = len(protocol_data.get("decisions") or [])
    tasks_count = len(protocol_data.get("action_items") or [])

    caption_lines = [
        "Протокол совещания",
        f"job_id: {job_id}",
        f"Решений: {decisions_count} | Задач: {tasks_count}",
    ]
    if unknown_count > 0:
        caption_lines.append("В документе есть UNKNOWN-спикеры — можно подписать и пересобрать протокол кнопкой ниже.")

    await send_document(
        msg,
        protocol_docx,
        caption="\n".join(caption_lines),
        reply_markup=keyboard,
    )

    if delete_status:
        await _try_delete_message(status_msg)
