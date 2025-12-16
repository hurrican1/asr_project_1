from __future__ import annotations

from pathlib import Path

import config
import ui
from handlers import flows
from services import asr_service, download_service, voice_service
from state import clear_pending_label, get_pending_label, is_pending_expired, set_pending_label


def _normalize_name(text: str) -> str:
    return " ".join((text or "").split()).strip()


async def _prompt_next_unknown(msg, job_id: str, unknown_id: str, *, total: int, queue_len: int) -> None:
    job_dir = config.JOBS_DIR / job_id
    snippets = voice_service.get_speaker_snippets(job_dir, unknown_id, limit=3)

    hint = ""
    if snippets:
        hint = "Примеры реплик:\n" + "\n".join(f"• {s}" for s in snippets) + "\n\n"

    step = max(1, int(total) - int(queue_len))
    progress = ""
    if total > 1:
        progress = f"Шаг {step}/{total}.\n"

    await msg.reply_text(
        f"{hint}{progress}"
        f"Введите имя/роль для {unknown_id} одним сообщением.\n"
        f"Пример: «Иван Петров» или «Представитель франшизы».\n\n"
        f"Для отмены нажмите «{ui.BTN_CANCEL}» или /cancel.",
        reply_markup=ui.cancel_keyboard(),
    )


async def handle_menu_text(update, context) -> None:
    msg = update.message
    if msg is None or not msg.text:
        return

    text = msg.text.strip()

    # Pending UNKNOWN labeling (single or bulk)
    pending = get_pending_label(context.user_data)
    if pending and not is_pending_expired(pending, config.PENDING_LABEL_TTL_SEC):
        if text == ui.BTN_CANCEL:
            clear_pending_label(context.user_data)
            await msg.reply_text("Ок, отменил. Возвращаюсь в меню.", reply_markup=ui.start_keyboard())
            return

        name = _normalize_name(text)
        job_id = str(pending.get("job_id") or "").strip()
        unknown_id = str(pending.get("unknown_id") or "").strip()
        queue = pending.get("queue") if isinstance(pending.get("queue"), list) else []
        total = int(pending.get("total") or (1 + len(queue)))

        auto_protocol_after = bool(pending.get("auto_protocol_after"))
        control_chat_id = pending.get("control_chat_id")
        control_message_id = pending.get("control_message_id")

        clear_pending_label(context.user_data)

        if not name:
            await msg.reply_text("Имя пустое. Попробуйте ещё раз.", reply_markup=ui.start_keyboard())
            return

        try:
            changed, remaining = await voice_service.apply_label_to_job(job_id, unknown_id, name)
        except Exception as e:
            await msg.reply_text(f"Ошибка: {e}", reply_markup=ui.start_keyboard())
            return

        # Continue bulk flow
        if queue:
            next_unknown = str(queue.pop(0)).strip()
            set_pending_label(
                context.user_data,
                job_id=job_id,
                unknown_id=next_unknown,
                queue=queue,
                total=total,
                auto_protocol_after=auto_protocol_after,
                control_chat_id=control_chat_id if isinstance(control_chat_id, int) else None,
                control_message_id=control_message_id if isinstance(control_message_id, int) else None,
            )

            await msg.reply_text(
                f"Сохранено: {unknown_id} = {name}. "
                f"Обновил сегментов: {changed}. Осталось UNKNOWN: {remaining}.",
                reply_markup=ui.cancel_keyboard(),
            )
            await _prompt_next_unknown(msg, job_id, next_unknown, total=total, queue_len=len(queue))
            return

        # Finished labeling
        if auto_protocol_after:
            # Restore normal keyboard + inform we are creating the protocol now
            await msg.reply_text(
                f"Готово: {unknown_id} = {name}. Имена сохранены.\n"
                f"Формирую финальный протокол в Word…",
                reply_markup=ui.start_keyboard(),
            )

            # Try to update the control (pre-protocol) message to stage2 (optional)
            if isinstance(control_chat_id, int) and isinstance(control_message_id, int):
                try:
                    await context.bot.edit_message_text(
                        chat_id=control_chat_id,
                        message_id=control_message_id,
                        text=ui.STAGE2_TEXT,
                    )
                except Exception:
                    pass

            # Generate and send protocol WITHOUT extra stage message
            await flows.generate_and_send_protocol(msg, job_id, announce=False)

            # Delete control message (keep chat clean) if possible
            if isinstance(control_chat_id, int) and isinstance(control_message_id, int):
                try:
                    await context.bot.delete_message(chat_id=control_chat_id, message_id=control_message_id)
                except Exception:
                    pass
            return

        # Non-auto finalize: show actions
        job_dir = config.JOBS_DIR / job_id
        kb = ui.job_keyboard(job_id, has_unknown=bool(voice_service.get_unknowns(job_dir)))

        await msg.reply_text(
            f"Готово. {unknown_id} = {name}. Запомнил голос.\n"
            f"Обновил сегментов: {changed}. Осталось UNKNOWN: {remaining}.\n\n"
            f"Чтобы обновить протокол по этой встрече — нажмите «Пересобрать протокол».",
            reply_markup=ui.start_keyboard(),
        )
        await msg.reply_text("Действия по встрече:", reply_markup=kb)
        return

    # Expired pending -> clear
    if pending and is_pending_expired(pending, config.PENDING_LABEL_TTL_SEC):
        clear_pending_label(context.user_data)

    # Menu buttons
    if text == ui.BTN_AUDIO:
        await msg.reply_text(ui.HINT_AUDIO_TEXT, reply_markup=ui.start_keyboard())
        return
    if text == ui.BTN_LINK:
        await msg.reply_text(ui.HINT_LINK_TEXT, reply_markup=ui.start_keyboard())
        return
    if text == ui.BTN_SPEAKERS:
        names = voice_service.list_known_speaker_names()
        if not names:
            await msg.reply_text("База голосов пока пустая.", reply_markup=ui.start_keyboard())
        else:
            await msg.reply_text("Известные голоса:\n- " + "\n- ".join(names), reply_markup=ui.start_keyboard())
        return
    if text == ui.BTN_HELP:
        await msg.reply_text(ui.HELP_TEXT, reply_markup=ui.start_keyboard())
        return

    # Otherwise treat as normal text
    await handle_text(update, context)


async def handle_text(update, context) -> None:
    msg = update.message
    if msg is None or not msg.text:
        return

    text = msg.text.strip()
    m = config.URL_RE.search(text)
    if m:
        await handle_url(update, context, m.group(1))
        return

    await msg.reply_text(ui.FALLBACK_TEXT, reply_markup=ui.start_keyboard())


async def handle_url(update, context, url: str) -> None:
    msg = update.message
    if msg is None:
        return

    clear_pending_label(context.user_data)

    if not download_service.is_http_url(url):
        await msg.reply_text("Это не похоже на прямую HTTP/HTTPS ссылку.", reply_markup=ui.start_keyboard())
        return

    job_id, job_dir = asr_service.create_job()
    ext = download_service.ext_from_url(url)
    local_path = config.AUDIO_INBOX / f"url_{job_id}.{ext}"

    status_msg = await msg.reply_text(ui.STAGE1_TEXT, reply_markup=ui.start_keyboard())

    try:
        await download_service.download_by_url(url, local_path, max_bytes=config.URL_MAX_DOWNLOAD_BYTES)
    except Exception as e:
        try:
            await status_msg.edit_text(f"Ошибка скачивания: {e}")
        except Exception:
            pass
        await msg.reply_text(f"Ошибка скачивания по ссылке: {e}", reply_markup=ui.start_keyboard())
        return

    try:
        async with config.PROCESS_LOCK:
            proc = await asr_service.run_asr(local_path, lang=config.DEFAULT_LANG, job_dir=job_dir)
    except Exception as e:
        try:
            await status_msg.edit_text(f"Ошибка ASR: {e}")
        except Exception:
            pass
        await msg.reply_text(f"Ошибка запуска распознавания: {e}", reply_markup=ui.start_keyboard())
        return
    finally:
        if config.DELETE_AUDIO_AFTER_PROCESS:
            try:
                local_path.unlink()
            except Exception:
                pass

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "no output")[-3500:]
        try:
            await status_msg.edit_text("Ошибка при обработке аудио (ASR).")
        except Exception:
            pass
        await msg.reply_text("Ошибка при обработке аудио. Лог ниже:", reply_markup=ui.start_keyboard())
        await msg.reply_text(err, reply_markup=ui.start_keyboard())
        return

    unknowns = voice_service.get_unknowns(job_dir)

    # If unknowns exist -> ask user; else auto-protocol if enabled.
    if unknowns or not config.AUTO_PROTOCOL_ALWAYS:
        await flows.send_post_asr_choice(msg, job_id, status_msg=status_msg)
        return

    # Auto protocol (no unknowns)
    await flows.generate_and_send_protocol(msg, job_id, announce=False, status_msg=status_msg, delete_status=True)


async def handle_audio(update, context) -> None:
    msg = update.message
    if msg is None:
        return

    clear_pending_label(context.user_data)

    tg_obj = None
    ext = ".m4a"

    if msg.voice:
        tg_obj = msg.voice
        ext = ".ogg"
    elif msg.audio:
        tg_obj = msg.audio
        if msg.audio.file_name:
            ext = Path(msg.audio.file_name).suffix.lower() or ".m4a"
    elif msg.document:
        tg_obj = msg.document
        fname = (msg.document.file_name or "").lower()
        fext = Path(fname).suffix.lower()
        mtype = (msg.document.mime_type or "").lower()

        if (mtype.startswith("audio/")) or (fext in download_service.AUDIO_EXTS):
            ext = fext or ".m4a"
        else:
            await msg.reply_text(
                "Похоже, это не аудиофайл. Пришлите voice/audio или аудио-документ (.mp3/.m4a/.wav...).",
                reply_markup=ui.start_keyboard(),
            )
            return
    else:
        await msg.reply_text("Это не аудиофайл. Пришлите voice/audio/document.", reply_markup=ui.start_keyboard())
        return

    file_size = getattr(tg_obj, "file_size", None)
    if file_size and file_size > config.TG_MAX_DOWNLOAD_BYTES:
        await msg.reply_text(
            f"Файл слишком большой для скачивания через Telegram: {file_size/1024/1024:.1f} MB.\n"
            "Пришлите прямую ссылку на скачивание (https://...), и я обработаю её.",
            reply_markup=ui.start_keyboard(),
        )
        return

    job_id, job_dir = asr_service.create_job()
    status_msg = await msg.reply_text(ui.STAGE1_TEXT, reply_markup=ui.start_keyboard())

    try:
        tg_file = await context.bot.get_file(tg_obj.file_id)
        local_path = config.AUDIO_INBOX / f"{tg_obj.file_id}_{job_id}{ext}"
        await tg_file.download_to_drive(custom_path=str(local_path))
    except Exception as e:
        try:
            await status_msg.edit_text(f"Ошибка скачивания из Telegram: {e}")
        except Exception:
            pass
        await msg.reply_text(f"Ошибка скачивания из Telegram: {e}", reply_markup=ui.start_keyboard())
        return

    try:
        async with config.PROCESS_LOCK:
            proc = await asr_service.run_asr(local_path, lang=config.DEFAULT_LANG, job_dir=job_dir)
    except Exception as e:
        try:
            await status_msg.edit_text(f"Ошибка ASR: {e}")
        except Exception:
            pass
        await msg.reply_text(f"Ошибка запуска распознавания: {e}", reply_markup=ui.start_keyboard())
        return
    finally:
        if config.DELETE_AUDIO_AFTER_PROCESS:
            try:
                local_path.unlink()
            except Exception:
                pass

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "no output")[-3500:]
        try:
            await status_msg.edit_text("Ошибка при обработке аудио (ASR).")
        except Exception:
            pass
        await msg.reply_text("Ошибка при обработке аудио. Лог ниже:", reply_markup=ui.start_keyboard())
        await msg.reply_text(err, reply_markup=ui.start_keyboard())
        return

    unknowns = voice_service.get_unknowns(job_dir)

    # If unknowns exist -> ask user; else auto-protocol if enabled.
    if unknowns or not config.AUTO_PROTOCOL_ALWAYS:
        await flows.send_post_asr_choice(msg, job_id, status_msg=status_msg)
        return

    # Auto protocol (no unknowns)
    await flows.generate_and_send_protocol(msg, job_id, announce=False, status_msg=status_msg, delete_status=True)
