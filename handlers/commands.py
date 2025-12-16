from __future__ import annotations

import config
import ui
from handlers import flows
from services import voice_service
from state import clear_pending_label, get_pending_label, is_pending_expired


async def cmd_start(update, context) -> None:
    msg = update.message
    if msg is None:
        return
    clear_pending_label(context.user_data)
    await msg.reply_text(ui.START_TEXT, reply_markup=ui.start_keyboard())


async def cmd_help(update, context) -> None:
    msg = update.message
    if msg is None:
        return
    await msg.reply_text(ui.HELP_TEXT, reply_markup=ui.start_keyboard())


async def cmd_cancel(update, context) -> None:
    msg = update.message
    if msg is None:
        return

    pending = get_pending_label(context.user_data)
    if pending and not is_pending_expired(pending, config.PENDING_LABEL_TTL_SEC):
        clear_pending_label(context.user_data)
        await msg.reply_text("Ок, отменил. Возвращаюсь в меню.", reply_markup=ui.start_keyboard())
    else:
        clear_pending_label(context.user_data)
        await msg.reply_text("Сейчас нечего отменять.", reply_markup=ui.start_keyboard())


async def cmd_speakers(update, context) -> None:
    msg = update.message
    if msg is None:
        return
    names = voice_service.list_known_speaker_names()
    if not names:
        await msg.reply_text("База голосов пока пустая.", reply_markup=ui.start_keyboard())
        return
    await msg.reply_text("Известные голоса:\n- " + "\n- ".join(names), reply_markup=ui.start_keyboard())


async def cmd_label(update, context) -> None:
    """
    Manual labeling:
      /label <job_id> <UNKNOWN_1> <Имя Фамилия>
    """
    msg = update.message
    if msg is None:
        return

    args = context.args or []
    if len(args) < 3:
        await msg.reply_text(
            "Использование:\n"
            "/label <job_id> <UNKNOWN_1> <Имя>\n"
            "Пример:\n"
            "/label ff686652... UNKNOWN_1 Иван",
            reply_markup=ui.start_keyboard(),
        )
        return

    job_id = args[0].strip()
    unknown_id = args[1].strip()
    name = " ".join(args[2:]).strip()

    try:
        changed, remaining = await voice_service.apply_label_to_job(job_id, unknown_id, name)
    except Exception as e:
        await msg.reply_text(f"Ошибка: {e}", reply_markup=ui.start_keyboard())
        return

    await msg.reply_text(
        f"Готово. {unknown_id} = {name}. Запомнил голос.\n"
        f"Обновил сегментов: {changed}. Осталось UNKNOWN: {remaining}.\n"
        f"Чтобы обновить протокол — нажмите «Пересобрать протокол» под документом или /protocol {job_id}.",
        reply_markup=ui.start_keyboard(),
    )


async def cmd_protocol(update, context) -> None:
    msg = update.message
    if msg is None:
        return

    args = context.args or []
    if not args:
        await msg.reply_text("Использование: /protocol <job_id> [тема]", reply_markup=ui.start_keyboard())
        return

    job_id = args[0].strip()
    topic = " ".join(args[1:]).strip() if len(args) > 1 else None

    job_dir = config.JOBS_DIR / job_id
    if not job_dir.exists():
        await msg.reply_text("job_id не найден.", reply_markup=ui.start_keyboard())
        return

    await flows.generate_and_send_protocol(msg, job_id, topic=topic, announce=True)


async def cmd_transcript(update, context) -> None:
    msg = update.message
    if msg is None:
        return

    args = context.args or []
    if not args:
        await msg.reply_text("Использование: /transcript <job_id>", reply_markup=ui.start_keyboard())
        return

    await flows.send_transcript(msg, args[0].strip())


async def cmd_files(update, context) -> None:
    msg = update.message
    if msg is None:
        return

    args = context.args or []
    if not args:
        await msg.reply_text("Использование: /files <job_id>", reply_markup=ui.start_keyboard())
        return

    await flows.send_job_zip(msg, args[0].strip())
