from __future__ import annotations

import re
from typing import List

import config
import ui
from handlers import flows
from services import voice_service
from state import clear_pending_label, set_pending_label


def _sort_unknown_ids(ids: List[str]) -> List[str]:
    def key(x: str) -> int:
        m = re.search(r"_(\d+)$", x)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return 999999
        return 999999

    return sorted(ids, key=key)


async def _prompt_unknown(msg, job_id: str, unknown_id: str, *, total: int, queue_len: int) -> None:
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


async def on_callback(update, context) -> None:
    q = update.callback_query
    if q is None:
        return
    await q.answer()

    data = (q.data or "").strip()
    parts = data.split(":")
    action = parts[0] if parts else ""

    if len(parts) < 2:
        return

    job_id = parts[1].strip()
    job_dir = config.JOBS_DIR / job_id

    if not job_dir.exists():
        await q.message.reply_text("job_id не найден.")  # type: ignore[union-attr]
        return

    # Full transcript (TXT)
    if action == "tx":
        await flows.send_transcript(q.message, job_id)  # type: ignore[arg-type]
        return

    # Zip artifacts
    if action == "zip":
        await flows.send_job_zip(q.message, job_id)  # type: ignore[arg-type]
        return

    # Regenerate protocol (under Word doc)
    if action == "prot":
        await flows.generate_and_send_protocol(q.message, job_id, announce=True)  # type: ignore[arg-type]
        return

    # ---- NEW: Word now (from post-ASR choice) ----
    if action == "w":
        # Use the current message as status/control, then delete it after sending Word
        await flows.generate_and_send_protocol(
            q.message,  # type: ignore[arg-type]
            job_id,
            announce=False,
            status_msg=q.message,
            delete_status=True,
        )
        return

    # ---- NEW: label unknowns first, then auto-send Word ----
    if action == "lw":
        unknowns = voice_service.get_unknowns(job_dir)
        ids = _sort_unknown_ids([str(u.get("id")) for u in unknowns if isinstance(u, dict) and u.get("id")])

        if not ids:
            # No unknowns -> just send Word
            await flows.generate_and_send_protocol(
                q.message,  # type: ignore[arg-type]
                job_id,
                announce=False,
                status_msg=q.message,
                delete_status=True,
            )
            return

        current = ids[0]
        queue = ids[1:]
        total = len(ids)

        set_pending_label(
            context.user_data,
            job_id=job_id,
            unknown_id=current,
            queue=queue,
            total=total,
            auto_protocol_after=True,
            control_chat_id=q.message.chat_id,      # type: ignore[union-attr]
            control_message_id=q.message.message_id,  # type: ignore[union-attr]
        )

        await q.message.reply_text(  # type: ignore[union-attr]
            "Ок. Сначала подпишем неизвестных спикеров. После последнего имени я пришлю финальный протокол в Word."
        )
        await _prompt_unknown(q.message, job_id, current, total=total, queue_len=len(queue))  # type: ignore[arg-type]
        return

    # Unknown menu (under Word doc)
    if action == "unk":
        unknowns = voice_service.get_unknowns(job_dir)
        if not unknowns:
            await q.message.reply_text("UNKNOWN-спикеры не обнаружены или уже подписаны.")  # type: ignore[union-attr]
            return
        kb = ui.unknown_select_keyboard(job_id, unknowns)
        await q.message.reply_text("Выберите, кого подписать:", reply_markup=kb)  # type: ignore[union-attr]
        return

    # Cancel pending label from inline menu
    if action == "uc":
        clear_pending_label(context.user_data)
        await q.message.reply_text("Ок, отменил.", reply_markup=ui.start_keyboard())  # type: ignore[union-attr]
        return

    # Single select unknown (after Word)
    if action == "us":
        if len(parts) < 3:
            await q.message.reply_text("Некорректные данные.")  # type: ignore[union-attr]
            return
        unknown_id = parts[2].strip()
        set_pending_label(context.user_data, job_id=job_id, unknown_id=unknown_id, queue=[], total=1)
        await _prompt_unknown(q.message, job_id, unknown_id, total=1, queue_len=0)  # type: ignore[arg-type]
        return

    # Bulk labeling (after Word) — without auto protocol
    if action == "ua":
        unknowns = voice_service.get_unknowns(job_dir)
        ids = _sort_unknown_ids([str(u.get("id")) for u in unknowns if isinstance(u, dict) and u.get("id")])
        if not ids:
            await q.message.reply_text("UNKNOWN-спикеры не обнаружены или уже подписаны.")  # type: ignore[union-attr]
            return

        current = ids[0]
        queue = ids[1:]
        total = len(ids)

        set_pending_label(context.user_data, job_id=job_id, unknown_id=current, queue=queue, total=total, auto_protocol_after=False)
        await _prompt_unknown(q.message, job_id, current, total=total, queue_len=len(queue))  # type: ignore[arg-type]
        return
