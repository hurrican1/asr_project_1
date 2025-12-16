from __future__ import annotations

from typing import Any, Dict, List

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup


# Reply keyboard buttons (simple menu)
BTN_AUDIO = "🎙 Расшифровать аудио"
BTN_LINK = "🔗 Расшифровать по ссылке"
BTN_SPEAKERS = "👤 Голоса"
BTN_HELP = "❓ Помощь"
BTN_CANCEL = "Отмена"


START_TEXT = (
    "Привет! Я расшифровываю аудио и автоматически формирую протокол совещания в Word (.docx).\n\n"
    "Как пользоваться:\n"
    "1) Отправьте voice/audio или ссылку на аудиофайл\n"
    "2) Я выполню расшифровку + диаризацию\n"
    "3) Сразу пришлю протокол в Word\n\n"
    "Если появятся UNKNOWN‑спикеры — вы сможете подписать их кнопками, и бот будет узнавать голоса."
)

HELP_TEXT = (
    "Команды:\n"
    "/speakers — список известных голосов\n"
    "/label <job_id> <UNKNOWN_1> <Имя> — подписать голос вручную\n"
    "/protocol <job_id> [тема] — пересобрать протокол (Word)\n"
    "/transcript <job_id> — получить расшифровку\n"
    "/files <job_id> — архив файлов по встрече\n"
    "/cancel — отменить текущий ввод (например, подпись UNKNOWN)\n\n"
    "Проще всего: просто отправьте аудио или ссылку — Word придёт автоматически.\n"
    "Если есть UNKNOWN, бот предложит выбор: протокол сразу или сначала подписать."
)

HINT_AUDIO_TEXT = (
    "Пришлите voice/audio/document с аудио.\n"
    "Если файл большой — лучше прислать прямую ссылку (https://...)."
)

HINT_LINK_TEXT = (
    "Пришлите прямую ссылку на скачивание аудио (https://...).\n"
    "После обработки я автоматически пришлю Word-протокол."
)

FALLBACK_TEXT = (
    "Отправьте аудио (voice/audio/document) или прямую ссылку (https://...).\n"
    "Протокол в Word придёт автоматически.\n\n"
    "Для справки: /help"
)

STAGE1_TEXT = "Принял. Этап 1/2: скачиваю и распознаю…"
STAGE2_TEXT = "Этап 2/2: формирую протокол в Word…"


def start_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        [[BTN_AUDIO, BTN_LINK], [BTN_SPEAKERS, BTN_HELP]],
        resize_keyboard=True,
        one_time_keyboard=False,
    )


def cancel_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup([[BTN_CANCEL]], resize_keyboard=True, one_time_keyboard=True)


def post_asr_keyboard(job_id: str, has_unknown: bool) -> InlineKeyboardMarkup:
    """
    Keyboard shown AFTER ASR, before sending Word, when user should choose:
    - Word now
    - Label unknowns first -> then Word
    - Transcript
    - Zip
    """
    rows: List[List[InlineKeyboardButton]] = []

    first_row = [InlineKeyboardButton("📝 Протокол (Word) сейчас", callback_data=f"w:{job_id}")]
    if has_unknown:
        first_row.append(InlineKeyboardButton("👤 Подписать UNKNOWN → Word", callback_data=f"lw:{job_id}"))
    rows.append(first_row)

    rows.append(
        [
            InlineKeyboardButton("📄 Расшифровка (TXT)", callback_data=f"tx:{job_id}"),
            InlineKeyboardButton("🗂 Все файлы (zip)", callback_data=f"zip:{job_id}"),
        ]
    )

    return InlineKeyboardMarkup(rows)


def job_keyboard(job_id: str, has_unknown: bool) -> InlineKeyboardMarkup:
    """
    Keyboard shown UNDER the Word document (after protocol is sent).
    """
    buttons = [
        [
            InlineKeyboardButton("📄 Расшифровка (TXT)", callback_data=f"tx:{job_id}"),
            InlineKeyboardButton("🔁 Пересобрать протокол", callback_data=f"prot:{job_id}"),
        ],
        [
            InlineKeyboardButton("🗂 Все файлы (zip)", callback_data=f"zip:{job_id}"),
        ],
    ]
    if has_unknown:
        buttons.insert(1, [InlineKeyboardButton("👤 Подписать UNKNOWN", callback_data=f"unk:{job_id}")])
    return InlineKeyboardMarkup(buttons)


def unknown_select_keyboard(job_id: str, unknowns: List[Dict[str, Any]]) -> InlineKeyboardMarkup:
    """
    Menu for labeling unknowns AFTER the Word was sent (optional).
    Includes bulk labeling.
    """
    rows: List[List[InlineKeyboardButton]] = []
    rows.append([InlineKeyboardButton("✅ Подписать всех (по очереди)", callback_data=f"ua:{job_id}")])

    row: List[InlineKeyboardButton] = []
    for u in unknowns:
        unk_id = str(u.get("id", "")).strip()
        if not unk_id:
            continue
        label = str(u.get("label") or "").strip()
        text = unk_id if not label else f"{unk_id} ({label})"
        row.append(InlineKeyboardButton(text, callback_data=f"us:{job_id}:{unk_id}"))
        if len(row) >= 2:
            rows.append(row)
            row = []

    if row:
        rows.append(row)

    rows.append([InlineKeyboardButton(BTN_CANCEL, callback_data=f"uc:{job_id}")])
    return InlineKeyboardMarkup(rows)
