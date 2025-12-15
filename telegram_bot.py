from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from voice_db import add_voiceprint, list_speakers, load_db, save_db

try:
    # Optional dependency: only needed for /protocol
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("asr-bot")


# ---------------- Helpers: env ----------------
def load_env_file(path: Path) -> None:
    """
    Minimal .env loader (no external deps). Does not override already-set env vars.
    """
    if not path.exists():
        return

    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and v and k not in os.environ:
                os.environ[k] = v
    except Exception as e:  # pragma: no cover
        logger.warning("Failed to read .env: %s", e)


def get_required_env(key: str) -> str:
    v = os.getenv(key)
    if not v:
        raise RuntimeError(f"{key} is not set. Put it into .env as {key}=...")
    return v


# ---------------- Paths ----------------
BASE_DIR = Path(__file__).resolve().parent
load_env_file(BASE_DIR / ".env")  # server-side only

AUDIO_INBOX = BASE_DIR / "audio" / "inbox"
AUDIO_INBOX.mkdir(parents=True, exist_ok=True)

JOBS_DIR = BASE_DIR / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

RUN_SCRIPT = BASE_DIR / "run_gpu.sh"
VOICE_DB_PATH = BASE_DIR / "voice_db.json"

# ---------------- Config ----------------
TG_BOT_TOKEN = get_required_env("TG_BOT_TOKEN")

DEFAULT_LANG = os.getenv("ASR_LANGUAGE", "ru")

# Telegram message practical limit
MAX_TELEGRAM_TEXT = 3800

# Telegram Bot API practical download limit (bot side): ~20MB
TG_MAX_DOWNLOAD_BYTES = int(os.getenv("TG_MAX_DOWNLOAD_MB", "20")) * 1024 * 1024

# URL download max safeguard (avoid filling disk)
URL_MAX_DOWNLOAD_BYTES = int(os.getenv("URL_MAX_DOWNLOAD_MB", "2048")) * 1024 * 1024  # default 2GB

# One GPU => process one job at a time (safe MVP)
PROCESS_LOCK = asyncio.Lock()

# Protect concurrent /label updates
VOICE_DB_LOCK = asyncio.Lock()

URL_RE = re.compile(r"(https?://\S+)", re.IGNORECASE)


# ---------------- Protocol (GPT) prompt ----------------
PROTOCOL_INSTRUCTIONS = r"""
Роль модели:
Ты — корпоративный секретарь и PMO-аналитик. Твоя задача — по транскрипту совещания подготовить официальный протокол: кратко, точно, без выдумок, с фиксацией решений, задач, сроков и ответственных.

Входные данные:
Тебе будет передано:
- Транскрипт совещания (возможны ошибки распознавания речи).
- Диаризация: реплики могут быть размечены как UNKNOWN_1, UNKNOWN_2 и т.п. или именами.
- (Опционально) Дата/время, тема, список участников, контекст проекта.

Главный принцип:
Никаких домыслов. Используй только информацию из транскрипта. Если важное поле не указано — пометь как «не указано» или сформулируй нейтрально «обсуждалось, но без финального решения».

1) НОРМАЛИЗАЦИЯ УЧАСТНИКОВ И РЕЧИ
- Если участники заданы как UNKNOWN_*, используй их как идентификаторы.
- Если в транскрипте есть явные соответствия (например: “UNKNOWN_2 — Даниил”), применяй их и дальше используй только имя.
- Если есть сомнения — оставляй UNKNOWN_* и не переименовывай.

2) ИЗВЛЕЧЕНИЕ СУЩНОСТЕЙ
По всему транскрипту извлеки и структурируй:
- Темы/вопросы повестки (что обсуждали).
- Решения (что решили сделать/утвердили/отклонили).
- Задачи (Action Items): что сделать, кто ответственный, срок, критерий готовности.
- Числа и факты: суммы, объемы, даты, KPI, количественные параметры.
- Риски/блокеры: что мешает, какие ограничения, что нужно для разблокировки.
- Открытые вопросы: что осталось без ответа/решения.
- Следующие шаги: кратко.

Если сроки не названы явно — не придумывай. Разрешено ставить: Срок: не указан.

3) ПРАВИЛА ОФОРМЛЕНИЯ ПРОТОКОЛА
- Стиль: деловой, нейтральный, без разговорных форм.
- Точность: короткие формулировки, без воды.
- Однозначность: каждое решение и задача должны быть отдельным пунктом.
- Приоритет: решения и задачи важнее пересказа дискуссии.
- Повторы/слова-паразиты/обрывки фраз из транскрипта не переносить.

4) ФОРМАТ ВЫХОДА (СТРОГО ПО ШАБЛОНУ)
Сформируй результат строго в следующей структуре без дополнительных разделов:

ПРОТОКОЛ СОВЕЩАНИЯ
1. Общая информация
Дата: {если есть в данных, иначе: не указано}
Время: {если есть, иначе: не указано}
Формат: {онлайн/офлайн/смешанный/не указано}
Тема: {сформулируй из контекста, если очевидно; иначе: не указано}
Участники:
{Имя или UNKNOWN_1}
{Имя или UNKNOWN_2}
…

2. Повестка (что обсуждали)
{Вопрос 1}
{Вопрос 2}
…
(Только фактические темы, без деталей дискуссии.)

3. Ключевые тезисы обсуждения (кратко)
{Тезис 1 в 1–2 строках}
{Тезис 2}
(Не более 8–12 пунктов. Только существенное.)

4. Принятые решения
Решение: {формулировка}
Основание/контекст: {1 строка, если нужно}
Решение: …
(Если решений не было: “Решения по итогам встречи не зафиксированы.”)

5. Задачи и поручения (Action Items)
Сделай таблицу в тексте:
№ | Задача | Ответственный | Срок | Критерий готовности / результат
1 | … | … | … | …

Правила:
- Каждая задача должна быть проверяемой.
- Если ответственный не назван — не указан.
- Если срок не назван — не указан.
- “Критерий готовности” — конкретный артефакт: документ, расчет, звонок, отправленное письмо, подготовленный файл, согласованный договор и т.п.

6. Риски и блокеры
{Риск/блокер 1} — {что требуется/как снять}
{Риск/блокер 2} — …

7. Открытые вопросы
{Вопрос 1}
{Вопрос 2}
(Если нет: “Открытые вопросы не зафиксированы.”)

8. Следующая встреча
Дата/время: {если есть, иначе: не указано}
Предварительная тема: {если есть, иначе: не указано}

5) ПРОВЕРКА КАЧЕСТВА ПЕРЕД ВЫВОДОМ
Перед финальным ответом проверь:
- Нет ли выдуманных сроков/ответственных/цифр.
- Все решения отделены от задач.
- Задачи сформулированы глаголом действия (подготовить/согласовать/отправить/рассчитать).
- Числа и даты переписаны точно как в транскрипте.
- Нет лишних разделов и комментариев.
""".strip()


# ---------------- Utilities ----------------
def ext_from_url(url: str) -> str:
    try:
        path = urlparse(url).path
        if "." in path:
            ext = path.rsplit(".", 1)[-1].lower()
            if 1 <= len(ext) <= 6:
                return ext
    except Exception:
        pass
    return "m4a"


def is_http_url(url: str) -> bool:
    try:
        u = urlparse(url)
        return u.scheme in ("http", "https") and bool(u.netloc)
    except Exception:
        return False


async def download_by_url(url: str, dest: Path, max_bytes: int) -> int:
    """
    Stream download URL to dest with a strict size limit.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    timeout = httpx.Timeout(connect=30.0, read=300.0, write=30.0, pool=30.0)
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
        async with client.stream("GET", url) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length") or 0)
            if total and total > max_bytes:
                raise RuntimeError(f"File too large by Content-Length: {total/1024/1024:.1f} MB")

            downloaded = 0
            with open(tmp, "wb") as f:
                async for chunk in r.aiter_bytes(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    downloaded += len(chunk)
                    if downloaded > max_bytes:
                        raise RuntimeError("File too large while downloading (limit exceeded).")
                    f.write(chunk)

    tmp.replace(dest)
    return downloaded


async def run_asr(local_audio_path: Path, lang: str, out_dir: Path) -> subprocess.CompletedProcess:
    """
    Executes run_gpu.sh <audio_path> <lang> <out_dir>
    """
    if not RUN_SCRIPT.exists():
        raise RuntimeError(f"run script not found: {RUN_SCRIPT}")

    cmd = [str(RUN_SCRIPT), str(local_audio_path), lang, str(out_dir)]
    logger.info("Running ASR: %s", " ".join(cmd))

    return await asyncio.to_thread(
        subprocess.run,
        cmd,
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
    )


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def format_ts(seconds: float) -> str:
    ms = int(round(float(seconds) * 1000))
    s, ms = divmod(ms, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def render_segments_to_text(segments: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for s in segments:
        try:
            lines.append(
                f"[{format_ts(s['start'])} --> {format_ts(s['end'])}] {s['speaker']}: {s.get('text','')}"
            )
        except Exception:
            continue
    return "\n".join(lines)


async def reply_document_from_path(msg, path: Path, caption: Optional[str] = None) -> None:
    with path.open("rb") as f:
        await msg.reply_document(document=f, filename=path.name, caption=caption)


async def reply_text_or_file(msg, text: str, *, file_path: Optional[Path] = None, caption: Optional[str] = None) -> None:
    """
    If text is short -> reply as text; otherwise send as a file.
    """
    text = (text or "").strip()
    if len(text) <= MAX_TELEGRAM_TEXT:
        if caption:
            await msg.reply_text(f"{caption}\n\n{text}" if text else caption)
        else:
            await msg.reply_text(text if text else "(пусто)")
        return

    if file_path and file_path.exists():
        await reply_document_from_path(msg, file_path, caption=caption)
        return

    tmp = BASE_DIR / f"long_{uuid.uuid4().hex}.txt"
    tmp.write_text(text, encoding="utf-8")
    try:
        await reply_document_from_path(msg, tmp, caption=caption)
    finally:
        try:
            tmp.unlink(missing_ok=True)  # type: ignore[attr-defined]
        except Exception:
            pass


async def post_unknown_prompt(msg, job_id: str, job_dir: Path) -> None:
    payload = read_json(job_dir / "result.json")
    if not payload:
        return

    unknowns = payload.get("unknown_speakers") or []
    if not unknowns:
        return

    lines = ["Обнаружены неизвестные говорящие:"]
    for u in unknowns:
        unk_id = u.get("id")
        label = u.get("label")
        if unk_id and label:
            lines.append(f"- {unk_id} = {label}")

    lines.append("")
    lines.append("Чтобы назвать и запомнить голос, отправьте команду:")
    lines.append(f"/label {job_id} UNKNOWN_1 Иван")
    lines.append("(подставьте ваш UNKNOWN_k и имя)")

    await msg.reply_text("\n".join(lines))


def extract_participants(job_dir: Path) -> List[str]:
    payload = read_json(job_dir / "result.json")
    if not payload:
        return []

    participants: List[str] = []
    for seg in (payload.get("segments") or []):
        sp = seg.get("speaker")
        if sp and sp not in participants:
            participants.append(sp)
    return participants


def generate_protocol_with_openai(transcript: str, *, topic: Optional[str], participants: List[str]) -> str:
    """
    Uses OpenAI API to generate meeting minutes protocol.
    Requires:
      - pip install openai
      - OPENAI_API_KEY in env/.env
    """
    if OpenAI is None:
        raise RuntimeError("Package 'openai' is not installed. Install: pip install openai")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env on the server.")

    model = os.getenv("PROTOCOL_MODEL", "gpt-4.1")
    max_output_tokens = int(os.getenv("PROTOCOL_MAX_OUTPUT_TOKENS", "2500"))
    temperature = float(os.getenv("PROTOCOL_TEMPERATURE", "0.2"))

    header_lines: List[str] = []
    if topic:
        header_lines.append(f"Тема (если задано): {topic}")
    if participants:
        header_lines.append("Участники (по транскрипту/диаризации):")
        header_lines.extend([f"- {p}" for p in participants])

    header = ("\n".join(header_lines).strip() + "\n\n") if header_lines else ""

    input_text = f"""{header}Ниже будет транскрипт. Используй его как единственный источник истины:
{transcript}
"""

    client = OpenAI()
    response = client.responses.create(
        model=model,
        instructions=PROTOCOL_INSTRUCTIONS,
        input=input_text,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        store=False,
    )
    return (getattr(response, "output_text", "") or "").strip()


# ---------------- Command handlers ----------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    if msg is None:
        return

    await msg.reply_text(
        "ASR бот запущен.\n\n"
        "Возможности:\n"
        "• Принимает voice/audio/document (до ~20MB) или прямую ссылку (http/https).\n"
        "• Делает транскрипт + диаризацию.\n"
        "• Если есть UNKNOWN_* — можно подписать голоса командой /label.\n"
        "• /speakers — список известных голосов.\n"
        "• /protocol <job_id> [тема] — протокол совещания через GPT (нужен OPENAI_API_KEY).\n\n"
        "Примеры:\n"
        "/label <job_id> UNKNOWN_1 Иван\n"
        "/protocol <job_id> Протокол по проекту"
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await cmd_start(update, context)


async def cmd_speakers(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    if msg is None:
        return

    db = load_db(VOICE_DB_PATH)
    names = list_speakers(db)
    if not names:
        await msg.reply_text("База голосов пока пустая.")
        return

    await msg.reply_text("Известные голоса:\n- " + "\n- ".join(names))


async def cmd_label(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /label <job_id> <UNKNOWN_1> <Имя Фамилия>
    """
    msg = update.message
    if msg is None:
        return

    if len(context.args) < 3:
        await msg.reply_text(
            "Использование:\n"
            "/label <job_id> <UNKNOWN_1> <Имя>\n"
            "Пример:\n"
            "/label ff686652... UNKNOWN_1 Иван"
        )
        return

    job_id = context.args[0].strip()
    unknown_id = context.args[1].strip()
    name = " ".join(context.args[2:]).strip()

    job_dir = JOBS_DIR / job_id
    result_json = job_dir / "result.json"
    if not result_json.exists():
        await msg.reply_text(f"Не найден job_id={job_id} (нет {result_json}).")
        return

    payload = read_json(result_json)
    if not payload:
        await msg.reply_text("Не удалось прочитать result.json.")
        return

    unknowns = payload.get("unknown_speakers") or []
    target = next((u for u in unknowns if u.get("id") == unknown_id), None)
    if not target:
        await msg.reply_text(f"В этом job нет {unknown_id}.")
        return

    emb = target.get("embedding")
    if not emb:
        await msg.reply_text(f"У {unknown_id} нет embedding (возможно, слишком мало речи для voiceprint).")
        return

    # Save voiceprint atomically
    async with VOICE_DB_LOCK:
        db = load_db(VOICE_DB_PATH)
        add_voiceprint(db, name, emb)
        save_db(VOICE_DB_PATH, db)

    # Update current job output for convenience (replace UNKNOWN_k -> name)
    segments = payload.get("segments") or []
    changed = 0
    for s in segments:
        if s.get("speaker") == unknown_id:
            s["speaker"] = name
            changed += 1

    for u in unknowns:
        if u.get("id") == unknown_id:
            u["assigned_name"] = name

    payload["segments"] = segments
    payload["unknown_speakers"] = unknowns

    result_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    result_txt = job_dir / "result.txt"
    result_txt.write_text(render_segments_to_text(segments), encoding="utf-8")

    await msg.reply_text(f"Ок. Запомнил голос как '{name}'. Обновил текущий job (заменил {changed} сегментов).")

    await reply_text_or_file(
        msg,
        result_txt.read_text(encoding="utf-8"),
        file_path=result_txt,
        caption=f"Обновлённый результат (job_id: {job_id})",
    )


async def cmd_protocol(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /protocol <job_id> [тема...]
    """
    msg = update.message
    if msg is None:
        return

    if len(context.args) < 1:
        await msg.reply_text(
            "Использование:\n"
            "/protocol <job_id> [тема]\n"
            "Пример:\n"
            "/protocol ff686652... Протокол совещания по проекту"
        )
        return

    job_id = context.args[0].strip()
    topic = " ".join(context.args[1:]).strip() if len(context.args) > 1 else None

    job_dir = JOBS_DIR / job_id
    result_txt = job_dir / "result.txt"
    if not result_txt.exists():
        await msg.reply_text(f"Не найден job_id={job_id} (нет {result_txt}).")
        return

    transcript = result_txt.read_text(encoding="utf-8").strip()
    if not transcript:
        await msg.reply_text("result.txt пустой — протокол составить невозможно.")
        return

    participants = extract_participants(job_dir)

    await msg.reply_text("Готовлю протокол совещания (GPT)...")

    try:
        protocol_text = await asyncio.to_thread(
            generate_protocol_with_openai,
            transcript,
            topic=topic,
            participants=participants,
        )
    except Exception as e:
        await msg.reply_text(f"Ошибка генерации протокола: {e}")
        return

    protocol_path = job_dir / "protocol.txt"
    protocol_path.write_text(protocol_text, encoding="utf-8")

    await reply_text_or_file(
        msg,
        protocol_text,
        file_path=protocol_path,
        caption=f"ПРОТОКОЛ (job_id: {job_id})",
    )


# ---------------- Message handlers ----------------
async def handle_url(update: Update, context: ContextTypes.DEFAULT_TYPE, url: str) -> None:
    msg = update.message
    if msg is None:
        return

    if not is_http_url(url):
        await msg.reply_text("Это не похоже на прямую HTTP/HTTPS ссылку.")
        return

    job_id = uuid.uuid4().hex
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    ext = ext_from_url(url)
    local_path = AUDIO_INBOX / f"url_{job_id}.{ext}"

    await msg.reply_text("Ссылка получена. Скачиваю файл на сервер...")

    try:
        size = await download_by_url(url, local_path, URL_MAX_DOWNLOAD_BYTES)
        await msg.reply_text(f"Файл скачан ({size/1024/1024:.1f} MB). Запускаю распознавание...")
    except Exception as e:
        logger.exception("URL download failed: %s", e)
        await msg.reply_text(f"Ошибка скачивания по ссылке: {e}")
        return

    try:
        async with PROCESS_LOCK:
            proc = await run_asr(local_path, DEFAULT_LANG, job_dir)
    except Exception as e:
        logger.exception("ASR execution failed: %s", e)
        await msg.reply_text(f"Ошибка запуска распознавания: {e}")
        return

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "no output")[-3500:]
        await msg.reply_text("Ошибка при обработке аудио. Лог ниже:")
        await msg.reply_text(err)
        return

    result_txt = job_dir / "result.txt"
    result_json = job_dir / "result.json"

    if not result_txt.exists():
        await msg.reply_text("Ошибка: result.txt не найден после обработки.")
        return

    transcript = result_txt.read_text(encoding="utf-8")
    await reply_text_or_file(msg, transcript, file_path=result_txt, caption=f"job_id: {job_id}")

    if result_json.exists():
        await post_unknown_prompt(msg, job_id, job_dir)


async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    if msg is None:
        return

    tg_obj = None
    ext = "m4a"

    if msg.voice:
        tg_obj = msg.voice
        ext = "ogg"
    elif msg.audio:
        tg_obj = msg.audio
        if msg.audio.file_name and "." in msg.audio.file_name:
            ext = msg.audio.file_name.rsplit(".", 1)[-1].lower()
    elif msg.document:
        tg_obj = msg.document
        if msg.document.file_name and "." in msg.document.file_name:
            ext = msg.document.file_name.rsplit(".", 1)[-1].lower()
    else:
        await msg.reply_text("Это не аудиофайл. Пришлите voice/audio/document.")
        return

    file_size = getattr(tg_obj, "file_size", None)
    if file_size and file_size > TG_MAX_DOWNLOAD_BYTES:
        await msg.reply_text(
            f"Файл слишком большой для скачивания через Telegram Bot API: {file_size/1024/1024:.1f} MB.\n"
            f"Пришлите, пожалуйста, прямую ссылку на скачивание (https://...), и я обработаю её."
        )
        return

    job_id = uuid.uuid4().hex
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    await msg.reply_text("Аудио получено. Скачиваю файл из Telegram на сервер...")

    try:
        tg_file = await context.bot.get_file(tg_obj.file_id)
        local_path = AUDIO_INBOX / f"{tg_obj.file_id}_{job_id}.{ext}"
        await tg_file.download_to_drive(custom_path=str(local_path))
        await msg.reply_text("Файл скачан. Запускаю распознавание...")
    except Exception as e:
        logger.exception("Telegram download failed: %s", e)
        await msg.reply_text(
            f"Ошибка скачивания из Telegram: {e}\n"
            f"Если файл большой, пришлите прямую ссылку (https://...)."
        )
        return

    try:
        async with PROCESS_LOCK:
            proc = await run_asr(local_path, DEFAULT_LANG, job_dir)
    except Exception as e:
        logger.exception("ASR execution failed: %s", e)
        await msg.reply_text(f"Ошибка запуска распознавания: {e}")
        return

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "no output")[-3500:]
        await msg.reply_text("Ошибка при обработке аудио. Лог ниже:")
        await msg.reply_text(err)
        return

    result_txt = job_dir / "result.txt"
    result_json = job_dir / "result.json"

    if not result_txt.exists():
        await msg.reply_text("Ошибка: result.txt не найден после обработки.")
        return

    transcript = result_txt.read_text(encoding="utf-8")
    await reply_text_or_file(msg, transcript, file_path=result_txt, caption=f"job_id: {job_id}")

    if result_json.exists():
        await post_unknown_prompt(msg, job_id, job_dir)


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    if msg is None or not msg.text:
        return

    text = msg.text.strip()
    m = URL_RE.search(text)
    if m:
        await handle_url(update, context, m.group(1))
        return

    await msg.reply_text("Пришлите аудио (voice/audio/document) или прямую ссылку (https://...).\n/help")


def main() -> None:
    app = ApplicationBuilder().token(TG_BOT_TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("label", cmd_label))
    app.add_handler(CommandHandler("speakers", cmd_speakers))
    app.add_handler(CommandHandler("protocol", cmd_protocol))

    # Messages
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO | filters.Document.ALL, handle_audio))

    logger.info("Telegram ASR bot started (polling).")
    app.run_polling()


if __name__ == "__main__":
    main()
