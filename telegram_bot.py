import os
import re
import uuid
import json
import asyncio
import logging
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import httpx
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    MessageHandler,
    CommandHandler,
    ContextTypes,
    filters,
)

from voice_db import load_db, save_db, add_voiceprint, list_speakers

# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("asr-bot")

# ---------------- .env loader ----------------
def load_dotenv_if_exists(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and v and k not in os.environ:
            os.environ[k] = v


BASE_DIR = Path(__file__).parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv_if_exists(ENV_PATH)

AUDIO_INBOX = BASE_DIR / "audio" / "inbox"
AUDIO_INBOX.mkdir(parents=True, exist_ok=True)

JOBS_DIR = BASE_DIR / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

RUN_SCRIPT = BASE_DIR / "run_gpu.sh"
VOICE_DB_PATH = BASE_DIR / "voice_db.json"

TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN")
if not TG_BOT_TOKEN:
    raise RuntimeError("TG_BOT_TOKEN is not set. Put it into .env as TG_BOT_TOKEN=...")

DEFAULT_LANG = "ru"

# Telegram Bot API download limit (practical): 20 MB
TG_MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024  # 20MB
# For URL downloads, set your own limit (to avoid filling disk)
URL_MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2GB

URL_RE = re.compile(r"(https?://\S+)", re.IGNORECASE)

# GPU/IO: one job at a time (safe baseline)
PROCESS_LOCK = asyncio.Lock()


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


async def download_by_url(url: str, dest: Path, max_bytes: int) -> int:
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
    cmd = [str(RUN_SCRIPT), str(local_audio_path), lang, str(out_dir)]
    return await asyncio.to_thread(
        subprocess.run,
        cmd,
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
    )


def format_ts(seconds: float) -> str:
    ms = int(round(float(seconds) * 1000))
    s, ms = divmod(ms, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def render_segments_to_text(segments: list[dict]) -> str:
    lines = []
    for s in segments:
        lines.append(
            f"[{format_ts(s['start'])} --> {format_ts(s['end'])}] {s['speaker']}: {s.get('text','')}"
        )
    return "\n".join(lines)


async def post_unknown_prompt(msg, job_id: str, result_json_path: Path) -> None:
    try:
        payload = json.loads(result_json_path.read_text(encoding="utf-8"))
    except Exception:
        return

    unknowns = payload.get("unknown_speakers") or []
    if not unknowns:
        return

    # Сформируем список UNKNOWN_k = SPEAKER_XX
    lines = ["Обнаружены неизвестные говорящие:"]
    for u in unknowns:
        unk_id = u.get("id")
        label = u.get("label")
        lines.append(f"- {unk_id} = {label}")

    lines.append("")
    lines.append("Чтобы назвать и запомнить (добавить voiceprint в базу):")
    lines.append(f"/label {job_id} UNKNOWN_1 Иван")
    lines.append("(подставьте ваш UNKNOWN_k и имя)")

    await msg.reply_text("\n".join(lines))


async def handle_url(update: Update, context: ContextTypes.DEFAULT_TYPE, url: str):
    msg = update.message
    if msg is None:
        return

    await msg.reply_text("Ссылка получена. Скачиваю файл на сервер...")

    job_id = uuid.uuid4().hex
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    ext = ext_from_url(url)
    local_path = AUDIO_INBOX / f"url_{job_id}.{ext}"
    logger.info("Downloading URL: %s -> %s", url, local_path)

    try:
        size = await download_by_url(url, local_path, URL_MAX_DOWNLOAD_BYTES)
        await msg.reply_text(f"Файл скачан ({size/1024/1024:.1f} MB). Запускаю распознавание...")
    except Exception as e:
        logger.exception("URL download failed")
        await msg.reply_text(f"Ошибка скачивания по ссылке: {e}")
        return

    async with PROCESS_LOCK:
        proc = await run_asr(local_path, DEFAULT_LANG, job_dir)

    if proc.returncode != 0:
        logger.error("ASR failed: %s", proc.stderr[-2000:] if proc.stderr else proc.stdout[-2000:])
        await msg.reply_text("Ошибка при обработке аудио. Лог ниже:")
        err = (proc.stderr or proc.stdout or "no output")[-3500:]
        await msg.reply_text(err)
        return

    result_txt = job_dir / "result.txt"
    result_json = job_dir / "result.json"

    if not result_txt.exists():
        await msg.reply_text("Ошибка: result.txt не найден после обработки.")
        return

    text = result_txt.read_text(encoding="utf-8").strip()
    if len(text) > 3800:
        await msg.reply_document(document=result_txt, caption=f"job_id: {job_id}")
    else:
        await msg.reply_text(f"job_id: {job_id}\n\n{text}")

    if result_json.exists():
        await post_unknown_prompt(msg, job_id, result_json)


async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if msg is None:
        return

    await msg.reply_text("Аудио получено. Начинаю обработку...")

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

    # Download file from Telegram
    try:
        await msg.reply_text("Скачиваю файл из Telegram на сервер...")
        tg_file = await context.bot.get_file(tg_obj.file_id)
        local_path = AUDIO_INBOX / f"{tg_obj.file_id}_{job_id}.{ext}"
        await tg_file.download_to_drive(local_path)
        logger.info("Downloaded TG file -> %s", local_path)
        await msg.reply_text("Файл скачан. Запускаю распознавание...")
    except Exception as e:
        logger.exception("Telegram download failed")
        await msg.reply_text(
            f"Ошибка скачивания из Telegram: {e}\n"
            f"Если файл большой, пришлите прямую ссылку (https://...)."
        )
        return

    if not RUN_SCRIPT.exists():
        await msg.reply_text("Ошибка: run_gpu.sh не найден в корне проекта.")
        return

    async with PROCESS_LOCK:
        proc = await run_asr(local_path, DEFAULT_LANG, job_dir)

    if proc.returncode != 0:
        await msg.reply_text("Ошибка при обработке аудио. Лог ниже:")
        err = (proc.stderr or proc.stdout or "no output")[-3500:]
        await msg.reply_text(err)
        return

    result_txt = job_dir / "result.txt"
    result_json = job_dir / "result.json"

    if not result_txt.exists():
        await msg.reply_text("Ошибка: result.txt не найден после обработки.")
        return

    text = result_txt.read_text(encoding="utf-8").strip()
    if not text:
        await msg.reply_text("Готово, но result.txt пустой (возможно, в аудио нет речи).")
        return

    if len(text) > 3800:
        await msg.reply_document(document=result_txt, caption=f"job_id: {job_id}")
    else:
        await msg.reply_text(f"job_id: {job_id}\n\n{text}")

    if result_json.exists():
        await post_unknown_prompt(msg, job_id, result_json)


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if msg is None or not msg.text:
        return
    m = URL_RE.search(msg.text.strip())
    if not m:
        await msg.reply_text("Пришлите аудиофайл или прямую ссылку на скачивание (https://...).")
        return
    url = m.group(1)
    await handle_url(update, context, url)


async def cmd_label(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /label <job_id> <UNKNOWN_1> <Имя Фамилия>
    """
    msg = update.message
    if msg is None:
        return

    if len(context.args) < 3:
        await msg.reply_text("Использование:\n/label <job_id> <UNKNOWN_1> <Имя>\nПример:\n/label abcd1234 UNKNOWN_1 Иван")
        return

    job_id = context.args[0].strip()
    unknown_id = context.args[1].strip()
    name = " ".join(context.args[2:]).strip()

    job_dir = JOBS_DIR / job_id
    result_json = job_dir / "result.json"
    if not result_json.exists():
        await msg.reply_text(f"Не найден job_id={job_id} (нет {result_json}).")
        return

    payload = json.loads(result_json.read_text(encoding="utf-8"))
    unknowns = payload.get("unknown_speakers") or []
    target = next((u for u in unknowns if u.get("id") == unknown_id), None)
    if not target:
        await msg.reply_text(f"В этом job нет {unknown_id}.")
        return

    emb = target.get("embedding")
    if not emb:
        await msg.reply_text(f"У {unknown_id} нет embedding (возможно, слишком мало речи для voiceprint).")
        return

    db = load_db(VOICE_DB_PATH)
    add_voiceprint(db, name, emb)
    save_db(VOICE_DB_PATH, db)

    # Обновим job-результат (удобно сразу получить “перемапленный” текст)
    segments = payload.get("segments") or []
    changed = 0
    for s in segments:
        if s.get("speaker") == unknown_id:
            s["speaker"] = name
            changed += 1

    # Пометим в unknown_speakers что назначено имя
    for u in unknowns:
        if u.get("id") == unknown_id:
            u["assigned_name"] = name

    # Перепишем result.json и result.txt
    payload["segments"] = segments
    payload["unknown_speakers"] = unknowns
    result_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    result_txt = job_dir / "result.txt"
    result_txt.write_text(render_segments_to_text(segments), encoding="utf-8")

    await msg.reply_text(f"Ок. Запомнил голос как '{name}'. Обновил текущий job (заменил {changed} сегментов).")

    # Отправим обновлённый текст
    text = result_txt.read_text(encoding="utf-8").strip()
    if len(text) > 3800:
        await msg.reply_document(document=result_txt, caption=f"Обновлённый результат job_id: {job_id}")
    else:
        await msg.reply_text(f"Обновлённый job_id: {job_id}\n\n{text}")


async def cmd_speakers(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if msg is None:
        return
    db = load_db(VOICE_DB_PATH)
    names = list_speakers(db)
    if not names:
        await msg.reply_text("База голосов пока пустая.")
        return
    await msg.reply_text("Известные голоса:\n- " + "\n- ".join(names))


def main():
    app = ApplicationBuilder().token(TG_BOT_TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("label", cmd_label))
    app.add_handler(CommandHandler("speakers", cmd_speakers))

    # Regular messages
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO | filters.Document.ALL, handle_audio))

    print("Telegram ASR bot started (polling). Press Ctrl+C to stop.")
    app.run_polling()


if __name__ == "__main__":
    main()
