import os
import re
import uuid
import asyncio
import logging
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import httpx
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters


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

RUN_SCRIPT = BASE_DIR / "run_gpu.sh"

TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN")
if not TG_BOT_TOKEN:
    raise RuntimeError("TG_BOT_TOKEN is not set. Put it into .env as TG_BOT_TOKEN=...")

DEFAULT_LANG = "ru"

# Telegram Bot API download limit (practical): 20 MB
TG_MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024  # 20MB
# For URL downloads, set your own limit (to avoid filling disk)
URL_MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2GB

URL_RE = re.compile(r"(https?://\S+)", re.IGNORECASE)

# Prevent concurrent runs (because run_asr.py writes result.txt/result.json)
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


async def run_asr(local_audio_path: Path, lang: str) -> subprocess.CompletedProcess:
    cmd = [str(RUN_SCRIPT), str(local_audio_path), lang]
    # run in a thread to avoid blocking the async event loop
    return await asyncio.to_thread(
        subprocess.run,
        cmd,
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
    )


async def handle_url(update: Update, context: ContextTypes.DEFAULT_TYPE, url: str):
    msg = update.message
    if msg is None:
        return

    await msg.reply_text("Ссылка получена. Скачиваю файл на сервер...")

    ext = ext_from_url(url)
    local_path = AUDIO_INBOX / f"url_{uuid.uuid4().hex}.{ext}"

    logger.info("Downloading URL: %s -> %s", url, local_path)

    try:
        size = await download_by_url(url, local_path, URL_MAX_DOWNLOAD_BYTES)
        await msg.reply_text(f"Файл скачан ({size/1024/1024:.1f} MB). Запускаю распознавание...")
    except Exception as e:
        logger.exception("URL download failed")
        await msg.reply_text(f"Ошибка скачивания по ссылке: {e}")
        return

    async with PROCESS_LOCK:
        proc = await run_asr(local_path, DEFAULT_LANG)

    if proc.returncode != 0:
        logger.error("ASR failed: %s", proc.stderr[-2000:] if proc.stderr else proc.stdout[-2000:])
        await msg.reply_text("Ошибка при обработке аудио. Лог ниже:")
        err = (proc.stderr or proc.stdout or "no output")[-3500:]
        await msg.reply_text(err)
        return

    result_txt = BASE_DIR / "result.txt"
    if not result_txt.exists():
        await msg.reply_text("Ошибка: result.txt не найден после обработки.")
        return

    text = result_txt.read_text(encoding="utf-8").strip()
    if len(text) > 3800:
        await msg.reply_document(document=result_txt)
    else:
        await msg.reply_text(text)


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

    # Check size if provided
    file_size = getattr(tg_obj, "file_size", None)
    if file_size and file_size > TG_MAX_DOWNLOAD_BYTES:
        await msg.reply_text(
            f"Файл слишком большой для скачивания через Telegram Bot API: "
            f"{file_size/1024/1024:.1f} MB.\n"
            f"Telegram Bot API обычно позволяет боту скачивать файлы только до ~20 MB.\n"
            f"Пришлите, пожалуйста, прямую ссылку на скачивание (https://...), и я обработаю её."
        )
        return

    # Download file from Telegram
    try:
        await msg.reply_text("Скачиваю файл из Telegram на сервер...")
        tg_file = await context.bot.get_file(tg_obj.file_id)
        local_path = AUDIO_INBOX / f"{tg_obj.file_id}.{ext}"
        await tg_file.download_to_drive(local_path)
        logger.info("Downloaded TG file -> %s", local_path)
        await msg.reply_text("Файл скачан. Запускаю распознавание...")
    except Exception as e:
        logger.exception("Telegram download failed")
        await msg.reply_text(f"Ошибка скачивания из Telegram: {e}\n"
                             f"Если файл большой, пришлите прямую ссылку (https://...).")
        return

    if not RUN_SCRIPT.exists():
        await msg.reply_text("Ошибка: run_gpu.sh не найден в корне проекта.")
        return

    async with PROCESS_LOCK:
        proc = await run_asr(local_path, DEFAULT_LANG)

    if proc.returncode != 0:
        await msg.reply_text("Ошибка при обработке аудио. Лог ниже:")
        err = (proc.stderr or proc.stdout or "no output")[-3500:]
        await msg.reply_text(err)
        return

    result_txt = BASE_DIR / "result.txt"
    if not result_txt.exists():
        await msg.reply_text("Ошибка: result.txt не найден после обработки.")
        return

    text = result_txt.read_text(encoding="utf-8").strip()
    if not text:
        await msg.reply_text("Готово, но result.txt пустой (возможно, в аудио нет речи).")
        return

    if len(text) > 3800:
        await msg.reply_document(document=result_txt)
    else:
        await msg.reply_text(text)


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


def main():
    app = ApplicationBuilder().token(TG_BOT_TOKEN).build()

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO | filters.Document.ALL, handle_audio))

    print("Telegram ASR bot started (polling). Press Ctrl+C to stop.")
    app.run_polling()


if __name__ == "__main__":
    main()