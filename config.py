from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path


logger = logging.getLogger("asr-bot")


def load_env_file(path: Path) -> None:
    """
    Minimal .env loader (no external deps).
    Does NOT override already-set environment variables.
    """
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if not k:
            continue

        if k not in os.environ and v != "":
            os.environ[k] = v


def get_required_env(key: str) -> str:
    v = os.getenv(key)
    if not v:
        raise RuntimeError(f"{key} is not set. Put it into .env as {key}=...")
    return v


def env_bool(key: str, default: bool = False) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


# ---- Base paths ----
BASE_DIR = Path(__file__).resolve().parent

# Load .env (server-only, should be in .gitignore)
load_env_file(BASE_DIR / ".env")

AUDIO_INBOX = BASE_DIR / "audio" / "inbox"
AUDIO_INBOX.mkdir(parents=True, exist_ok=True)

JOBS_DIR = BASE_DIR / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

RUN_SCRIPT = BASE_DIR / "run_gpu.sh"
VOICE_DB_PATH = BASE_DIR / "voice_db.json"

# ---- Telegram ----
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "")


def require_tg_bot_token() -> str:
    return get_required_env("TG_BOT_TOKEN")


MAX_TELEGRAM_TEXT = int(os.getenv("MAX_TELEGRAM_TEXT", "3800"))
TG_MAX_DOWNLOAD_BYTES = int(os.getenv("TG_MAX_DOWNLOAD_MB", "20")) * 1024 * 1024

# ---- URL downloads ----
URL_MAX_DOWNLOAD_BYTES = int(os.getenv("URL_MAX_DOWNLOAD_MB", "2048")) * 1024 * 1024
URL_RE = re.compile(r"(https?://\S+)", re.IGNORECASE)

# ---- ASR ----
DEFAULT_LANG = os.getenv("ASR_LANGUAGE", "ru")

# ---- Runtime behaviour ----
AUTO_PROTOCOL_ALWAYS = env_bool("AUTO_PROTOCOL_ALWAYS", default=True)
DELETE_AUDIO_AFTER_PROCESS = env_bool("DELETE_AUDIO_AFTER_PROCESS", default=False)

# ---- Pending label state ----
PENDING_LABEL_TTL_SEC = int(os.getenv("PENDING_LABEL_TTL_SEC", "900"))  # 15 minutes

# ---- Protocol / OpenAI ----
PROTOCOL_MODEL = os.getenv("PROTOCOL_MODEL", "gpt-4.1")
OPENAI_TPM_LIMIT = int(os.getenv("OPENAI_TPM_LIMIT", "30000"))
PROTOCOL_CHUNK_TOKENS = int(os.getenv("PROTOCOL_CHUNK_TOKENS", "6500"))
PROTOCOL_EXTRACT_OUT_TOKENS = int(os.getenv("PROTOCOL_EXTRACT_OUT_TOKENS", "1400"))
PROTOCOL_EXTRACT_TEMPERATURE = float(os.getenv("PROTOCOL_EXTRACT_TEMPERATURE", "0.0"))
EST_CHARS_PER_TOKEN = float(os.getenv("EST_CHARS_PER_TOKEN", "3.0"))
PROTOCOL_SAVE_DEBUG = env_bool("PROTOCOL_SAVE_DEBUG", default=False)

# ---- Concurrency locks (single GPU, file writes, etc.) ----
PROCESS_LOCK = asyncio.Lock()
PROTOCOL_LOCK = asyncio.Lock()
VOICE_DB_LOCK = asyncio.Lock()
