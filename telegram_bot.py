from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import subprocess
import time
import uuid
import zipfile
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx
from telegram import (
    InputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
    Update,
)
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from voice_db import add_voiceprint, list_speakers, load_db, save_db
from protocol_docx import build_protocol_docx

try:
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
    """Minimal .env loader (no external deps). Does not override already-set env vars."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and v and k not in os.environ:
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


# ---------------- Paths ----------------
BASE_DIR = Path(__file__).resolve().parent
load_env_file(BASE_DIR / ".env")

AUDIO_INBOX = BASE_DIR / "audio" / "inbox"
AUDIO_INBOX.mkdir(parents=True, exist_ok=True)

JOBS_DIR = BASE_DIR / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

RUN_SCRIPT = BASE_DIR / "run_gpu.sh"
VOICE_DB_PATH = BASE_DIR / "voice_db.json"


# ---------------- Config ----------------
TG_BOT_TOKEN = get_required_env("TG_BOT_TOKEN")

DEFAULT_LANG = os.getenv("ASR_LANGUAGE", "ru")

MAX_TELEGRAM_TEXT = int(os.getenv("MAX_TELEGRAM_TEXT", "3800"))

TG_MAX_DOWNLOAD_BYTES = int(os.getenv("TG_MAX_DOWNLOAD_MB", "20")) * 1024 * 1024
URL_MAX_DOWNLOAD_BYTES = int(os.getenv("URL_MAX_DOWNLOAD_MB", "2048")) * 1024 * 1024

DELETE_AUDIO_AFTER_PROCESS = env_bool("DELETE_AUDIO_AFTER_PROCESS", default=False)

AUTO_PROTOCOL_ALWAYS = env_bool("AUTO_PROTOCOL_ALWAYS", default=True)

# One GPU => process one job at a time
PROCESS_LOCK = asyncio.Lock()
VOICE_DB_LOCK = asyncio.Lock()
PROTOCOL_LOCK = asyncio.Lock()

URL_RE = re.compile(r"(https?://\S+)", re.IGNORECASE)

# Menu button texts
BTN_AUDIO = "🎙 Расшифровать аудио"
BTN_LINK = "🔗 Расшифровать по ссылке"
BTN_SPEAKERS = "👤 Голоса"
BTN_HELP = "❓ Помощь"


# ---------------- Protocol (Map-Reduce extract) ----------------
EXTRACT_INSTRUCTIONS = """
Роль модели:
Ты — корпоративный секретарь и PMO-аналитик.

Задача:
Из входного фрагмента транскрипта извлеки факты по встрече. Никаких домыслов.
Если информация не указана явно — используй значение "не указано" или не добавляй элемент в список.

Правила:
- Не придумывай сроки/ответственных/цифры.
- В "decisions" добавляй только явные решения (сделаем/утвердили/решили).
- В "action_items" добавляй только явные поручения. Если нет ответственного/срока — ставь "не указан".
- Верни СТРОГО валидный JSON-объект. Никаких комментариев, текста, Markdown.

Формат JSON (строго):
{
  "agenda": [string],
  "key_points": [string],
  "decisions": [{"decision": string, "context": string}],
  "action_items": [{"task": string, "owner": string, "due": string, "done_criteria": string}],
  "numbers_facts": [string],
  "risks": [{"risk": string, "mitigation": string}],
  "open_questions": [string],
  "next_meeting": {"datetime": string, "topic": string}
}
""".strip()

TS_RE = re.compile(r"^\[\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}\]\s*", re.M)


def strip_timestamps(transcript: str) -> str:
    return TS_RE.sub("", transcript).strip()


def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def estimate_tokens(text: str, chars_per_token: float) -> int:
    text = text or ""
    if not text:
        return 0
    return int((len(text) / max(chars_per_token, 1.5)) + 1)


def chunk_by_lines(text: str, *, max_input_tokens_est: int, chars_per_token: float) -> List[str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    chunks: List[str] = []
    buf: List[str] = []
    buf_tokens = 0
    for ln in lines:
        ln_tokens = estimate_tokens(ln, chars_per_token) + 1
        if buf and (buf_tokens + ln_tokens) > max_input_tokens_est:
            chunks.append("\n".join(buf))
            buf = [ln]
            buf_tokens = ln_tokens
        else:
            buf.append(ln)
            buf_tokens += ln_tokens
    if buf:
        chunks.append("\n".join(buf))
    return chunks


def parse_json_robust(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(raw)
    except Exception:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(raw[start : end + 1])

    raise ValueError("Model returned non-JSON output")


class TokenRateLimiter:
    """Simple TPM limiter using estimated tokens over the last 60 seconds."""
    def __init__(self, tpm_limit: int):
        self.tpm_limit = max(1, int(tpm_limit))
        self.window_sec = 60.0
        self.events: Deque[Tuple[float, int]] = deque()

    def _prune(self, now: float) -> None:
        while self.events and (now - self.events[0][0]) > self.window_sec:
            self.events.popleft()

    def wait_for_budget(self, tokens_needed: int) -> None:
        tokens_needed = max(0, int(tokens_needed))
        while True:
            now = time.monotonic()
            self._prune(now)
            used = sum(t for _, t in self.events)
            if used + tokens_needed <= self.tpm_limit:
                self.events.append((now, tokens_needed))
                return

            if not self.events:
                time.sleep(1.0)
                continue

            oldest_ts, _ = self.events[0]
            wait_s = (oldest_ts + self.window_sec) - now
            time.sleep(max(0.2, min(wait_s, 10.0)))


def openai_call_extract_json(
    client: "OpenAI",
    *,
    model: str,
    instructions: str,
    chunk_text: str,
    max_output_tokens: int,
    temperature: float,
    limiter: TokenRateLimiter,
    chars_per_token: float,
    max_retries: int = 8,
) -> str:
    """
    JSON extraction call with:
    - local TPM throttling
    - JSON mode text.format=json_object
    - requirement: 'json' must appear in INPUT text (we include it explicitly)
    """
    # IMPORTANT: input must include "json" word for json_object mode
    input_text = f"Ответ строго в json (json_object). Только json.\n\n{chunk_text}"

    est = (
        estimate_tokens(instructions, chars_per_token)
        + estimate_tokens(input_text, chars_per_token)
        + int(max_output_tokens)
    )

    for attempt in range(max_retries):
        try:
            limiter.wait_for_budget(est)

            resp = client.responses.create(
                model=model,
                instructions=instructions,
                input=input_text,
                text={"format": {"type": "json_object"}},
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                store=False,
            )

            if getattr(resp, "status", None) == "incomplete":
                raise RuntimeError(
                    "OpenAI ответ incomplete (скорее всего max_output_tokens). "
                    "Увеличьте PROTOCOL_EXTRACT_OUT_TOKENS или уменьшите PROTOCOL_CHUNK_TOKENS."
                )

            return (getattr(resp, "output_text", "") or "").strip()

        except Exception as e:
            msg = str(e)
            if ("429" in msg) or ("rate_limit" in msg) or ("TPM" in msg) or ("tokens per min" in msg):
                time.sleep(min((2 ** attempt) + random.random(), 20.0))
                continue
            raise

    raise RuntimeError("OpenAI rate limit: retries exhausted")


def normalize_extract_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    def as_list(x) -> List[Any]:
        return x if isinstance(x, list) else []

    def as_dict(x) -> Dict[str, Any]:
        return x if isinstance(x, dict) else {}

    out: Dict[str, Any] = {
        "agenda": [normalize_space(s) for s in as_list(payload.get("agenda")) if normalize_space(str(s))],
        "key_points": [normalize_space(s) for s in as_list(payload.get("key_points")) if normalize_space(str(s))],
        "decisions": [],
        "action_items": [],
        "numbers_facts": [normalize_space(s) for s in as_list(payload.get("numbers_facts")) if normalize_space(str(s))],
        "risks": [],
        "open_questions": [normalize_space(s) for s in as_list(payload.get("open_questions")) if normalize_space(str(s))],
        "next_meeting": {"datetime": "не указано", "topic": "не указано"},
    }

    for d in as_list(payload.get("decisions")):
        dct = as_dict(d)
        decision = normalize_space(dct.get("decision", ""))
        context = normalize_space(dct.get("context", "")) or "не указано"
        if decision:
            out["decisions"].append({"decision": decision, "context": context})

    for a in as_list(payload.get("action_items")):
        dct = as_dict(a)
        task = normalize_space(dct.get("task", ""))
        if not task:
            continue
        owner = normalize_space(dct.get("owner", "")) or "не указан"
        due = normalize_space(dct.get("due", "")) or "не указан"
        done = normalize_space(dct.get("done_criteria", "")) or "не указано"
        out["action_items"].append({"task": task, "owner": owner, "due": due, "done_criteria": done})

    for r in as_list(payload.get("risks")):
        dct = as_dict(r)
        risk = normalize_space(dct.get("risk", ""))
        if not risk:
            continue
        mitigation = normalize_space(dct.get("mitigation", "")) or "не указано"
        out["risks"].append({"risk": risk, "mitigation": mitigation})

    nm = as_dict(payload.get("next_meeting"))
    out["next_meeting"] = {
        "datetime": normalize_space(nm.get("datetime", "")) or "не указано",
        "topic": normalize_space(nm.get("topic", "")) or "не указано",
    }

    return out


def merge_extracts(extracts: List[Dict[str, Any]]) -> Dict[str, Any]:
    def norm_key(s: str) -> str:
        return normalize_space(s).lower()

    def dedup_str_list(items: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in items:
            k = norm_key(x)
            if not k or k in seen:
                continue
            seen.add(k)
            out.append(x)
        return out

    merged = {
        "agenda": [],
        "key_points": [],
        "decisions": [],
        "action_items": [],
        "numbers_facts": [],
        "risks": [],
        "open_questions": [],
        "next_meeting": {"datetime": "не указано", "topic": "не указано"},
    }

    dec_seen = set()
    task_seen = set()
    risk_seen = set()

    for ex in extracts:
        merged["agenda"].extend(ex.get("agenda", []))
        merged["key_points"].extend(ex.get("key_points", []))
        merged["numbers_facts"].extend(ex.get("numbers_facts", []))
        merged["open_questions"].extend(ex.get("open_questions", []))

        for d in ex.get("decisions", []):
            decision = normalize_space(d.get("decision", ""))
            context = normalize_space(d.get("context", "")) or "не указано"
            k = norm_key(decision)
            if not k or k in dec_seen:
                continue
            dec_seen.add(k)
            merged["decisions"].append({"decision": decision, "context": context})

        for a in ex.get("action_items", []):
            task = normalize_space(a.get("task", ""))
            owner = normalize_space(a.get("owner", "")) or "не указан"
            due = normalize_space(a.get("due", "")) or "не указан"
            done = normalize_space(a.get("done_criteria", "")) or "не указано"
            k = f"{norm_key(task)}|{norm_key(owner)}|{norm_key(due)}"
            if not norm_key(task) or k in task_seen:
                continue
            task_seen.add(k)
            merged["action_items"].append({"task": task, "owner": owner, "due": due, "done_criteria": done})

        for r in ex.get("risks", []):
            risk = normalize_space(r.get("risk", ""))
            mitigation = normalize_space(r.get("mitigation", "")) or "не указано"
            k = norm_key(risk)
            if not k or k in risk_seen:
                continue
            risk_seen.add(k)
            merged["risks"].append({"risk": risk, "mitigation": mitigation})

        nm = ex.get("next_meeting", {}) or {}
        if nm.get("datetime") and nm["datetime"] != "не указано":
            merged["next_meeting"]["datetime"] = nm["datetime"]
        if nm.get("topic") and nm["topic"] != "не указано":
            merged["next_meeting"]["topic"] = nm["topic"]

    merged["agenda"] = dedup_str_list(merged["agenda"])[:12]
    merged["key_points"] = dedup_str_list(merged["key_points"] + merged["numbers_facts"])[:12]
    merged["decisions"] = merged["decisions"][:30]
    merged["action_items"] = merged["action_items"][:50]
    merged["risks"] = merged["risks"][:20]
    merged["open_questions"] = dedup_str_list(merged["open_questions"])[:20]

    return merged


def generate_protocol_structured(
    transcript_txt: str,
    *,
    participants: List[str],
    topic: Optional[str],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Returns protocol_data (for DOCX) and debug payload.
    """
    if OpenAI is None:
        raise RuntimeError("Package 'openai' is not installed. Install: pip install openai")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env on the server.")

    model = os.getenv("PROTOCOL_MODEL", "gpt-4.1")

    chars_per_token = float(os.getenv("EST_CHARS_PER_TOKEN", "3.0"))
    tpm_limit = int(os.getenv("OPENAI_TPM_LIMIT", "30000"))
    chunk_tokens = int(os.getenv("PROTOCOL_CHUNK_TOKENS", "6500"))
    extract_out_tokens = int(os.getenv("PROTOCOL_EXTRACT_OUT_TOKENS", "1400"))
    temperature = float(os.getenv("PROTOCOL_EXTRACT_TEMPERATURE", "0.0"))

    limiter = TokenRateLimiter(tpm_limit)
    client = OpenAI()

    clean = strip_timestamps(transcript_txt)
    if not clean:
        raise RuntimeError("Transcript is empty after preprocessing.")

    chunks = chunk_by_lines(clean, max_input_tokens_est=chunk_tokens, chars_per_token=chars_per_token)

    extracts: List[Dict[str, Any]] = []
    for idx, ch in enumerate(chunks, start=1):
        chunk_text = f"ЧАСТЬ {idx}/{len(chunks)}\n{ch}"
        out_text = openai_call_extract_json(
            client,
            model=model,
            instructions=EXTRACT_INSTRUCTIONS,
            chunk_text=chunk_text,
            max_output_tokens=extract_out_tokens,
            temperature=temperature,
            limiter=limiter,
            chars_per_token=chars_per_token,
        )
        payload = parse_json_robust(out_text)
        extracts.append(normalize_extract_payload(payload))

    merged = merge_extracts(extracts)

    protocol_data: Dict[str, Any] = {
        "date": "не указано",
        "time": "не указано",
        "format": "не указано",
        "topic": normalize_space(topic) if topic else "не указано",
        "participants": participants or ["не указано"],
        "agenda": merged.get("agenda") or [],
        "key_points": merged.get("key_points") or [],
        "decisions": merged.get("decisions") or [],
        "action_items": merged.get("action_items") or [],
        "risks": merged.get("risks") or [],
        "open_questions": merged.get("open_questions") or [],
        "next_meeting": merged.get("next_meeting") or {"datetime": "не указано", "topic": "не указано"},
    }

    debug = {
        "chunks": len(chunks),
        "model": model,
        "tpm_limit": tpm_limit,
        "chunk_tokens_est": chunk_tokens,
        "extract_out_tokens": extract_out_tokens,
        "chars_per_token": chars_per_token,
        "merged": merged,
    }
    return protocol_data, debug


# ---------------- ASR utilities ----------------
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".opus", ".aac", ".flac", ".mp4", ".webm"}


def ext_from_url(url: str) -> str:
    try:
        path = urlparse(url).path
        if "." in path:
            ext = "." + path.rsplit(".", 1)[-1].lower()
            if ext in AUDIO_EXTS:
                return ext.lstrip(".")
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


async def send_file(chat_msg, path: Path, caption: str, reply_markup=None) -> None:
    with path.open("rb") as f:
        await chat_msg.reply_document(
            document=InputFile(f, filename=path.name),
            caption=caption[:1000],
            reply_markup=reply_markup,
        )


def get_unknowns(job_dir: Path) -> List[Dict[str, Any]]:
    payload = read_json(job_dir / "result.json") or {}
    unknowns = payload.get("unknown_speakers") or []
    return unknowns if isinstance(unknowns, list) else []


def build_job_keyboard(job_id: str, has_unknown: bool) -> InlineKeyboardMarkup:
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


async def auto_protocol_and_send(update_msg, job_id: str, job_dir: Path, topic: Optional[str] = None) -> None:
    """
    Auto-generate protocol.docx and send to user with inline keyboard.
    Fallback: send transcript if protocol fails.
    """
    result_txt = job_dir / "result.txt"
    if not result_txt.exists():
        await update_msg.reply_text("Ошибка: result.txt не найден после обработки.")
        return

    transcript = result_txt.read_text(encoding="utf-8").strip()
    unknowns = get_unknowns(job_dir)
    has_unknown = len(unknowns) > 0
    kb = build_job_keyboard(job_id, has_unknown=has_unknown)

    await update_msg.reply_text("Этап 2/2: формирую протокол в Word…")

    try:
        async with PROTOCOL_LOCK:
            protocol_data, debug_payload = await asyncio.to_thread(
                generate_protocol_structured,
                transcript,
                participants=extract_participants(job_dir),
                topic=topic,
            )
    except Exception as e:
        await update_msg.reply_text(
            f"Протокол не сформировался: {e}\n"
            f"Отправляю расшифровку. Вы можете нажать «Пересобрать протокол» позже."
        )
        await send_file(update_msg, result_txt, caption=f"Расшифровка (job_id: {job_id})", reply_markup=kb)
        return

    protocol_docx = job_dir / "protocol.docx"
    try:
        build_protocol_docx(protocol_data, protocol_docx)
    except Exception as e:
        await update_msg.reply_text(f"Ошибка формирования Word-файла: {e}")
        await send_file(update_msg, result_txt, caption=f"Расшифровка (job_id: {job_id})", reply_markup=kb)
        return

    # Optionally store debug
    if env_bool("PROTOCOL_SAVE_DEBUG", default=False):
        (job_dir / "protocol_debug.json").write_text(
            json.dumps(debug_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    caption = f"Протокол совещания (job_id: {job_id})"
    await send_file(update_msg, protocol_docx, caption=caption, reply_markup=kb)


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


# ---------------- Telegram commands ----------------
def start_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        [[BTN_AUDIO, BTN_LINK], [BTN_SPEAKERS, BTN_HELP]],
        resize_keyboard=True,
        one_time_keyboard=False,
    )


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    if msg is None:
        return
    await msg.reply_text(
        "Привет! Я расшифровываю аудио и автоматически формирую протокол совещания в Word (.docx).\n\n"
        "Как пользоваться:\n"
        "1) Отправьте voice/audio или ссылку на аудиофайл\n"
        "2) Я выполню расшифровку + диаризацию\n"
        "3) Сразу пришлю протокол в Word\n\n"
        "Если появятся UNKNOWN‑спикеры — вы сможете подписать их, и бот будет узнавать голоса в следующих встречах.",
        reply_markup=start_keyboard(),
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    if msg is None:
        return
    await msg.reply_text(
        "Команды:\n"
        "/speakers — список известных голосов\n"
        "/label <job_id> <UNKNOWN_1> <Имя> — подписать и запомнить голос\n"
        "/protocol <job_id> [тема] — пересобрать протокол (Word)\n"
        "/transcript <job_id> — получить расшифровку\n"
        "/files <job_id> — архив всех файлов по встрече\n\n"
        "Самый простой сценарий: просто отправьте аудио или ссылку — Word придёт автоматически.",
        reply_markup=start_keyboard(),
    )


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
        await msg.reply_text(f"Не найден job_id={job_id}.")
        return

    payload = read_json(result_json) or {}
    unknowns = payload.get("unknown_speakers") or []
    target = next((u for u in unknowns if isinstance(u, dict) and u.get("id") == unknown_id), None)
    if not target:
        await msg.reply_text(f"В этом job нет {unknown_id}.")
        return

    emb = target.get("embedding")
    if not emb:
        await msg.reply_text(f"У {unknown_id} нет embedding (возможно, слишком мало речи для voiceprint).")
        return

    async with VOICE_DB_LOCK:
        db = load_db(VOICE_DB_PATH)
        add_voiceprint(db, name, emb)
        save_db(VOICE_DB_PATH, db)

    # Update job segments speaker names (optional convenience)
    segments = payload.get("segments") or []
    changed = 0
    for s in segments:
        if isinstance(s, dict) and s.get("speaker") == unknown_id:
            s["speaker"] = name
            changed += 1

    for u in unknowns:
        if isinstance(u, dict) and u.get("id") == unknown_id:
            u["assigned_name"] = name

    payload["segments"] = segments
    payload["unknown_speakers"] = unknowns
    result_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    await msg.reply_text(f"Ок. Запомнил голос как '{name}'. Обновил текущий job (заменил {changed} сегментов).")


async def cmd_protocol(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Manual /protocol to regenerate Word.
    """
    msg = update.message
    if msg is None:
        return
    if len(context.args) < 1:
        await msg.reply_text("Использование: /protocol <job_id> [тема]")
        return
    job_id = context.args[0].strip()
    topic = " ".join(context.args[1:]).strip() if len(context.args) > 1 else None

    job_dir = JOBS_DIR / job_id
    if not job_dir.exists():
        await msg.reply_text("job_id не найден.")
        return

    await msg.reply_text("Пересобираю протокол (Word)…")
    await auto_protocol_and_send(msg, job_id, job_dir, topic=topic)


async def cmd_transcript(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    if msg is None:
        return
    if len(context.args) < 1:
        await msg.reply_text("Использование: /transcript <job_id>")
        return
    job_id = context.args[0].strip()
    job_dir = JOBS_DIR / job_id
    path = job_dir / "result.txt"
    if not path.exists():
        await msg.reply_text("Расшифровка не найдена.")
        return
    await send_file(msg, path, caption=f"Расшифровка (job_id: {job_id})")


async def cmd_files(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    if msg is None:
        return
    if len(context.args) < 1:
        await msg.reply_text("Использование: /files <job_id>")
        return
    job_id = context.args[0].strip()
    await send_job_zip(msg, job_id)


# ---------------- Inline buttons callbacks ----------------
async def send_job_zip(msg, job_id: str) -> None:
    job_dir = JOBS_DIR / job_id
    if not job_dir.exists():
        await msg.reply_text("job_id не найден.")
        return

    zip_path = job_dir / f"job_{job_id}.zip"
    # Only pack known artifacts
    to_pack = ["result.txt", "result.json", "protocol.docx", "protocol_debug.json"]

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name in to_pack:
            p = job_dir / name
            if p.exists():
                z.write(p, arcname=name)

    await send_file(msg, zip_path, caption=f"Файлы встречи (job_id: {job_id})")
    try:
        zip_path.unlink(missing_ok=True)  # type: ignore[attr-defined]
    except Exception:
        pass


async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    if q is None:
        return
    await q.answer()

    data = (q.data or "").strip()
    if ":" not in data:
        return
    action, job_id = data.split(":", 1)
    job_dir = JOBS_DIR / job_id

    if not job_dir.exists():
        await q.message.reply_text("job_id не найден.")  # type: ignore[union-attr]
        return

    if action == "tx":
        path = job_dir / "result.txt"
        if not path.exists():
            await q.message.reply_text("Расшифровка не найдена.")  # type: ignore[union-attr]
            return
        await send_file(q.message, path, caption=f"Расшифровка (job_id: {job_id})")  # type: ignore[arg-type]
        return

    if action == "unk":
        unknowns = get_unknowns(job_dir)
        if not unknowns:
            await q.message.reply_text("UNKNOWN-спикеры не обнаружены.")  # type: ignore[union-attr]
            return

        lines = ["Обнаружены неизвестные говорящие:"]
        for u in unknowns:
            if not isinstance(u, dict):
                continue
            unk_id = u.get("id")
            label = u.get("label")
            if unk_id:
                if label:
                    lines.append(f"- {unk_id} = {label}")
                else:
                    lines.append(f"- {unk_id}")

        lines.append("")
        lines.append("Чтобы назвать и запомнить голос, отправьте:")
        lines.append(f"/label {job_id} UNKNOWN_1 Иван")
        await q.message.reply_text("\n".join(lines))  # type: ignore[union-attr]
        return

    if action == "prot":
        await q.message.reply_text("Пересобираю протокол (Word)…")  # type: ignore[union-attr]
        await auto_protocol_and_send(q.message, job_id, job_dir)  # type: ignore[arg-type]
        return

    if action == "zip":
        await send_job_zip(q.message, job_id)  # type: ignore[arg-type]
        return


# ---------------- Message handlers ----------------
async def handle_menu_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    if msg is None or not msg.text:
        return
    t = msg.text.strip()

    if t == BTN_AUDIO:
        await msg.reply_text(
            "Пришлите voice/audio/document с аудио.\n"
            "Если файл большой — лучше прислать прямую ссылку (https://...).",
            reply_markup=start_keyboard(),
        )
        return

    if t == BTN_LINK:
        await msg.reply_text(
            "Пришлите прямую ссылку на скачивание аудио (https://...).\n"
            "После загрузки я автоматически пришлю Word-протокол.",
            reply_markup=start_keyboard(),
        )
        return

    if t == BTN_SPEAKERS:
        await cmd_speakers(update, context)
        return

    if t == BTN_HELP:
        await cmd_help(update, context)
        return

    # Fallback: treat as normal text (maybe contains URL)
    await handle_text(update, context)


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    if msg is None or not msg.text:
        return

    text = msg.text.strip()
    m = URL_RE.search(text)
    if m:
        await handle_url(update, context, m.group(1))
        return

    await msg.reply_text(
        "Отправьте аудио (voice/audio/document) или прямую ссылку (https://...).\n"
        "Протокол в Word придёт автоматически.\n\n"
        "Для справки: /help",
        reply_markup=start_keyboard(),
    )


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

    await msg.reply_text("Принял. Этап 1/2: скачиваю и распознаю…")

    try:
        await download_by_url(url, local_path, URL_MAX_DOWNLOAD_BYTES)
    except Exception as e:
        await msg.reply_text(f"Ошибка скачивания по ссылке: {e}")
        return

    try:
        async with PROCESS_LOCK:
            proc = await run_asr(local_path, DEFAULT_LANG, job_dir)
    except Exception as e:
        await msg.reply_text(f"Ошибка запуска распознавания: {e}")
        return
    finally:
        if DELETE_AUDIO_AFTER_PROCESS:
            try:
                local_path.unlink(missing_ok=True)  # type: ignore[attr-defined]
            except Exception:
                pass

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "no output")[-3500:]
        await msg.reply_text("Ошибка при обработке аудио. Лог ниже:")
        await msg.reply_text(err)
        return

    if AUTO_PROTOCOL_ALWAYS:
        await auto_protocol_and_send(msg, job_id, job_dir)
    else:
        await msg.reply_text(f"Готово. Job ID: {job_id}\nИспользуйте /protocol {job_id} чтобы получить Word.")


async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    if msg is None:
        return

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

        if (mtype.startswith("audio/")) or (fext in AUDIO_EXTS):
            ext = fext or ".m4a"
        else:
            await msg.reply_text("Похоже, это не аудиофайл. Пришлите voice/audio или аудио-документ (.mp3/.m4a/.wav...).")
            return
    else:
        await msg.reply_text("Это не аудиофайл. Пришлите voice/audio/document.")
        return

    file_size = getattr(tg_obj, "file_size", None)
    if file_size and file_size > TG_MAX_DOWNLOAD_BYTES:
        await msg.reply_text(
            f"Файл слишком большой для скачивания через Telegram: {file_size/1024/1024:.1f} MB.\n"
            "Пришлите прямую ссылку на скачивание (https://...), и я обработаю её."
        )
        return

    job_id = uuid.uuid4().hex
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    await msg.reply_text("Принял. Этап 1/2: скачиваю и распознаю…")

    try:
        tg_file = await context.bot.get_file(tg_obj.file_id)
        local_path = AUDIO_INBOX / f"{tg_obj.file_id}_{job_id}{ext}"
        await tg_file.download_to_drive(custom_path=str(local_path))
    except Exception as e:
        await msg.reply_text(f"Ошибка скачивания из Telegram: {e}")
        return

    try:
        async with PROCESS_LOCK:
            proc = await run_asr(local_path, DEFAULT_LANG, job_dir)
    except Exception as e:
        await msg.reply_text(f"Ошибка запуска распознавания: {e}")
        return
    finally:
        if DELETE_AUDIO_AFTER_PROCESS:
            try:
                local_path.unlink(missing_ok=True)  # type: ignore[attr-defined]
            except Exception:
                pass

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "no output")[-3500:]
        await msg.reply_text("Ошибка при обработке аудио. Лог ниже:")
        await msg.reply_text(err)
        return

    if AUTO_PROTOCOL_ALWAYS:
        await auto_protocol_and_send(msg, job_id, job_dir)
    else:
        await msg.reply_text(f"Готово. Job ID: {job_id}\nИспользуйте /protocol {job_id} чтобы получить Word.")


# ---------------- Main ----------------
def main() -> None:
    app = ApplicationBuilder().token(TG_BOT_TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("speakers", cmd_speakers))
    app.add_handler(CommandHandler("label", cmd_label))
    app.add_handler(CommandHandler("protocol", cmd_protocol))
    app.add_handler(CommandHandler("transcript", cmd_transcript))
    app.add_handler(CommandHandler("files", cmd_files))

    # Inline buttons
    app.add_handler(CallbackQueryHandler(on_callback))

    # Menu button texts
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_menu_text))

    # Audio/doc
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO | filters.Document.ALL, handle_audio))

    logger.info("Telegram ASR bot started (polling).")
    app.run_polling()


if __name__ == "__main__":
    main()
