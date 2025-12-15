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
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx
from telegram import InputFile, Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from voice_db import add_voiceprint, list_speakers, load_db, save_db

# Optional: only required for /protocol
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


def env_bool(key: str, default: bool = False) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


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

MAX_TELEGRAM_TEXT = int(os.getenv("MAX_TELEGRAM_TEXT", "3800"))

TG_MAX_DOWNLOAD_BYTES = int(os.getenv("TG_MAX_DOWNLOAD_MB", "20")) * 1024 * 1024
URL_MAX_DOWNLOAD_BYTES = int(os.getenv("URL_MAX_DOWNLOAD_MB", "2048")) * 1024 * 1024  # default 2GB

DELETE_AUDIO_AFTER_PROCESS = env_bool("DELETE_AUDIO_AFTER_PROCESS", default=False)

PROCESS_LOCK = asyncio.Lock()     # One GPU -> one ASR job at a time
VOICE_DB_LOCK = asyncio.Lock()    # Protect voice_db.json writes
PROTOCOL_LOCK = asyncio.Lock()    # Avoid parallel /protocol (cost + rate limits)

URL_RE = re.compile(r"(https?://\S+)", re.IGNORECASE)


# ---------------- Protocol prompts ----------------
# 1) Extraction prompt (small output JSON for each chunk)
EXTRACT_INSTRUCTIONS = """
Роль модели:
Ты — корпоративный секретарь и PMO-аналитик.

Задача:
Из входного фрагмента транскрипта извлеки факты по встрече. Никаких домыслов.
Если информация не указана явно — используй значение "не указано" или не добавляй элемент в список.

Вход:
Фрагмент транскрипта (возможны ошибки ASR). Реплики размечены говорящими (именами или UNKNOWN_*).

Правила:
- Не придумывай сроки/ответственных/цифры.
- В "decisions" добавляй только явные решения (сделаем/утвердили/решили).
- В "action_items" добавляй только явные поручения. Если нет ответственного/срока — ставь "не указан".
- Пиши кратко и делово, без лишних пояснений.
- Верни СТРОГО валидный JSON-объект. Никаких комментариев, текста, Markdown, ```.

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

Подсказка:
- agenda: 1–6 пунктов на фрагмент (если есть).
- key_points: 2–6 пунктов на фрагмент (если есть).
""".strip()

# 2) Final protocol template (your strict format)
PROTOCOL_TEMPLATE = """
ПРОТОКОЛ СОВЕЩАНИЯ
1. Общая информация

Дата: {date}
Время: {time}
Формат: {format}
Тема: {topic}
Участники:
{participants}

2. Повестка (что обсуждали)
{agenda}

3. Ключевые тезисы обсуждения (кратко)
{key_points}

4. Принятые решения
{decisions}

5. Задачи и поручения (Action Items)
№ | Задача | Ответственный | Срок | Критерий готовности / результат
{action_items_table}

6. Риски и блокеры
{risks}

7. Открытые вопросы
{open_questions}

8. Следующая встреча

Дата/время: {next_datetime}
Предварительная тема: {next_topic}
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
        await msg.reply_document(document=InputFile(f, filename=path.name), caption=caption)


async def reply_text_or_file(msg, text: str, *, file_path: Optional[Path] = None, caption: Optional[str] = None) -> None:
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
            if tmp.exists():
                tmp.unlink()
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


# ---------------- Protocol (Map-Reduce) ----------------
TS_RE = re.compile(r"^\[\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}\]\s*", re.M)


def strip_timestamps(transcript: str) -> str:
    # Remove leading "[.. --> ..]" at each line start to save tokens
    return TS_RE.sub("", transcript).strip()


def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def estimate_tokens(text: str, chars_per_token: float) -> int:
    """
    Conservative token estimator. Smaller chars_per_token => larger estimate => safer.
    Default chars_per_token ~ 3.0 for Cyrillic safety.
    """
    text = text or ""
    if not text:
        return 0
    return int((len(text) / max(chars_per_token, 1.5)) + 1)


def chunk_by_lines(text: str, *, max_input_tokens_est: int, chars_per_token: float) -> List[str]:
    """
    Chunk transcript by lines (speaker lines) to keep structure.
    """
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
    """
    Robust JSON parsing for model outputs (in case it accidentally adds extra chars).
    """
    raw = (text or "").strip()

    # Remove fenced code blocks if any
    raw = raw.replace("```json", "").replace("```", "").strip()

    # Try direct
    try:
        return json.loads(raw)
    except Exception:
        pass

    # Extract substring from first "{" to last "}"
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        sub = raw[start : end + 1]
        return json.loads(sub)

    raise ValueError("Model returned non-JSON output")


class TokenRateLimiter:
    """
    Simple TPM limiter using estimated tokens over the last 60 seconds.

    This helps avoid 429 TPM bursts for long multi-chunk protocols.
    """
    def __init__(self, tpm_limit: int):
        self.tpm_limit = max(1, int(tpm_limit))
        self.window_sec = 60.0
        self.events: Deque[Tuple[float, int]] = deque()

    def _prune(self, now: float) -> None:
        while self.events and (now - self.events[0][0]) > self.window_sec:
            self.events.popleft()

    def used(self, now: float) -> int:
        self._prune(now)
        return sum(t for _, t in self.events)

    def wait_for_budget(self, tokens_needed: int) -> None:
        tokens_needed = max(0, int(tokens_needed))
        while True:
            now = time.monotonic()
            self._prune(now)
            used = sum(t for _, t in self.events)
            if used + tokens_needed <= self.tpm_limit:
                self.events.append((now, tokens_needed))
                return

            # Need to wait until enough old tokens expire
            if not self.events:
                # should not happen, but just in case
                time.sleep(1.0)
                continue

            oldest_ts, oldest_tokens = self.events[0]
            wait_s = (oldest_ts + self.window_sec) - now
            wait_s = max(0.2, min(wait_s, 20.0))
            time.sleep(wait_s)


def openai_call_with_backoff(
    client: "OpenAI",
    *,
    model: str,
    instructions: str,
    input_text: str,
    max_output_tokens: int,
    temperature: float,
    store: bool,
    limiter: TokenRateLimiter,
    chars_per_token: float,
    max_retries: int = 8,
) -> str:
    """
    Synchronous call with:
    - local TPM throttling (estimated)
    - retry on 429 with exponential backoff + jitter
    """
    # Conservative estimate includes instructions + input + output
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
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                store=store,
            )
            return (getattr(resp, "output_text", "") or "").strip()

        except Exception as e:
            msg = str(e)
            # Retry only on likely rate-limit signals
            if ("429" in msg) or ("rate_limit" in msg) or ("TPM" in msg) or ("tokens per min" in msg):
                sleep_s = min((2 ** attempt) + random.random(), 30.0)
                time.sleep(sleep_s)
                continue
            raise

    raise RuntimeError("OpenAI rate limit: retries exhausted")


def normalize_extract_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure keys exist and types are sane.
    """
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
        context = normalize_space(dct.get("context", ""))
        if decision:
            out["decisions"].append({"decision": decision, "context": context or "не указано"})

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
    dt = normalize_space(nm.get("datetime", "")) or "не указано"
    tp = normalize_space(nm.get("topic", "")) or "не указано"
    out["next_meeting"] = {"datetime": dt, "topic": tp}

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

    merged["agenda"] = dedup_str_list(merged["agenda"])
    merged["key_points"] = dedup_str_list(merged["key_points"])
    merged["numbers_facts"] = dedup_str_list(merged["numbers_facts"])
    merged["open_questions"] = dedup_str_list(merged["open_questions"])

    # Mix numbers/facts into key points (no extra sections allowed)
    merged["key_points"] = dedup_str_list(merged["key_points"] + merged["numbers_facts"])

    # Apply caps to keep protocol compact
    merged["agenda"] = merged["agenda"][:12]
    merged["key_points"] = merged["key_points"][:12]
    merged["decisions"] = merged["decisions"][:30]
    merged["action_items"] = merged["action_items"][:50]
    merged["risks"] = merged["risks"][:20]
    merged["open_questions"] = merged["open_questions"][:20]

    return merged


def build_protocol_text(
    merged: Dict[str, Any],
    *,
    participants: List[str],
    topic: Optional[str],
) -> str:
    # 1) General info (we do not invent)
    date = "не указано"
    time_s = "не указано"
    fmt = "не указано"
    topic_s = normalize_space(topic) if topic else "не указано"

    participants_lines = participants if participants else ["не указано"]
    participants_block = "\n".join(participants_lines)

    # 2) Agenda
    agenda_list = merged.get("agenda") or []
    agenda_block = "\n".join(agenda_list) if agenda_list else "не указано"

    # 3) Key points
    kp_list = merged.get("key_points") or []
    key_points_block = "\n".join(kp_list) if kp_list else "не указано"

    # 4) Decisions
    dec_list = merged.get("decisions") or []
    if not dec_list:
        decisions_block = "Решения по итогам встречи не зафиксированы."
    else:
        parts = []
        for d in dec_list:
            decision = normalize_space(d.get("decision", "")) or "не указано"
            context = normalize_space(d.get("context", "")) or "не указано"
            parts.append(f"Решение: {decision}\nОснование/контекст: {context}")
        decisions_block = "\n\n".join(parts)

    # 5) Action items table
    ai_list = merged.get("action_items") or []
    if not ai_list:
        action_table = "— | Задачи не зафиксированы | — | — | —"
    else:
        rows = []
        for i, a in enumerate(ai_list, start=1):
            task = normalize_space(a.get("task", "")) or "не указано"
            owner = normalize_space(a.get("owner", "")) or "не указан"
            due = normalize_space(a.get("due", "")) or "не указан"
            done = normalize_space(a.get("done_criteria", "")) or "не указано"
            rows.append(f"{i} | {task} | {owner} | {due} | {done}")
        action_table = "\n".join(rows)

    # 6) Risks
    risks_list = merged.get("risks") or []
    if not risks_list:
        risks_block = "Риски и блокеры не зафиксированы."
    else:
        lines = []
        for r in risks_list:
            risk = normalize_space(r.get("risk", "")) or "не указано"
            mit = normalize_space(r.get("mitigation", "")) or "не указано"
            lines.append(f"{risk} — {mit}")
        risks_block = "\n".join(lines)

    # 7) Open questions
    oq_list = merged.get("open_questions") or []
    if not oq_list:
        open_questions_block = "Открытые вопросы не зафиксированы."
    else:
        open_questions_block = "\n".join(oq_list)

    # 8) Next meeting
    nm = merged.get("next_meeting") or {}
    next_dt = normalize_space(nm.get("datetime", "")) or "не указано"
    next_tp = normalize_space(nm.get("topic", "")) or "не указано"

    return PROTOCOL_TEMPLATE.format(
        date=date,
        time=time_s,
        format=fmt,
        topic=topic_s,
        participants=participants_block,
        agenda=agenda_block,
        key_points=key_points_block,
        decisions=decisions_block,
        action_items_table=action_table,
        risks=risks_block,
        open_questions=open_questions_block,
        next_datetime=next_dt,
        next_topic=next_tp,
    ).strip()


def generate_protocol_map_reduce(
    transcript_txt: str,
    *,
    participants: List[str],
    topic: Optional[str],
) -> Tuple[str, Dict[str, Any]]:
    """
    Returns: (protocol_text, debug_payload)
    debug_payload can be optionally saved for inspection.
    """
    if OpenAI is None:
        raise RuntimeError("Package 'openai' is not installed. Install: pip install openai")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env on the server.")

    model = os.getenv("PROTOCOL_MODEL", "gpt-4.1")

    # Conservative estimation & pacing settings
    chars_per_token = float(os.getenv("EST_CHARS_PER_TOKEN", "3.0"))  # smaller => safer
    tpm_limit = int(os.getenv("OPENAI_TPM_LIMIT", "30000"))           # your org limit from error
    limiter = TokenRateLimiter(tpm_limit)

    chunk_tokens = int(os.getenv("PROTOCOL_CHUNK_TOKENS", "9000"))    # input tokens estimate per chunk
    extract_out_tokens = int(os.getenv("PROTOCOL_EXTRACT_OUT_TOKENS", "900"))
    temperature = float(os.getenv("PROTOCOL_EXTRACT_TEMPERATURE", "0.0"))

    client = OpenAI()

    clean = strip_timestamps(transcript_txt)
    if not clean:
        raise RuntimeError("Transcript is empty after preprocessing.")

    chunks = chunk_by_lines(clean, max_input_tokens_est=chunk_tokens, chars_per_token=chars_per_token)
    extracts: List[Dict[str, Any]] = []

    for idx, ch in enumerate(chunks, start=1):
        # Add light header to help chunk ordering, minimal extra tokens
        input_text = f"ЧАСТЬ {idx}/{len(chunks)}\n{ch}"

        out = openai_call_with_backoff(
            client,
            model=model,
            instructions=EXTRACT_INSTRUCTIONS,
            input_text=input_text,
            max_output_tokens=extract_out_tokens,
            temperature=temperature,
            store=False,
            limiter=limiter,
            chars_per_token=chars_per_token,
        )

        payload = parse_json_robust(out)
        extracts.append(normalize_extract_payload(payload))

    merged = merge_extracts(extracts)
    protocol_text = build_protocol_text(merged, participants=participants, topic=topic)

    debug = {
        "chunks": len(chunks),
        "extracts": extracts,
        "merged": merged,
        "participants": participants,
        "topic": topic or "не указано",
        "model": model,
        "tpm_limit": tpm_limit,
        "chunk_tokens_est": chunk_tokens,
        "chars_per_token": chars_per_token,
    }
    return protocol_text, debug


# ---------------- Command handlers ----------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    if msg is None:
        return

    await msg.reply_text(
        "ASR бот запущен.\n\n"
        "Что умеет:\n"
        "• Принимает voice/audio/document (до ~20MB) или прямую ссылку (http/https).\n"
        "• Делает транскрипт + диаризацию.\n"
        "• Если есть UNKNOWN_* — можно подписать голоса командой /label.\n"
        "• /speakers — список известных голосов.\n"
        "• /protocol <job_id> [тема] — протокол совещания (поддерживает длинные транскрипты).\n\n"
        "Примеры:\n"
        "/label <job_id> UNKNOWN_1 Иван\n"
        "/protocol <job_id> Протокол встречи по проекту"
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

    await msg.reply_text("Готовлю протокол (GPT)...")

    async with PROTOCOL_LOCK:
        try:
            protocol_text, debug_payload = await asyncio.to_thread(
                generate_protocol_map_reduce,
                transcript,
                participants=participants,
                topic=topic,
            )
        except Exception as e:
            await msg.reply_text(f"Ошибка генерации протокола: {e}")
            return

    protocol_path = job_dir / "protocol.txt"
    protocol_path.write_text(protocol_text, encoding="utf-8")

    # Optional debug file (turn on via env)
    if env_bool("PROTOCOL_SAVE_DEBUG", default=False):
        debug_path = job_dir / "protocol_debug.json"
        debug_path.write_text(json.dumps(debug_payload, ensure_ascii=False, indent=2), encoding="utf-8")

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
    finally:
        if DELETE_AUDIO_AFTER_PROCESS:
            try:
                if local_path.exists():
                    local_path.unlink()
            except Exception:
                pass

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
    finally:
        if DELETE_AUDIO_AFTER_PROCESS:
            try:
                if local_path.exists():
                    local_path.unlink()
            except Exception:
                pass

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
