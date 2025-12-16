from __future__ import annotations

import json
import os
import random
import re
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import config

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


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


TS_RE = re.compile(
    r"^\[\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}\]\s*",
    re.M,
)


def strip_timestamps(transcript: str) -> str:
    return TS_RE.sub("", transcript or "").strip()


def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def estimate_tokens(text: str, chars_per_token: float) -> int:
    if not text:
        return 0
    return int((len(text) / max(chars_per_token, 1.5)) + 1)


def chunk_by_lines(text: str, *, max_input_tokens_est: int, chars_per_token: float) -> List[str]:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
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
    """
    Simple TPM limiter using estimated tokens over the last 60 seconds.
    Protects you from 429 TPM bursts.
    """

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
        "agenda": [normalize_space(str(s)) for s in as_list(payload.get("agenda")) if normalize_space(str(s))],
        "key_points": [normalize_space(str(s)) for s in as_list(payload.get("key_points")) if normalize_space(str(s))],
        "decisions": [],
        "action_items": [],
        "numbers_facts": [
            normalize_space(str(s)) for s in as_list(payload.get("numbers_facts")) if normalize_space(str(s))
        ],
        "risks": [],
        "open_questions": [
            normalize_space(str(s)) for s in as_list(payload.get("open_questions")) if normalize_space(str(s))
        ],
        "next_meeting": {"datetime": "не указано", "topic": "не указано"},
    }

    for d in as_list(payload.get("decisions")):
        dct = as_dict(d)
        decision = normalize_space(str(dct.get("decision", "")))
        context = normalize_space(str(dct.get("context", ""))) or "не указано"
        if decision:
            out["decisions"].append({"decision": decision, "context": context})

    for a in as_list(payload.get("action_items")):
        dct = as_dict(a)
        task = normalize_space(str(dct.get("task", "")))
        if not task:
            continue
        owner = normalize_space(str(dct.get("owner", ""))) or "не указан"
        due = normalize_space(str(dct.get("due", ""))) or "не указан"
        done = normalize_space(str(dct.get("done_criteria", ""))) or "не указано"
        out["action_items"].append({"task": task, "owner": owner, "due": due, "done_criteria": done})

    for r in as_list(payload.get("risks")):
        dct = as_dict(r)
        risk = normalize_space(str(dct.get("risk", "")))
        if not risk:
            continue
        mitigation = normalize_space(str(dct.get("mitigation", ""))) or "не указано"
        out["risks"].append({"risk": risk, "mitigation": mitigation})

    nm = as_dict(payload.get("next_meeting"))
    out["next_meeting"] = {
        "datetime": normalize_space(str(nm.get("datetime", ""))) or "не указано",
        "topic": normalize_space(str(nm.get("topic", ""))) or "не указано",
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

    merged: Dict[str, Any] = {
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
    topic: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Synchronous function:
      transcript -> map-reduce extract -> merged protocol_data + debug.
    Run it in a thread from asyncio code.
    """
    if OpenAI is None:
        raise RuntimeError("Package 'openai' is not installed. Install: pip install openai")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env on the server.")

    clean = strip_timestamps(transcript_txt)
    if not clean:
        raise RuntimeError("Transcript is empty after preprocessing.")

    client = OpenAI()
    limiter = TokenRateLimiter(config.OPENAI_TPM_LIMIT)

    chunks = chunk_by_lines(
        clean,
        max_input_tokens_est=config.PROTOCOL_CHUNK_TOKENS,
        chars_per_token=config.EST_CHARS_PER_TOKEN,
    )

    extracts: List[Dict[str, Any]] = []
    for idx, ch in enumerate(chunks, start=1):
        chunk_text = f"ЧАСТЬ {idx}/{len(chunks)}\n{ch}"
        out_text = openai_call_extract_json(
            client,
            model=config.PROTOCOL_MODEL,
            instructions=EXTRACT_INSTRUCTIONS,
            chunk_text=chunk_text,
            max_output_tokens=config.PROTOCOL_EXTRACT_OUT_TOKENS,
            temperature=config.PROTOCOL_EXTRACT_TEMPERATURE,
            limiter=limiter,
            chars_per_token=config.EST_CHARS_PER_TOKEN,
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
        "model": config.PROTOCOL_MODEL,
        "tpm_limit": config.OPENAI_TPM_LIMIT,
        "chunk_tokens_est": config.PROTOCOL_CHUNK_TOKENS,
        "extract_out_tokens": config.PROTOCOL_EXTRACT_OUT_TOKENS,
        "chars_per_token": config.EST_CHARS_PER_TOKEN,
        "merged": merged,
    }

    return protocol_data, debug
