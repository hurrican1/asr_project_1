from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import config
from voice_db import add_voiceprint, list_speakers, load_db, save_db


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def format_ts(seconds: float) -> str:
    try:
        ms = int(round(float(seconds) * 1000))
    except Exception:
        ms = 0
    s, ms = divmod(ms, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def render_segments_to_text(segments: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        sp = seg.get("speaker", "")
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
        lines.append(f"[{format_ts(start)} --> {format_ts(end)}] {sp}: {text}")
    return "\n".join(lines).strip()


def extract_participants(job_dir: Path) -> List[str]:
    payload = read_json(job_dir / "result.json") or {}
    segments = payload.get("segments") or []
    if not isinstance(segments, list):
        return []
    participants: List[str] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        sp = seg.get("speaker")
        if sp and sp not in participants:
            participants.append(sp)
    return participants


def get_unknowns(job_dir: Path) -> List[Dict[str, Any]]:
    """
    Return UNKNOWN speakers that are not assigned yet.
    Expected schema in result.json:
      unknown_speakers: [{id: "UNKNOWN_1", embedding: [...], label?: str, assigned_name?: str}, ...]
    """
    payload = read_json(job_dir / "result.json") or {}
    unknowns = payload.get("unknown_speakers") or []
    if not isinstance(unknowns, list):
        return []

    out: List[Dict[str, Any]] = []
    for u in unknowns:
        if not isinstance(u, dict):
            continue
        unk_id = u.get("id")
        if not unk_id or not str(unk_id).startswith("UNKNOWN"):
            continue
        if u.get("assigned_name"):
            continue
        out.append(u)
    return out


def get_speaker_snippets(job_dir: Path, speaker_id: str, *, limit: int = 3) -> List[str]:
    payload = read_json(job_dir / "result.json") or {}
    segments = payload.get("segments") or []
    if not isinstance(segments, list):
        return []

    snippets: List[str] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        if seg.get("speaker") != speaker_id:
            continue
        txt = (seg.get("text") or "").strip()
        if not txt:
            continue
        start = seg.get("start", 0.0)
        s = txt.replace("\n", " ")
        if len(s) > 140:
            s = s[:140].rstrip() + "…"
        snippets.append(f"[{format_ts(start)}] {s}")
        if len(snippets) >= limit:
            break
    return snippets


async def apply_label_to_job(job_id: str, unknown_id: str, name: str) -> Tuple[int, int]:
    """
    Save voiceprint to DB and update job result.json/result.txt.
    Returns: (changed_segments, remaining_unknown_count)
    """
    job_dir = config.JOBS_DIR / job_id
    result_json = job_dir / "result.json"
    if not result_json.exists():
        raise RuntimeError("result.json не найден для этого job")

    payload = read_json(result_json) or {}
    unknowns = payload.get("unknown_speakers") or []
    if not isinstance(unknowns, list):
        unknowns = []

    target = None
    for u in unknowns:
        if isinstance(u, dict) and u.get("id") == unknown_id:
            target = u
            break
    if not target:
        raise RuntimeError(f"В этом job нет {unknown_id}")

    emb = target.get("embedding")
    if emb is None:
        raise RuntimeError(f"У {unknown_id} нет embedding (возможно, слишком мало речи).")

    # Save in global DB
    async with config.VOICE_DB_LOCK:
        db = load_db(config.VOICE_DB_PATH)
        add_voiceprint(db, name, emb)
        save_db(config.VOICE_DB_PATH, db)

    # Update segments for this job
    segments = payload.get("segments") or []
    if not isinstance(segments, list):
        segments = []

    changed = 0
    for seg in segments:
        if isinstance(seg, dict) and seg.get("speaker") == unknown_id:
            seg["speaker"] = name
            changed += 1

    target["assigned_name"] = name

    payload["segments"] = segments
    payload["unknown_speakers"] = unknowns
    write_json(result_json, payload)

    # Rewrite result.txt so protocol rebuild uses updated speakers
    result_txt = job_dir / "result.txt"
    try:
        result_txt.write_text(render_segments_to_text(segments), encoding="utf-8")
    except Exception:
        pass

    remaining = len(get_unknowns(job_dir))
    return changed, remaining


def list_known_speaker_names() -> List[str]:
    db = load_db(config.VOICE_DB_PATH)
    return list_speakers(db)
