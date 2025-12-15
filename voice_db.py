from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

DB_VERSION = 1
DEFAULT_MODEL_ID = "pyannote/wespeaker-voxceleb-resnet34-LM"


def _now_ts() -> int:
    return int(time.time())


def load_db(path: Path) -> Dict[str, Any]:
    """
    Формат:
    {
      "version": 1,
      "model_id": "...",
      "speakers": {
         "Иван": {"embeddings": [[...],[...]], "created_at": 123, "updated_at": 456},
         "Мария": {...}
      }
    }
    """
    if not path.exists():
        return {"version": DB_VERSION, "model_id": DEFAULT_MODEL_ID, "speakers": {}}

    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("version", DB_VERSION)
    data.setdefault("model_id", DEFAULT_MODEL_ID)
    if "speakers" not in data or not isinstance(data["speakers"], dict):
        data["speakers"] = {}
    return data


def save_db(path: Path, db: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(db, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def cosine_sim(a: List[float], b: List[float]) -> float:
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(va) * np.linalg.norm(vb)) + 1e-8
    return float(np.dot(va, vb) / denom)


def match_speaker(
    db: Dict[str, Any], embedding: List[float], threshold: float
) -> Tuple[Optional[str], float]:
    """
    Возвращает (name, score). Если score < threshold -> (None, best_score).
    """
    best_name: Optional[str] = None
    best_score: float = -1.0

    speakers = db.get("speakers") or {}
    for name, rec in speakers.items():
        for e in (rec.get("embeddings") or []):
            try:
                score = cosine_sim(embedding, e)
            except Exception:
                continue
            if score > best_score:
                best_score = score
                best_name = name

    if best_name is None or best_score < threshold:
        return None, best_score
    return best_name, best_score


def add_voiceprint(
    db: Dict[str, Any], name: str, embedding: List[float], max_embeddings: int = 5
) -> None:
    """
    Добавляет voiceprint к speaker name. Храним последние max_embeddings.
    """
    name = name.strip()
    if not name:
        raise ValueError("Speaker name is empty.")

    speakers = db.setdefault("speakers", {})
    rec = speakers.setdefault(
        name, {"embeddings": [], "created_at": _now_ts(), "updated_at": _now_ts()}
    )
    rec["embeddings"].append(embedding)
    rec["embeddings"] = rec["embeddings"][-max_embeddings:]
    rec["updated_at"] = _now_ts()


def list_speakers(db: Dict[str, Any]) -> List[str]:
    return sorted((db.get("speakers") or {}).keys(), key=str.lower)
