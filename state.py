from __future__ import annotations

import time
from typing import Any, Dict, List, Optional


def set_pending_label(
    user_data: Dict[str, Any],
    *,
    job_id: str,
    unknown_id: str,
    queue: Optional[List[str]] = None,
    total: Optional[int] = None,
    auto_protocol_after: bool = False,
    control_chat_id: Optional[int] = None,
    control_message_id: Optional[int] = None,
) -> None:
    q = list(queue) if queue else []
    if total is None:
        total = 1 + len(q)

    user_data["pending_label"] = {
        "job_id": job_id,
        "unknown_id": unknown_id,
        "queue": q,
        "total": int(total),
        "auto_protocol_after": bool(auto_protocol_after),
        "control_chat_id": control_chat_id,
        "control_message_id": control_message_id,
        "created": time.time(),
    }


def get_pending_label(user_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    x = user_data.get("pending_label")
    return x if isinstance(x, dict) else None


def clear_pending_label(user_data: Dict[str, Any]) -> None:
    user_data.pop("pending_label", None)


def is_pending_expired(pending: Dict[str, Any], ttl_sec: int) -> bool:
    try:
        created = float(pending.get("created") or 0.0)
    except Exception:
        created = 0.0
    if not created:
        return True
    return (time.time() - created) > float(ttl_sec)
