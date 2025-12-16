from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import httpx


AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".opus", ".aac", ".flac", ".mp4", ".webm"}


def is_http_url(url: str) -> bool:
    try:
        u = urlparse(url)
        return u.scheme in ("http", "https") and bool(u.netloc)
    except Exception:
        return False


def ext_from_url(url: str) -> str:
    """
    Returns extension without dot, if it looks like an audio ext. Otherwise returns 'm4a'.
    """
    try:
        path = urlparse(url).path
        if "." in path:
            ext = "." + path.rsplit(".", 1)[-1].lower()
            if ext in AUDIO_EXTS:
                return ext.lstrip(".")
    except Exception:
        pass
    return "m4a"


async def download_by_url(url: str, dest: Path, *, max_bytes: int) -> int:
    """
    Streams a URL into dest with size limit. Returns downloaded bytes.
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
