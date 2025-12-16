from __future__ import annotations

import zipfile
from pathlib import Path
from typing import List, Optional

import config


DEFAULT_ARTIFACTS = ["result.txt", "result.json", "protocol.docx", "protocol_debug.json"]


def list_existing_artifacts(job_dir: Path, *, names: Optional[List[str]] = None) -> List[Path]:
    names = names or DEFAULT_ARTIFACTS
    out: List[Path] = []
    for n in names:
        p = job_dir / n
        if p.exists():
            out.append(p)
    return out


def create_job_zip(job_id: str) -> Path:
    job_dir = config.JOBS_DIR / job_id
    if not job_dir.exists():
        raise RuntimeError("job_id не найден")

    zip_path = job_dir / f"job_{job_id}.zip"
    files = list_existing_artifacts(job_dir)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, arcname=p.name)

    return zip_path
