from __future__ import annotations

import asyncio
import subprocess
import uuid
from pathlib import Path
from typing import Tuple

import config


def create_job() -> Tuple[str, Path]:
    job_id = uuid.uuid4().hex
    job_dir = config.JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_id, job_dir


def result_txt_path(job_dir: Path) -> Path:
    return job_dir / "result.txt"


def result_json_path(job_dir: Path) -> Path:
    return job_dir / "result.json"


async def run_asr(local_audio_path: Path, *, lang: str, job_dir: Path) -> subprocess.CompletedProcess:
    if not config.RUN_SCRIPT.exists():
        raise RuntimeError(f"run script not found: {config.RUN_SCRIPT}")

    cmd = [str(config.RUN_SCRIPT), str(local_audio_path), lang, str(job_dir)]

    return await asyncio.to_thread(
        subprocess.run,
        cmd,
        cwd=str(config.BASE_DIR),
        capture_output=True,
        text=True,
    )
