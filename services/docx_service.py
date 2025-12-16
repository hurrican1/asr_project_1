from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from protocol_docx import build_protocol_docx as _build_protocol_docx


def build_protocol_docx(protocol_data: Dict[str, Any], output_path: Path) -> None:
    """
    Wrapper around protocol_docx.build_protocol_docx, kept as a separate service
    for cleaner imports.
    """
    _build_protocol_docx(protocol_data, output_path)
