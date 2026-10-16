"""Workspace path checks and shared result-size limits."""
from __future__ import annotations

from pathlib import Path

from harness import config

RESULT_HARD_CAP = int(getattr(config, "TOOL_RESULT_HARD_CAP", 1_000_000))
LIST_FILES_MAX = 200
SKILL_FILE_HARD_CAP = 60_000

def hard_cap(text: str, cap: int = RESULT_HARD_CAP) -> str:
    if len(text) <= cap:
        return text
    return text[:cap]

def resolve_path(path: str) -> Path:
    """Resolve a relative path inside the workspace. Prevent escaping."""
    p = Path(config.WORKSPACE, path).resolve()
    ws = Path(config.WORKSPACE).resolve()
    if not str(p).startswith(str(ws)):
        raise ValueError(f"Path escapes workspace: {path}")
    return p
