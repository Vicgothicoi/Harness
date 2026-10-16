"""Workspace file tools and preference writes."""
from __future__ import annotations

from pathlib import Path

from harness import config
from harness.tools.result import ToolResult, result_error, result_ok
from harness.tools.workspace import (
    LIST_FILES_MAX,
    RESULT_HARD_CAP,
    SKILL_FILE_HARD_CAP,
    hard_cap,
    resolve_path,
)

def _coerce_start_line(start_line: int | str | None) -> int | str:
    """Return a 1-based start line, or an error string."""
    if start_line is None or start_line == "":
        return 1
    try:
        value = int(start_line)
    except (TypeError, ValueError):
        return f"[error] start_line must be an integer, got {start_line!r}"
    if value < 1:
        return f"[error] start_line must be >= 1, got {value}"
    return value


def read_file(path: str, start_line: int | str = 1) -> ToolResult:
    """Read a workspace file, optionally starting at a 1-based line."""
    parsed = _coerce_start_line(start_line)
    if isinstance(parsed, str):
        return result_error(parsed)
    start_line = parsed

    p = resolve_path(path)
    if not p.exists():
        return result_error(f"[error] File not found: {path}")
    try:
        with p.open("r", encoding="utf-8", errors="replace") as f:
            if start_line > 1:
                for skipped in range(1, start_line):
                    if f.readline() == "":
                        return result_error(
                            f"[error] start_line={start_line} is past end of file "
                            f"({path} has {skipped - 1} lines)"
                        )
            content = f.read(RESULT_HARD_CAP)
            if start_line > 1 and content == "":
                return result_error(
                    f"[error] start_line={start_line} is past end of file "
                    f"({path} has {start_line - 1} lines)"
                )
            return result_ok(content)
    except OSError as e:
        return result_error(f"[error] {e}")

def read_skill_file(path: str) -> ToolResult:
    """Read a file from the skills directory (outside workspace). Path must be relative to project root."""
    project_root = config.REPO_ROOT
    p = (project_root / path).resolve()
    # Must stay within the skills directory
    skills_dir = (project_root / "skills").resolve()
    if not str(p).startswith(str(skills_dir)):
        return result_error(f"[error] Path must be inside skills/ directory: {path}")
    if not p.exists():
        return result_error(f"[error] Skill file not found: {path}")
    return result_ok(
        hard_cap(
            p.read_text(encoding="utf-8", errors="replace"),
            SKILL_FILE_HARD_CAP,
        )
    )

def write_file(path: str, content: str) -> ToolResult:
    if not path or not path.strip():
        return result_error("[error] Empty file path")
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return result_ok(f"Wrote {len(content)} chars to {path}")

def remember_preference(key: str, value: str) -> ToolResult:
    """Explicitly store a user preference in global long-term memory."""
    from harness.memory.long_term_memory import LongTermMemory

    key = (key or "").strip()
    value = (value or "").strip()
    if not key:
        return result_error("[error] remember_preference requires a non-empty key")
    if not value:
        return result_error("[error] remember_preference requires a non-empty value")
    ltm = LongTermMemory.load()
    ltm.set_preference(key, value)
    path = ltm.save()
    return result_ok(f"Saved preference {key}={value!r} to {path}")

def list_files(directory: str = ".") -> ToolResult:
    p = resolve_path(directory)
    if not p.is_dir():
        return result_error(f"[error] Not a directory: {directory}")
    entries = []
    ws = Path(config.WORKSPACE).resolve()
    for item in sorted(p.rglob("*")):
        if not item.is_file():
            continue
        rel = item.relative_to(ws)
        if any(part.startswith(".") for part in rel.parts):
            continue
        entries.append(str(rel))
    if not entries:
        return result_ok("(empty)")
    if len(entries) > LIST_FILES_MAX:
        return result_ok(
            "\n".join(entries[:LIST_FILES_MAX])
            + f"\n...[list stopped at {LIST_FILES_MAX} files]"
        )
    return result_ok("\n".join(entries))
