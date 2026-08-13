"""
Long-term memory — cross-task preferences and reusable lessons.

Stored globally (not inside a single workspace). This module supports:
  - persisting user preferences (explicit tool / API)
  - learning patterns after a harness run completes
  - injecting preferences into the working context

Pattern retrieval (RAG / keyword top-k) is intentionally NOT implemented yet.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import config
from memory.inject import upsert_marked_block
from memory.project_memory import PROJECT_MARKER
from memory.state_memory import STATE_MARKER

log = logging.getLogger("harness")

LONG_TERM_MARKER = "[LONG-TERM MEMORY]"


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def default_memory_dir() -> Path:
    raw = getattr(config, "LONG_TERM_MEMORY_DIR", "") or ""
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".harness" / "memory"


@dataclass
class LongTermMemory:
    schema_version: int = 1
    updated_at: str = ""
    user_preferences: dict[str, str] = field(default_factory=dict)
    patterns: list[dict] = field(default_factory=list)
    anti_patterns: list[str] = field(default_factory=list)

    @classmethod
    def path(cls, memory_dir: str | Path | None = None) -> Path:
        base = Path(memory_dir) if memory_dir else default_memory_dir()
        return base / config.LONG_TERM_MEMORY_FILE

    @classmethod
    def load(cls, memory_dir: str | Path | None = None) -> "LongTermMemory":
        path = cls.path(memory_dir)
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return cls()
            return cls.from_dict(data)
        except Exception as e:
            log.warning(f"Failed to load long-term memory: {e}")
            return cls()

    def save(self, memory_dir: str | Path | None = None) -> Path:
        path = self.path(memory_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = _now_iso()
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return path

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "LongTermMemory":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in data.items() if k in known}
        if "user_preferences" in kwargs and not isinstance(kwargs["user_preferences"], dict):
            kwargs["user_preferences"] = {}
        if "patterns" in kwargs and not isinstance(kwargs["patterns"], list):
            kwargs["patterns"] = []
        if "anti_patterns" in kwargs and not isinstance(kwargs["anti_patterns"], list):
            kwargs["anti_patterns"] = []
        # Normalize preference values to str
        if "user_preferences" in kwargs:
            kwargs["user_preferences"] = {
                str(k): str(v) for k, v in kwargs["user_preferences"].items()
            }
        return cls(**kwargs)

    def set_preference(self, key: str, value: str) -> None:
        key = (key or "").strip()
        value = (value or "").strip()
        if not key:
            raise ValueError("preference key must be non-empty")
        self.user_preferences[key] = value

    def add_patterns(self, patterns: list[dict], anti_patterns: list[str] | None = None) -> int:
        """Append new pattern cards; returns number of patterns added."""
        added = 0
        existing_lessons = {
            str(p.get("lesson", "")).strip().lower()
            for p in self.patterns
            if isinstance(p, dict)
        }
        for p in patterns or []:
            if not isinstance(p, dict):
                continue
            lesson = str(p.get("lesson", "")).strip()
            trigger = str(p.get("trigger", "")).strip()
            if not lesson or not trigger:
                continue
            if lesson.lower() in existing_lessons:
                continue
            entry = {
                "id": str(p.get("id") or f"ltm-{uuid4().hex[:10]}"),
                "trigger": trigger,
                "lesson": lesson,
                "anti_pattern": str(p.get("anti_pattern", "")).strip(),
                "confidence": float(p.get("confidence", 0.6) or 0.6),
                "source_task": str(p.get("source_task", "")).strip(),
                "tags": [str(t).strip() for t in (p.get("tags") or []) if str(t).strip()][
                    :8
                ],
                "created_at": str(p.get("created_at") or _now_iso()),
            }
            self.patterns.append(entry)
            existing_lessons.add(lesson.lower())
            added += 1

        if anti_patterns:
            for ap in anti_patterns:
                s = str(ap).strip()
                if s and s not in self.anti_patterns:
                    self.anti_patterns.append(s)
            self.anti_patterns = self.anti_patterns[-50:]

        # Cap growth
        self.patterns = self.patterns[-200:]
        return added

    def preferences_context_block(self, max_chars: int | None = None) -> str:
        """
        Projection for working memory — preferences only (no RAG over patterns yet).
        """
        if max_chars is None:
            max_chars = getattr(config, "LONG_TERM_PREFS_MAX_CHARS", 800)
        if not self.user_preferences:
            return ""
        lines = [LONG_TERM_MARKER, "User preferences:"]
        for k, v in list(self.user_preferences.items())[:20]:
            lines.append(f"  - {k}: {v}")
        # Note that patterns exist but are not retrieved yet
        if self.patterns:
            lines.append(
                f"(Stored lessons: {len(self.patterns)} — retrieval not enabled yet)"
            )
        body = "\n".join(lines)
        if len(body) > max_chars:
            body = body[: max_chars - 20] + "\n...(truncated)"
        return body


def inject_long_term_preferences(
    messages: list[dict], memory: LongTermMemory | None = None
) -> list[dict]:
    memory = memory if memory is not None else LongTermMemory.load()
    block = memory.preferences_context_block()
    return upsert_marked_block(
        messages,
        LONG_TERM_MARKER,
        block,
        after_markers=(STATE_MARKER, PROJECT_MARKER),
    )


_LEARN_INSTRUCTION = """\
You extract reusable cross-task lessons from a completed coding-agent run.
Return ONLY JSON (no fences) with:
  "patterns": [
    {
      "trigger": "short phrase describing when this applies",
      "lesson": "actionable reusable advice (no repo-specific file paths)",
      "anti_pattern": "what not to do",
      "confidence": 0.0-1.0,
      "tags": ["tag1", "tag2"]
    }
  ],
  "anti_patterns": ["short anti-pattern strings"],
  "preferences": {"optional_key": "optional_value"}

Rules:
- Prefer 1-5 high-quality patterns; skip trivial or task-specific details.
- Do NOT include absolute paths or one-off filenames unless universally meaningful.
- If the run failed or there is nothing reusable, return {"patterns": [], "anti_patterns": []}.
"""


def learn_from_task(
    user_prompt: str,
    llm_call,
    *,
    passed: bool,
    score_history: list[float] | None = None,
    workspace: str | Path | None = None,
    memory_dir: str | Path | None = None,
) -> LongTermMemory:
    """
    After harness completion, summarize transferable lessons into long-term memory.
    Still runs on failure but with lower expected yield (model may return empty).
    """
    from memory.json_parse import parse_json_object
    from memory.project_memory import ProjectMemory
    from pathlib import Path

    ws = Path(workspace or config.WORKSPACE)
    ltm = LongTermMemory.load(memory_dir)
    pm = ProjectMemory.load(ws)

    feedback = ""
    fb_path = ws / config.FEEDBACK_FILE
    if fb_path.exists():
        feedback = fb_path.read_text(encoding="utf-8", errors="replace")[:4000]

    progress = ""
    pr_path = ws / config.PROGRESS_FILE
    if pr_path.exists():
        progress = pr_path.read_text(encoding="utf-8", errors="replace")[:2000]

    raw = llm_call(
        [
            {"role": "system", "content": _LEARN_INSTRUCTION},
            {
                "role": "user",
                "content": (
                    f"Task: {user_prompt}\n"
                    f"Passed: {passed}\n"
                    f"Score history: {score_history or []}\n\n"
                    f"Project memory:\n{json.dumps(pm.to_dict(), ensure_ascii=False)[:6000]}\n\n"
                    f"Progress:\n{progress}\n\n"
                    f"Feedback:\n{feedback}"
                ),
            },
        ]
    )
    data = parse_json_object(raw) or {}
    patterns = data.get("patterns") if isinstance(data.get("patterns"), list) else []
    # Attach source_task
    for p in patterns:
        if isinstance(p, dict):
            p.setdefault("source_task", (user_prompt or "")[:200])
            if not passed:
                # Down-weight lessons from failed runs
                try:
                    p["confidence"] = min(float(p.get("confidence", 0.4)), 0.45)
                except (TypeError, ValueError):
                    p["confidence"] = 0.4

    anti = data.get("anti_patterns") if isinstance(data.get("anti_patterns"), list) else []
    added = ltm.add_patterns(patterns, anti)

    prefs = data.get("preferences")
    if isinstance(prefs, dict):
        for k, v in prefs.items():
            if str(k).strip() and str(v).strip():
                # Don't overwrite existing user prefs from auto-learn
                key = str(k).strip()
                if key not in ltm.user_preferences:
                    ltm.user_preferences[key] = str(v).strip()

    ltm.save(memory_dir)
    log.info(f"Long-term memory updated (+{added} patterns, prefs={len(ltm.user_preferences)})")
    return ltm
