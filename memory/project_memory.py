"""
Project memory — durable per-workspace knowledge for multi-round builds.

Stored as project_memory.json in the workspace root. 
Updated at the end of each harness build round (not tied to full-context compression).
"""
from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import config
from memory.inject import upsert_marked_block
from memory.state_memory import STATE_MARKER

log = logging.getLogger("harness")

PROJECT_MARKER = "[PROJECT MEMORY]"


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class ProjectMemory:
    schema_version: int = 1
    project_id: str = ""
    created_at: str = ""
    updated_at: str = ""
    source_prompt: str = ""
    tech_stack: list[str] = field(default_factory=list)
    architecture: str = ""
    key_files: dict[str, str] = field(default_factory=dict)
    decisions: list[dict] = field(default_factory=list)
    known_issues: list[str] = field(default_factory=list)
    verification_commands: list[str] = field(default_factory=list)
    round_summaries: list[dict] = field(default_factory=list)

    # --- persistence ---

    @classmethod
    def path(cls, workspace: str | Path | None = None) -> Path:
        ws = Path(workspace or config.WORKSPACE)
        return ws / config.PROJECT_MEMORY_FILE

    @classmethod
    def load(cls, workspace: str | Path | None = None) -> "ProjectMemory":
        path = cls.path(workspace)
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return cls()
            return cls.from_dict(data)
        except Exception as e:
            log.warning(f"Failed to load project memory: {e}")
            return cls()

    def save(self, workspace: str | Path | None = None) -> Path:
        path = self.path(workspace)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = _now_iso()
        if not self.created_at:
            self.created_at = self.updated_at
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return path

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ProjectMemory":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in data.items() if k in known}
        # Coerce types lightly
        if "key_files" in kwargs and not isinstance(kwargs["key_files"], dict):
            kwargs["key_files"] = {}
        for list_key in (
            "tech_stack",
            "decisions",
            "known_issues",
            "verification_commands",
            "round_summaries",
        ):
            if list_key in kwargs and not isinstance(kwargs[list_key], list):
                kwargs[list_key] = []
        return cls(**kwargs)

    # --- context projection ---

    def to_context_block(self, max_chars: int | None = None) -> str:
        limit = (
            max_chars
            if max_chars is not None
            else int(getattr(config, "PROJECT_CONTEXT_MAX_CHARS", 2000) or 2000)
        )

        if not self.architecture and not self.tech_stack and not self.key_files:
            if not self.source_prompt and not self.round_summaries:
                return ""

        lines = [PROJECT_MARKER]
        if self.source_prompt:
            prompt = self.source_prompt.strip()
            if len(prompt) > 200:
                prompt = prompt[:200] + "..."
            lines.append(f"Task: {prompt}")
        if self.tech_stack:
            lines.append("Stack: " + ", ".join(self.tech_stack[:12]))
        if self.architecture:
            arch = self.architecture.strip()
            if len(arch) > 400:
                arch = arch[:400] + "..."
            lines.append(f"Architecture: {arch}")
        if self.key_files:
            lines.append("Key files:")
            for path, role in list(self.key_files.items())[:12]:
                role_s = str(role).strip()
                if len(role_s) > 80:
                    role_s = role_s[:80] + "..."
                lines.append(f"  - {path}: {role_s}")
        if self.decisions:
            lines.append("Recent decisions:")
            for d in self.decisions[-4:]:
                if isinstance(d, dict):
                    what = str(d.get("what", "")).strip()
                    why = str(d.get("why", "")).strip()
                    item = what if not why else f"{what} ({why})"
                else:
                    item = str(d)
                if len(item) > 120:
                    item = item[:120] + "..."
                if item:
                    lines.append(f"  - {item}")
        if self.known_issues:
            lines.append("Open issues:")
            for issue in self.known_issues[-5:]:
                text = str(issue).strip()
                if len(text) > 100:
                    text = text[:100] + "..."
                lines.append(f"  - {text}")
        if self.round_summaries:
            last = self.round_summaries[-1]
            if isinstance(last, dict):
                r = last.get("round", "?")
                score = last.get("score")
                summary = str(last.get("summary", "")).strip()
                if len(summary) > 120:
                    summary = summary[:120] + "..."
                score_s = f" score={score}" if score is not None else ""
                lines.append(f"Last round: #{r}{score_s} — {summary}")

        body = "\n".join(lines)
        if len(body) > limit:
            body = body[: max(0, limit - 20)] + "\n...(truncated)"
        return body

    def merge_delta(self, delta: dict, round_num: int | None = None) -> None:
        """Merge a partial update dict into this memory (additive / overwrite fields)."""
        if not isinstance(delta, dict):
            return

        if isinstance(delta.get("tech_stack"), list):
            merged = list(self.tech_stack)
            for item in delta["tech_stack"]:
                s = str(item).strip()
                if s and s not in merged:
                    merged.append(s)
            self.tech_stack = merged[:30]

        if isinstance(delta.get("architecture"), str) and delta["architecture"].strip():
            self.architecture = delta["architecture"].strip()

        if isinstance(delta.get("key_files"), dict):
            for k, v in delta["key_files"].items():
                key = str(k).strip()
                if key:
                    self.key_files[key] = str(v).strip()

        if isinstance(delta.get("decisions"), list):
            for d in delta["decisions"]:
                if isinstance(d, dict) and d.get("what"):
                    entry = {
                        "what": str(d.get("what", "")).strip(),
                        "why": str(d.get("why", "")).strip(),
                        "when": str(d.get("when", "")).strip()
                        or (f"round-{round_num}" if round_num else ""),
                        "round": d.get("round", round_num),
                    }
                    # Avoid exact duplicate "what"
                    if not any(
                        isinstance(x, dict) and x.get("what") == entry["what"]
                        for x in self.decisions
                    ):
                        self.decisions.append(entry)
            self.decisions = self.decisions[-40:]

        if isinstance(delta.get("known_issues"), list):
            merged = list(self.known_issues)
            for issue in delta["known_issues"]:
                s = str(issue).strip()
                if s and s not in merged:
                    merged.append(s)
            self.known_issues = merged[-30:]

        if isinstance(delta.get("verification_commands"), list):
            merged = list(self.verification_commands)
            for cmd in delta["verification_commands"]:
                s = str(cmd).strip()
                if s and s not in merged:
                    merged.append(s)
            self.verification_commands = merged[:20]

        if isinstance(delta.get("round_summary"), dict):
            summary = dict(delta["round_summary"])
            if round_num is not None:
                summary.setdefault("round", round_num)
            self.round_summaries.append(summary)
            self.round_summaries = self.round_summaries[-20:]


def inject_project_summary(
    messages: list[dict], memory: ProjectMemory | None = None, max_chars: int | None = None
) -> list[dict]:
    memory = memory if memory is not None else ProjectMemory.load()
    block = memory.to_context_block(max_chars=max_chars)
    return upsert_marked_block(
        messages, PROJECT_MARKER, block, after_markers=(STATE_MARKER,)
    )


def seed_project_memory(user_prompt: str, workspace: str | Path | None = None) -> ProjectMemory:
    """Create an initial project_memory.json after planning / workspace setup."""
    ws = Path(workspace or config.WORKSPACE)
    pm = ProjectMemory.load(ws)
    if not pm.project_id:
        pm.project_id = ws.name
    if not pm.source_prompt:
        pm.source_prompt = (user_prompt or "").strip()
    if not pm.created_at:
        pm.created_at = _now_iso()
    pm.save(ws)
    return pm


def _workspace_snapshot(workspace: Path) -> str:
    """Collect a bounded text snapshot for LLM project-memory refresh."""
    parts: list[str] = []

    for name in (
        config.SPEC_FILE,
        config.PROGRESS_FILE,
        config.FEEDBACK_FILE,
        config.CONTRACT_FILE,
        config.HANDOFF_FILE,
    ):
        path = workspace / name
        if path.exists():
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
                parts.append(f"## {name}\n{text[:4000]}")
            except Exception:
                pass

    try:
        files = sorted(
            p.relative_to(workspace).as_posix()
            for p in workspace.rglob("*")
            if p.is_file()
            and not any(part.startswith(".") for part in p.relative_to(workspace).parts)
            and p.name not in {"project_memory.json"}
            and not p.name.startswith("_trace_")
        )
        parts.append("## Workspace files\n" + "\n".join(files[:80]))
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["git", "diff", "--stat", "HEAD"],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.stdout.strip():
            parts.append("## git diff --stat\n" + result.stdout.strip()[:2000])
    except Exception:
        pass

    return "\n\n".join(parts)[:20000]


_PROJECT_REFRESH_INSTRUCTION = """\
You maintain a structured project memory JSON for an autonomous coding agent.
Given the current project memory and a workspace snapshot, produce a DELTA
object (not the full memory) as JSON only (no markdown fences) with optional keys:
  "tech_stack": string[]
  "architecture": string
  "key_files": { "path": "one-line role", ... }
  "decisions": [ {"what": str, "why": str, "when": str}, ... ]
  "known_issues": string[]
  "verification_commands": string[]
  "round_summary": {"summary": str, "artifacts": string[], "score": number|null}

Rules:
- Do NOT invent files that are not in the snapshot.
- Prefer concrete paths and short phrases.
- round_summary.summary should describe what changed this round.
- If nothing new, return {"round_summary": {"summary": "no material changes", "artifacts": []}}.
"""


def refresh_project_memory(
    user_prompt: str,
    round_num: int,
    llm_call,
    score: float | None = None,
    workspace: str | Path | None = None,
) -> ProjectMemory:
    """
    LLM-merge project memory from workspace artifacts after a build round.
    """
    from memory.json_parse import parse_json_object

    ws = Path(workspace or config.WORKSPACE)
    pm = ProjectMemory.load(ws)
    if not pm.project_id:
        pm.project_id = ws.name
    if not pm.source_prompt:
        pm.source_prompt = (user_prompt or "").strip()

    snapshot = _workspace_snapshot(ws)
    current = json.dumps(pm.to_dict(), ensure_ascii=False, indent=2)[:8000]

    raw = llm_call(
        [
            {"role": "system", "content": _PROJECT_REFRESH_INSTRUCTION},
            {
                "role": "user",
                "content": (
                    f"Round: {round_num}\n"
                    f"Score this round (if any): {score}\n"
                    f"User prompt: {user_prompt}\n\n"
                    f"Current project_memory.json:\n{current}\n\n"
                    f"Workspace snapshot:\n{snapshot}"
                ),
            },
        ]
    )
    delta = parse_json_object(raw) or {}
    if score is not None and isinstance(delta.get("round_summary"), dict):
        delta["round_summary"].setdefault("score", score)
    elif score is not None and "round_summary" not in delta:
        delta["round_summary"] = {
            "summary": f"round {round_num} completed",
            "artifacts": [],
            "score": score,
        }

    pm.merge_delta(delta, round_num=round_num)
    pm.save(ws)
    log.info(
        f"Project memory refreshed (round={round_num}, "
        f"files={len(pm.key_files)}, decisions={len(pm.decisions)})"
    )
    return pm
