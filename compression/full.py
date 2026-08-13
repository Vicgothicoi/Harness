"""
Full-context compression (tier 4) — session reset via handoff.md.

When the window is too large or context anxiety is detected, serialize a
handoff document, clear the conversation, and resume from system + handoff.
Memory projections (STATE / PROJECT / LTM) are re-applied by the caller via
memory.working_memory.build_working_memory — not here.

Fork-subagent compression is intentionally not implemented here.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import config

log = logging.getLogger("harness")

HANDOFF_SYSTEM = (
    "You are creating a handoff document for the next agent session. "
    "The next session starts with a COMPLETELY EMPTY context window — "
    "it has zero memory of anything that happened here.\n\n"
    "Structure the handoff as:\n"
    "## Completed Work\n(what was built, with file paths)\n"
    "## Current State\n(what works, what's broken right now)\n"
    "## Next Steps\n(exactly what to do next, in order)\n"
    "## Key Decisions & Rationale\n(why things were done this way)\n"
    "## Known Issues\n(bugs, incomplete features, technical debt)\n\n"
    "Be thorough and specific — file paths, function names, error messages. "
    "The next session's success depends entirely on this document."
)


def messages_to_text(messages: list[dict]) -> str:
    """Flatten messages into readable text for summarization / handoff."""
    parts = []
    for msg in messages:
        role = msg.get("role", "?")
        content = msg.get("content") or ""
        if isinstance(content, list):
            content = " ".join(
                block.get("text", "") for block in content if isinstance(block, dict)
            )
        if content:
            parts.append(f"[{role}] {content[:3000]}")
        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {})
            parts.append(
                f"[tool_call] {fn.get('name', '?')}({fn.get('arguments', '')[:500]})"
            )
    return "\n".join(parts)


def create_checkpoint(messages: list[dict], llm_call) -> str:
    """
    Serialize current session into a structured handoff document.
    Persists to handoff.md (NOT progress.md — that file is owned by state memory).
    Returns the checkpoint text.
    """
    text = messages_to_text(messages)
    checkpoint = llm_call(
        [
            {"role": "system", "content": HANDOFF_SYSTEM},
            {"role": "user", "content": text},
        ]
    )

    handoff_path = Path(config.WORKSPACE) / config.HANDOFF_FILE
    handoff_path.parent.mkdir(parents=True, exist_ok=True)
    handoff_path.write_text(checkpoint, encoding="utf-8")
    log.info(f"Full-compress checkpoint written to {config.HANDOFF_FILE}")

    return checkpoint


def restore_from_checkpoint(
    checkpoint: str,
    system_prompt: str,
    task_board=None,
) -> list[dict]:
    """
    Build a fresh message list from handoff content (+ optional git diff).

    `task_board` is accepted for API compatibility but is not injected here;
    callers should run build_working_memory afterward.
    """
    del task_board  # unused — working memory builder owns projections

    git_context = ""
    try:
        result = subprocess.run(
            "git diff --stat HEAD~5 2>/dev/null || git log --oneline -5 2>/dev/null",
            shell=True,
            cwd=config.WORKSPACE,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.stdout.strip():
            git_context = (
                f"\n\nRecent code changes:\n```\n{result.stdout.strip()[:2000]}\n```"
            )
    except Exception:
        pass

    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "You are resuming an in-progress project. Your previous session's "
                "context was reset to give you a clean slate.\n\n"
                "Here is the handoff document from the previous session:\n\n"
                + checkpoint
                + git_context
                + "\n\nContinue from where the previous session left off. "
                "Do NOT redo work that's already completed."
            ),
        },
    ]


def full_compress_reset(
    messages: list[dict],
    system_prompt: str,
    llm_call,
    *,
    task_board=None,
    trace_buffer=None,
) -> list[dict]:
    """
    Full-context compression: write handoff.md, rebuild a fresh message list,
    and clear the lightweight ActionRecord trace buffer if provided.

    Does not project memory blocks — caller must build_working_memory.
    """
    checkpoint = create_checkpoint(messages, llm_call)
    new_messages = restore_from_checkpoint(
        checkpoint, system_prompt, task_board=task_board
    )
    if trace_buffer is not None:
        clear_trace_buffer = getattr(trace_buffer, "clear", None)
        if callable(clear_trace_buffer):
            clear_trace_buffer()
        else:
            trace_buffer.records.clear()
            trace_buffer.compressed_through = 0
    return new_messages
