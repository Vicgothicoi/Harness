"""
State compression — LLM-extract TaskBoard updates from conversation.

Persists via memory.state_memory; seeding a fresh board lives in
memory.state_memory.seed_task_board.
"""
from __future__ import annotations

import logging

from compression.full import messages_to_text
from memory.json_parse import parse_json_object
from memory.state_memory import apply_patch, persist_task_board, to_markdown

log = logging.getLogger("harness")

_STATE_COMPRESS_INSTRUCTION = """\
You maintain a structured task board for an autonomous coding agent.
Extract the CURRENT task state from the conversation log below.
Do NOT invent work that did not happen. Prefer concrete facts from tool results.

Keep these four steps when possible:
  understand task → implement → smoke-check → done
Do not replace "smoke-check" with open-ended product QA. Browser QA and scoring
belong to a later evaluator, not this agent.

Return ONLY a JSON object (no markdown fences) with these keys:
  "goal": string
  "steps": string[]           — ordered plan steps (prefer the four steps above)
  "current_step": string      — must be one of steps when steps is non-empty
  "completed_steps": string[] — subset of steps
  "blockers": string[]
  "next_action": string       — the single next concrete action, OR a STOP line

Wrap-up rules:
- If P0 deliverables exist on disk AND a smoke/syntax/build check succeeded
  (and a git commit if one was made): mark understand/implement/smoke-check
  complete, set current_step to "done", and set
  next_action to exactly: STOP — no further tool calls
- Do NOT invent extra work as next_action (patch scripts, headless test
  harnesses, another round of the same check).
- If deliverables are missing or smoke-check failed: next_action must be one
  concrete action, not STOP.

If the board is already accurate, still return a full JSON object reflecting it.
"""


def compress_state(messages: list[dict], board, llm_call) -> dict | None:
    """
    LLM-extract a TaskBoard patch from recent conversation, apply it, persist
    to progress.md. Returns the patch dict, or None on failure.
    """
    recent = messages[-40:] if len(messages) > 40 else messages
    transcript = messages_to_text(recent)
    current = to_markdown(board)

    raw = llm_call(
        [
            {"role": "system", "content": _STATE_COMPRESS_INSTRUCTION},
            {
                "role": "user",
                "content": (
                    "Current task board:\n"
                    f"{current}\n\n"
                    "Recent conversation / tool log:\n"
                    f"{transcript}"
                ),
            },
        ]
    )

    patch = parse_json_object(raw)
    if not patch:
        log.warning("State compression: failed to parse JSON patch")
        return None

    apply_patch(board, patch)
    persist_task_board(board)
    log.info(
        f"State compression: step={board.current_step!r} "
        f"next={board.next_action[:80]!r} updates={board.update_count}"
    )
    return patch
