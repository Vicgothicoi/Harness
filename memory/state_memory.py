"""
State memory — TaskBoard as the single source of truth for task progress.

Persists to progress.md. Injects a replaceable [STATE MEMORY] block into the
agent's working messages each iteration.
"""
from __future__ import annotations

from pathlib import Path

import config

STATE_MARKER = "[STATE MEMORY]"


def to_markdown(board) -> str:
    """Serialize TaskBoard to the progress.md format."""
    goal = board.goal or "(unset)"
    steps = board.steps or []
    current_step = board.current_step or ""
    completed = set(board.completed_steps or [])
    blockers = board.blockers or []
    next_action = board.next_action or "(unset)"

    lines = [
        "# Progress",
        "",
        "## Goal",
        goal,
        "",
        "## Steps",
    ]
    if steps:
        for step in steps:
            marker = "x" if step in completed else " "
            current = " (current)" if step == current_step else ""
            lines.append(f"- [{marker}] {step}{current}")
    else:
        lines.append("- (none yet)")

    lines.extend(["", "## Blockers"])
    if blockers:
        lines.extend(f"- {item}" for item in blockers)
    else:
        lines.append("- (none)")

    lines.extend(
        [
            "",
            "## Next Action",
            next_action,
            "",
            "## Updates",
            str(board.update_count),
        ]
    )
    return "\n".join(lines) + "\n"


def to_context_block(board, max_chars: int | None = None) -> str:
    """Short summary injected into the working context window."""
    if max_chars is None:
        max_chars = getattr(config, "STATE_CONTEXT_MAX_CHARS", 1500)

    body = to_markdown(board).strip()
    if len(body) > max_chars:
        body = body[: max_chars - 20] + "\n...(truncated)"
    return f"{STATE_MARKER}\n{body}"


def persist_task_board(board, workspace: str | Path | None = None) -> Path:
    """Write TaskBoard to progress.md. Returns the path written."""
    ws = Path(workspace or config.WORKSPACE)
    path = ws / config.PROGRESS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_markdown(board), encoding="utf-8")
    return path


def apply_patch(board, patch: dict) -> None:
    """
    Apply a partial patch dict onto TaskBoard.

    Expected keys (all optional): goal, steps, current_step, completed_steps,
    blockers, next_action.
    """
    if not isinstance(patch, dict):
        return

    if "goal" in patch and isinstance(patch["goal"], str) and patch["goal"].strip():
        board.goal = patch["goal"].strip()

    if "steps" in patch and isinstance(patch["steps"], list):
        steps = [str(s).strip() for s in patch["steps"] if str(s).strip()]
        if steps:
            board.steps = steps

    if "completed_steps" in patch and isinstance(patch["completed_steps"], list):
        completed = [str(s).strip() for s in patch["completed_steps"] if str(s).strip()]
        # Keep only steps that exist on the board (after possible steps update)
        known = set(board.steps)
        board.completed_steps = [s for s in completed if not known or s in known]

    if "current_step" in patch and isinstance(patch["current_step"], str):
        current = patch["current_step"].strip()
        if current and (not board.steps or current in board.steps):
            board.current_step = current
        elif board.steps and not board.current_step:
            board.current_step = board.steps[0]

    if "blockers" in patch and isinstance(patch["blockers"], list):
        board.blockers = [str(b).strip() for b in patch["blockers"] if str(b).strip()]

    if "next_action" in patch and isinstance(patch["next_action"], str):
        next_action = patch["next_action"].strip()
        if next_action:
            board.next_action = next_action

    # Normalize: current_step should be in steps when steps exist
    if board.steps and board.current_step and board.current_step not in board.steps:
        board.current_step = board.steps[0]

    board.update_count += 1
    board.requires_update = False
    board.needs_final_update = False


def inject_state_summary(messages: list[dict], board, max_chars: int | None = None) -> list[dict]:
    """
    Ensure exactly one [STATE MEMORY] user message exists in the conversation.

    Replaces any previous STATE_MARKER message in place (prefer after system),
    rather than appending every iteration.
    """
    block = to_context_block(board, max_chars=max_chars)
    state_msg = {"role": "user", "content": block}

    # Remove existing state blocks
    cleaned = [
        m
        for m in messages
        if not (
            m.get("role") == "user"
            and isinstance(m.get("content"), str)
            and m["content"].startswith(STATE_MARKER)
        )
    ]

    # Insert after system prompt if present, else at front
    if cleaned and cleaned[0].get("role") == "system":
        return [cleaned[0], state_msg] + cleaned[1:]
    return [state_msg] + cleaned


def seed_task_board(board, task: str) -> None:
    """Initialize TaskBoard from the raw task string without an LLM call."""
    goal = (task or "").strip()
    if len(goal) > 500:
        goal = goal[:500] + "..."
    apply_patch(
        board,
        {
            "goal": goal or board.goal or "complete the assigned task",
            "steps": board.steps or ["understand task", "implement", "verify"],
            "current_step": board.current_step or "understand task",
            "completed_steps": board.completed_steps or [],
            "blockers": board.blockers or [],
            "next_action": board.next_action or "read the task requirements and begin",
        },
    )
    persist_task_board(board)
