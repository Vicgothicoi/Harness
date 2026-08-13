"""
Working memory — Context Builder for the LLM window.

Assembles projections of off-window stores into the conversation in a fixed
order (replace-in-place, never append duplicates):

  system
  [STATE MEMORY]
  [PROJECT MEMORY]
  [LONG-TERM MEMORY]   # preferences only until RAG exists
  ... task / dialogue / tools ...

Does not run compression; callers refresh TaskBoard / project JSON separately.
"""
from __future__ import annotations

from memory.long_term_memory import LongTermMemory, inject_long_term_preferences
from memory.project_memory import ProjectMemory, inject_project_summary
from memory.state_memory import inject_state_summary


def build_working_memory(
    messages: list[dict],
    *,
    task_board=None,
    project_memory: ProjectMemory | None = None,
    long_term_memory: LongTermMemory | None = None,
    load_defaults: bool = True,
) -> list[dict]:
    """
    Project memory layers into `messages` and return the working window.

    If `load_defaults` is True, missing project/LTM args are loaded from disk.
    Pass explicit empty ProjectMemory()/LongTermMemory() with load_defaults=False
    to skip loading (tests).
    """
    if task_board is not None:
        messages = inject_state_summary(messages, task_board)

    if project_memory is None and load_defaults:
        project_memory = ProjectMemory.load()
    if project_memory is not None:
        messages = inject_project_summary(messages, project_memory)

    if long_term_memory is None and load_defaults:
        long_term_memory = LongTermMemory.load()
    if long_term_memory is not None:
        messages = inject_long_term_preferences(messages, long_term_memory)

    return messages


def build_working_memory_from_runtime(
    messages: list[dict],
    runtime_state,
) -> list[dict]:
    """Convenience: project using AgentRuntimeState.task_board + disk loads."""
    board = getattr(runtime_state, "task_board", None)
    return build_working_memory(messages, task_board=board, load_defaults=True)
