"""
Observation compression — shrink individual tool results before they enter
the conversation window. Rules-only (no LLM) in v1.

Existing tools may already coarse-truncate; this layer applies a unified
per-tool / global ceiling on top.
"""
from __future__ import annotations

import config

_DEFAULT_MAX = 10000
_TOOL_MAX = {
    "run_shell": 20000,
    "read_file": 10000,
    "web_fetch": 6000,
    "web_search": 4000,
    "delegate_task": 6000,
    # Browser Testing MCP tools
    "browser_snapshot": 4000,
    "browser_console": 3000,
    "browser_evaluate": 3000,
}
_SKIP_COMPRESSION = {"read_skill_file"}


def _limit_for(tool_name: str) -> int:
    tool_max = int(_TOOL_MAX.get(tool_name, _DEFAULT_MAX))
    global_max = int(getattr(config, "OBSERVATION_MAX_CHARS", tool_max) or tool_max)
    return min(tool_max, global_max)


def compress_observation(
    tool_name: str,
    tool_args: dict | None,
    result: str,
) -> str:
    """
    Rules-based compression of a single tool result.

    Returns the string stored in the conversation as the tool message content.
    """
    if not isinstance(result, str):
        result = str(result)

    limit = _limit_for(tool_name)
    if tool_name in _SKIP_COMPRESSION or len(result) <= limit:
        return result

    head_size = int(limit * 0.55)
    tail_size = limit - head_size - 120
    if tail_size < 200:
        tail_size = 200
        head_size = max(200, limit - tail_size - 120)

    head = result[:head_size]
    tail = result[-tail_size:]
    omitted = len(result) - head_size - tail_size
    path_hint = ""
    if isinstance(tool_args, dict):
        path = tool_args.get("path") or tool_args.get("command")
        if path:
            path_hint = f" source={str(path)[:120]}"

    return (
        f"{head}\n"
        f"\n[OBSERVATION COMPRESSED] omitted ~{omitted} chars "
        f"(tool={tool_name}{path_hint}; limit={limit}). "
        f"Re-read with a narrower command/path if you need the middle.\n"
        f"\n{tail}"
    )
