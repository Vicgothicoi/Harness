"""
Lightweight trace compression — rules-only ActionRecord buffer.

Each tool call appends an ActionRecord (off-window). When the conversation
exceeds the compress threshold, older assistant/tool turns are replaced by a
single [TRACE SUMMARY] built from those records (no LLM).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import config

TRACE_MARKER = "[TRACE SUMMARY]"


@dataclass
class ActionRecord:
    tool: str
    ok: bool
    detail: str = ""  # path, command snippet, or short note
    iteration: int = 0


@dataclass
class TraceBuffer:
    """Off-window action log for the current agent run."""

    records: list[ActionRecord] = field(default_factory=list)
    compressed_through: int = 0  # index in records already folded into a summary

    def clear(self) -> None:
        """Drop all ActionRecords after a full-context reset."""
        self.records.clear()
        self.compressed_through = 0

    def add(self, record: ActionRecord) -> None:
        self.records.append(record)

    def pending(self) -> list[ActionRecord]:
        return self.records[self.compressed_through :]

    def mark_all_compressed(self) -> None:
        self.compressed_through = len(self.records)


def record_from_tool(
    tool_name: str,
    tool_args: dict | None,
    result: str,
    iteration: int = 0,
) -> ActionRecord:
    """Build an ActionRecord from a tool call (rules only)."""
    args = tool_args or {}
    ok = not (
        isinstance(result, str)
        and (result.startswith("[error]") or result.startswith("[blocked]"))
    )
    detail = ""
    if tool_name in {"write_file", "read_file", "read_skill_file"}:
        detail = str(args.get("path", ""))[:120]
    elif tool_name == "run_shell":
        detail = str(args.get("command", ""))[:120]
    elif tool_name in {"delegate_task", "delegate_tasks"}:
        detail = str(args.get("task") or args.get("tasks") or "")[:120]
    elif tool_name in {"web_search", "web_fetch"}:
        detail = str(args.get("query") or args.get("url") or "")[:120]
    elif tool_name in {"browser_open", "browser_goto"}:
        detail = str(args.get("url", ""))[:120]
    elif tool_name in {"browser_click", "browser_fill"}:
        detail = str(args.get("selector", ""))[:120]
    elif tool_name == "browser_evaluate":
        detail = str(args.get("expression", ""))[:120]
    elif tool_name.startswith("browser_") or tool_name in {
        "start_dev_server",
        "stop_dev_server",
    }:
        detail = str(args)[:80]
    else:
        detail = str(args)[:80]

    if not ok and isinstance(result, str):
        # Keep a short failure hint
        hint = result.split("\n", 1)[0][:100]
        detail = f"{detail} | {hint}" if detail else hint

    return ActionRecord(tool=tool_name, ok=ok, detail=detail.strip(), iteration=iteration)


def format_trace_summary(records: list[ActionRecord], *, max_chars: int | None = None) -> str:
    """Render ActionRecords as a [TRACE SUMMARY] user message body."""
    if max_chars is None:
        max_chars = int(getattr(config, "TRACE_SUMMARY_MAX_CHARS", 4000))

    if not records:
        return f"{TRACE_MARKER}\n(no tool actions recorded)"

    lines = [
        TRACE_MARKER,
        f"Earlier tool actions ({len(records)}), compressed without raw outputs:",
    ]
    # Cap listed lines; keep head + tail of the log
    max_lines = 80
    if len(records) <= max_lines:
        shown = list(enumerate(records))
    else:
        head_n = max_lines // 2
        tail_n = max_lines - head_n
        shown = list(enumerate(records[:head_n]))
        omitted = len(records) - head_n - tail_n
        mid = [(-1, None)]  # placeholder
        shown = shown + mid + list(enumerate(records[-tail_n:], start=len(records) - tail_n))

    for idx, rec in shown:
        if rec is None:
            lines.append(f"  ... ({omitted} actions omitted) ...")
            continue
        status = "ok" if rec.ok else "FAIL"
        detail = f" — {rec.detail}" if rec.detail else ""
        iter_s = f" i{rec.iteration}" if rec.iteration else ""
        lines.append(f"  [{status}]{iter_s} {rec.tool}{detail}")

    # Aggregate stats
    fails = sum(1 for r in records if not r.ok)
    tools_used = sorted({r.tool for r in records})
    lines.append(f"Stats: {len(records)} actions, {fails} failures; tools={', '.join(tools_used)}")

    body = "\n".join(lines)
    if len(body) > max_chars:
        body = body[: max_chars - 20] + "\n...(truncated)"
    return body


def _safe_split_index(messages: list[dict], target_idx: int) -> int:
    """Ensure `recent` does not start on a lone tool response.

    If the cut lands on a `tool` message, walk back to its assistant
    tool_calls message so the pair stays together in `recent`.
    Landing on an assistant with tool_calls is a valid cut — do not walk further.
    """
    idx = max(0, min(target_idx, len(messages)))
    while idx > 0 and idx < len(messages) and messages[idx].get("role") == "tool":
        idx -= 1
    return idx


def _is_memory_or_trace_block(msg: dict) -> bool:
    if msg.get("role") != "user":
        return False
    content = msg.get("content")
    if not isinstance(content, str):
        return False
    return content.startswith((
        "[STATE MEMORY]",
        "[PROJECT MEMORY]",
        "[LONG-TERM MEMORY]",
        TRACE_MARKER,
        "[SYSTEM]",
    ))


def compress_trace(
    messages: list[dict],
    trace_buffer: TraceBuffer,
    *,
    keep_recent: int | None = None,
) -> list[dict]:
    """
    Replace older conversation turns with a rules-based [TRACE SUMMARY].

    Preserves: system message, memory/trace marker blocks near the front,
    and the most recent `keep_recent` non-system messages (tool-pair safe).
    """
    if not messages:
        return messages
    if keep_recent is None:
        keep_recent = int(getattr(config, "TRACE_KEEP_RECENT", 8))

    system = [messages[0]] if messages[0].get("role") == "system" else []
    rest = messages[len(system) :]

    # Pull leading memory / existing trace blocks out so they stay after system
    leading: list[dict] = []
    body = rest
    while body and _is_memory_or_trace_block(body[0]):
        # Drop old TRACE SUMMARY — we will write a fresh one
        if isinstance(body[0].get("content"), str) and body[0]["content"].startswith(
            TRACE_MARKER
        ):
            body = body[1:]
            continue
        leading.append(body[0])
        body = body[1:]

    if len(body) <= keep_recent:
        return messages

    split_idx = len(body) - keep_recent
    split_idx = _safe_split_index(body, split_idx)
    if split_idx <= 0:
        return messages

    recent = body[split_idx:]
    pending = trace_buffer.pending()
    # If buffer empty, still summarize from message structure lightly
    if pending:
        summary_text = format_trace_summary(pending)
        trace_buffer.mark_all_compressed()
    else:
        summary_text = (
            f"{TRACE_MARKER}\n"
            f"Earlier conversation turns were dropped to free context "
            f"({split_idx} messages removed). Continue from recent turns below."
        )

    summary_msg = {"role": "user", "content": summary_text}
    return system + leading + [summary_msg] + recent
