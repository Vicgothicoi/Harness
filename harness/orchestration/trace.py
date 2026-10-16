"""JSONL trace of agent events, written outside the workspace."""
from __future__ import annotations

import json
import time
from pathlib import Path

from harness import config

def _truncate(s: object, n: int) -> str:
    text = s if isinstance(s, str) else str(s)
    return text[:n] + "..." if len(text) > n else text

class TraceWriter:
    """Appends structured events to a JSONL trace file outside the agent workspace.

    Each line is a JSON object with: timestamp, agent, event_type, and data.
    Trace file: {LOG}/_trace_{agent_name}.jsonl
    """

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self._start_time = time.time()
        trace_dir = Path(config.LOG)
        trace_dir.mkdir(parents=True, exist_ok=True)
        self._path = trace_dir / f"_trace_{agent_name}.jsonl"

    def _write(self, event_type: str, data: dict):
        try:
            entry = {
                "t": round(time.time() - self._start_time, 2),
                "agent": self.agent_name,
                "event": event_type,
                **data,
            }
            line = json.dumps(entry, ensure_ascii=False)[:10000]
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            # Also print to stderr so Harbor logs capture it
            import sys

            print(f"[TRACE] {line}", file=sys.stderr)
        except Exception:
            pass  # never let tracing break the agent

    def iteration(self, n: int, tokens: int):
        self._write("iteration", {"n": n, "tokens": tokens})

    def llm_response(
        self, content: str | None, tool_calls: list | None, finish_reason: str | None
    ):
        self._write(
            "llm_response",
            {
                "content": (content or "")[:500],
                "tool_calls": [tc["function"]["name"] for tc in (tool_calls or [])],
                "finish_reason": finish_reason,
            },
        )

    def tool_call(self, name: str, args: dict, result: str):
        self._write(
            "tool_call",
            {
                "tool": name,
                "args": _truncate(json.dumps(args, ensure_ascii=False), 300),
                "result": _truncate(result, 500),
            },
        )

    def hook_inject(self, source: str, hook: str, message: str):
        self._write(
            "hook",
            {
                "source": source,
                "hook": hook,
                "message": message[:300],
            },
        )

    def context_event(self, event_type: str, reason: str = ""):
        self._write("context", {"type": event_type, "reason": reason})

    def error(self, error_type: str, message: str):
        self._write("error", {"type": error_type, "message": message[:500]})

    def finish(self, reason: str, iterations: int):
        self._write("finish", {"reason": reason, "iterations": iterations})

    def dump_messages(self, iteration: int, messages: list[dict], tokens: int):
        """Append one LLM-request context snapshot (JSONL, not truncated)."""
        try:
            path = Path(config.LOG) / f"_messages_{self.agent_name}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "iteration": iteration,
                "tokens": tokens,
                "messages": messages,
            }
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass
