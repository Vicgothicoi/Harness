"""
Structured tool results —
Tool calls should return ``ToolResult``.
trace / hooks read ``kind`` and ``exit_code`` for success/failure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Kind = Literal["ok", "error", "blocked", "command_failed"]

_LEGACY_ERROR_PREFIX = "[error]"
_LEGACY_BLOCKED_PREFIX = "[blocked]"


@dataclass
class ShellObservation:
    """Structured run_shell payload. stdout/stderr stay split for compression."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    timed_out: bool = False


@dataclass
class ToolResult:
    """
    kind:
      ok              — protocol succeeded and (if a process) exit_code == 0
      error           — harness/protocol failure (unknown tool, timeout, exception)
      blocked         — hook intercepted before execution
      command_failed  — process ran to completion with a non-zero exit
    """

    kind: Kind
    payload: str | ShellObservation = ""
    exit_code: int | None = None
    auto_fix: str | None = None

    @property
    def protocol_ok(self) -> bool:
        """True when the tool actually ran (including a failed command)."""
        return self.kind in ("ok", "command_failed")

    @property
    def task_ok(self) -> bool:
        """True when the invocation both ran and achieved its purpose."""
        return self.kind == "ok"

    def payload_text(self) -> str:
        """Flatten payload to text (no truncation)."""
        payload = self.payload
        if isinstance(payload, ShellObservation):
            stdout = (payload.stdout or "").strip()
            stderr = (payload.stderr or "").strip()
            if stderr:
                return (stdout + "\n\n--- STDERR ---\n" + stderr).strip()
            return stdout
        if payload is None:
            return ""
        if not isinstance(payload, str):
            return str(payload)
        return payload

    def with_auto_fix(self, auto_fix: str | None) -> ToolResult:
        if not auto_fix:
            return self
        return ToolResult(
            kind=self.kind,
            payload=self.payload,
            exit_code=self.exit_code,
            auto_fix=auto_fix,
        )


def result_ok(
    payload: str | ShellObservation = "",
    *,
    exit_code: int | None = None,
    auto_fix: str | None = None,
) -> ToolResult:
    return ToolResult(
        kind="ok",
        payload=payload,
        exit_code=exit_code,
        auto_fix=auto_fix,
    )


def result_error(
    payload: str | ShellObservation,
    *,
    exit_code: int | None = None,
    auto_fix: str | None = None,
) -> ToolResult:
    return ToolResult(
        kind="error",
        payload=payload,
        exit_code=exit_code,
        auto_fix=auto_fix,
    )


def result_blocked(payload: str) -> ToolResult:
    return ToolResult(kind="blocked", payload=payload)


def from_shell(
    obs: ShellObservation,
    *,
    auto_fix: str | None = None,
) -> ToolResult:
    """Lift a shell payload into ToolResult using exit_code / timed_out."""
    if obs.timed_out:
        return ToolResult(
            kind="error",
            payload=obs,
            exit_code=obs.exit_code,
            auto_fix=auto_fix,
        )
    if obs.exit_code != 0:
        return ToolResult(
            kind="command_failed",
            payload=obs,
            exit_code=obs.exit_code,
            auto_fix=auto_fix,
        )
    return ToolResult(
        kind="ok",
        payload=obs,
        exit_code=obs.exit_code,
        auto_fix=auto_fix,
    )


def wrap_legacy(value: object) -> ToolResult:
    """Adapt current str / ShellObservation / ToolResult returns.

    Prefix rules match today's text protocol so the agent loop can switch
    to ToolResult before every leaf tool is migrated.
    """
    if isinstance(value, ToolResult):
        return value
    if isinstance(value, ShellObservation):
        return from_shell(value)
    if value is None:
        return result_ok("")
    text = value if isinstance(value, str) else str(value)
    if text.startswith(_LEGACY_BLOCKED_PREFIX):
        return result_blocked(text)
    if text.startswith(_LEGACY_ERROR_PREFIX):
        return result_error(text)
    return result_ok(text)
