"""Detect repeated file edits and command loops."""
from __future__ import annotations

import logging

from harness.hooks.base import AgentHook, _as_result

log = logging.getLogger("harness")

class LoopDetectionHook(AgentHook):
    """
    Tracks per-file edit counts and detects repetitive command patterns.
    When the agent edits the same file or runs similar commands too many times,
    injects a nudge to reconsider the approach.

    Uses fuzzy matching for commands — catches variants like:
      python3 app.py  /  python3 app.py 2>&1  /  python3 ./app.py
    """

    def __init__(self, file_edit_threshold: int = 4, command_repeat_threshold: int = 3):
        self.file_edit_threshold = file_edit_threshold
        self.command_repeat_threshold = command_repeat_threshold
        self.file_edit_counts: dict[str, int] = {}
        self.recent_commands: list[str] = []
        self._file_warned: set[str] = set()  # avoid spamming same warning
        self._consecutive_failures = 0

    @staticmethod
    def _normalize_command(cmd: str) -> str:
        """Normalize a command for fuzzy comparison."""
        import re

        cmd = cmd.strip()
        # Remove common suffixes that don't change semantics
        cmd = re.sub(r"\s*2>&1\s*$", "", cmd)
        cmd = re.sub(r"\s*\|\s*head.*$", "", cmd)
        cmd = re.sub(r"\s*\|\s*tail.*$", "", cmd)
        # Normalize paths: ./foo → foo
        cmd = re.sub(r"\./(\S)", r"\1", cmd)
        # Collapse whitespace
        cmd = re.sub(r"\s+", " ", cmd)
        return cmd.strip()

    def post_tool(
        self,
        tool_name: str,
        tool_args: dict,
        result: object,
        messages: list[dict],
        runtime_state=None,
        agent_name: str | None = None,
    ) -> str | None:
        outcome = _as_result(result)
        # Track file edits
        if tool_name == "write_file" and outcome.kind != "blocked":
            path = tool_args.get("path", "")
            self.file_edit_counts[path] = self.file_edit_counts.get(path, 0) + 1
            count = self.file_edit_counts[path]
            if count >= self.file_edit_threshold and path not in self._file_warned:
                self._file_warned.add(path)
                log.warning(f"Loop detection: {path} edited {count} times")
                return (
                    f"[SYSTEM] You have edited '{path}' {count} times. "
                    "This pattern suggests your current approach may not be working. "
                    "STOP and reconsider:\n"
                    "1. Re-read the original task requirements.\n"
                    "2. Think about what's fundamentally wrong with your approach.\n"
                    "3. Try a completely different strategy."
                )

        # Track repeated commands (with fuzzy matching)
        if tool_name == "run_shell":
            cmd = tool_args.get("command", "").strip()
            self.recent_commands.append(cmd)
            if len(self.recent_commands) >= self.command_repeat_threshold:
                window = self.recent_commands[-self.command_repeat_threshold :]
                normalized = [self._normalize_command(c) for c in window]
                if len(set(normalized)) == 1:
                    log.warning(
                        f"Loop detection: similar command repeated {self.command_repeat_threshold}x"
                    )
                    return (
                        f"[SYSTEM] You have run essentially the same command {self.command_repeat_threshold} "
                        f"times in a row with no progress.\n"
                        f"Command pattern: {normalized[0][:200]}\n"
                        "This is a doom loop. The same action will not produce a different result.\n"
                        "STOP. Re-read the error output carefully. Try a fundamentally different approach."
                    )

            if outcome.kind in {"error", "command_failed"}:
                self._consecutive_failures += 1
                if self._consecutive_failures >= 3:
                    self._consecutive_failures = 0
                    return (
                        "[SYSTEM] Multiple consecutive commands have failed. "
                        "Stop and diagnose the root cause before trying more commands. "
                        "Check: Is the required tool installed? Are you in the right directory? "
                        "Is there a dependency missing?"
                    )
            elif outcome.kind == "ok":
                self._consecutive_failures = 0

        return None
