"""
Agent hooks — callbacks that run at specific points in the agent loop.

Hooks are the harness engineer's primary tool for shaping agent behavior
without changing the core loop. They intercept execution at defined points:

  - post_tool:    After a tool call completes. Use for loop detection, tracking.
  - pre_exit:     When the agent wants to stop (no more tool calls). Use for
                  forced verification passes.
  - per_iteration: At the start of each iteration. Use for time budget warnings.
  - before_tool:  Before a tool executes. Use for hard blocks (recovery modes).

Hooks return an optional message to inject into the conversation.
Returning None means "no intervention."

Task progress / state memory is NOT handled here — Agent.run runs automatic
state compression each iteration and persists progress.md.

Design principle: hooks are composable and profile-specific.
The base Agent loop knows nothing about terminal tasks or time budgets —
profiles wire in the hooks they need.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from tool_result import ToolResult, wrap_legacy

log = logging.getLogger("harness")


def _as_result(result: object) -> ToolResult:
    return wrap_legacy(result)


class AgentHook(ABC):
    """Base class for agent hooks."""

    def before_tool(
        self,
        tool_name: str,
        tool_args: dict,
        messages: list[dict],
        runtime_state=None,
        agent_name: str | None = None,
    ) -> str | None:
        """Called before each tool execution. Return a blocking message, or None."""
        return None

    def post_tool(
        self,
        tool_name: str,
        tool_args: dict,
        result: object,
        messages: list[dict],
        runtime_state=None,
        agent_name: str | None = None,
    ) -> str | None:
        """Called after each tool execution. Return a message to inject, or None."""
        return None

    def pre_exit(
        self, messages: list[dict], runtime_state=None, agent_name: str | None = None
    ) -> str | None:
        """Called when the agent wants to stop. Return a message to force continuation, or None."""
        return None

    def per_iteration(
        self,
        iteration: int,
        messages: list[dict],
        runtime_state=None,
        agent_name: str | None = None,
    ) -> str | None:
        """Called at the start of each iteration. Return a message to inject, or None."""
        return None


# ---------------------------------------------------------------------------
# Loop Detection
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Pre-Exit Verification
# ---------------------------------------------------------------------------


class PreExitVerificationHook(AgentHook):
    """
    Forces the agent to run a verification pass before it's allowed to stop.

    Three-level exit gate:
    1. First exit attempt with NO tool calls ever made → force agent to start working
    2. First exit attempt after some work → force verification pass
    3. Second exit attempt after verification → allow exit

    This prevents the "3-second exit" problem where weak models return text
    without calling any tools, and PreExitVerification lets them go after
    just one retry.
    """

    def __init__(
        self,
        verification_prompt: str | None = None,
        include_task_requirements: bool = True,
    ):
        self._exit_attempts = 0
        self._verification_prompt = verification_prompt
        self._include_task_requirements = include_task_requirements

    @staticmethod
    def _has_done_work(messages: list[dict]) -> bool:
        """Check if the agent has called any action tools (run_shell, write_file, delegate_task)."""
        action_tools = {"run_shell", "write_file", "delegate_task"}
        for msg in messages:
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls", []):
                    fn_name = tc.get("function", {}).get("name", "")
                    if fn_name in action_tools:
                        return True
        return False

    @staticmethod
    def _extract_task_requirements(messages: list[dict]) -> str | None:
        """Extract the original task requirements from the conversation."""
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str) and len(content) > 20:
                    if len(content) > 3000:
                        content = content[:3000] + "\n... (truncated)"
                    return content
        return None

    def pre_exit(
        self, messages: list[dict], runtime_state=None, agent_name: str | None = None
    ) -> str | None:
        self._exit_attempts += 1
        has_worked = self._has_done_work(messages)

        # Gate 1: Agent hasn't done ANY work — force it to start
        if not has_worked:
            log.warning(
                f"Pre-exit: agent wants to stop but has done NO work (attempt {self._exit_attempts})"
            )
            if self._exit_attempts <= 3:  # give up to 3 chances to start working
                return (
                    "[SYSTEM] You have NOT completed the task. You have not executed any commands "
                    "or written any files yet.\n"
                    "You MUST use run_shell to execute commands and write_file to create output files.\n"
                    "Read the task requirements again and START WORKING. Do not just describe "
                    "what you would do — actually DO it using the available tools."
                )
            # After 3 attempts with no work, give up
            log.error("Pre-exit: agent refused to work after 3 attempts")
            return None

        # Gate 2: Agent has done work, first exit → force verification
        if self._exit_attempts == 1:
            log.info("Pre-exit verification: forcing verification pass")

            parts = []
            parts.append(
                "[SYSTEM] MANDATORY VERIFICATION — You are about to finish, "
                "but you MUST verify your work first."
            )

            if self._include_task_requirements:
                task_text = self._extract_task_requirements(messages)
                if task_text:
                    parts.append(
                        "\n--- ORIGINAL TASK REQUIREMENTS (verify against these, not your memory) ---\n"
                        f"{task_text}\n"
                        "--- END ORIGINAL TASK REQUIREMENTS ---"
                    )

            if self._verification_prompt:
                parts.append(f"\n{self._verification_prompt}")
            else:
                parts.append(
                    "\nDo NOT just re-read your code. Run actual test/check commands:\n"
                    "1. Go through EACH requirement above one by one.\n"
                    "2. For each, run a concrete verification command "
                    "(cat, ls -la, test -f, diff, grep, python3 -c, etc.)\n"
                    "3. Compare ACTUAL output against what the task asked for.\n"
                    "4. Pay special attention to exact formats, column orders, "
                    "file paths, and edge-case rules mentioned in the task.\n"
                    "5. If ANY check fails, fix it before stopping.\n"
                    "Think like an automated test script — would your solution pass?"
                )

            return "\n".join(parts)

        # Gate 3: Agent has done work and verified → allow exit
        log.info("Pre-exit verification: agent verified, allowing exit")
        return None


# ---------------------------------------------------------------------------
# Time Budget
# ---------------------------------------------------------------------------


class TimeBudgetHook(AgentHook):
    """
    Injects time awareness into the agent loop.

    At configurable thresholds (default: 60% and 85% of budget),
    warns the agent about remaining time and nudges it toward
    wrapping up and verifying.

    Can track time from harness start (not just agent start) by calling
    sync_start_time() before the agent runs. This ensures the budget
    accounts for time already spent on planning/setup.
    """

    def __init__(
        self,
        budget_seconds: float,
        warn_threshold: float = 0.60,
        critical_threshold: float = 0.85,
    ):
        self.budget_seconds = budget_seconds
        self.warn_threshold = warn_threshold
        self.critical_threshold = critical_threshold
        self.start_time = time.time()
        self._warned = False
        self._critical = False

    def sync_start_time(self, harness_start: float):
        """Set start time to harness start, so budget includes planning/setup time."""
        self.start_time = harness_start

    def per_iteration(
        self,
        iteration: int,
        messages: list[dict],
        runtime_state=None,
        agent_name: str | None = None,
    ) -> str | None:
        elapsed = time.time() - self.start_time
        fraction = elapsed / self.budget_seconds
        remaining = self.budget_seconds - elapsed

        if remaining <= 0:
            if not self._critical:
                self._critical = True
                log.warning("Time budget EXPIRED")
                return (
                    "[SYSTEM] ⚠️ TIME IS UP. You have exceeded the time budget.\n"
                    "STOP immediately. Save whatever you have and finish NOW."
                )
            return None

        if fraction >= self.critical_threshold and not self._critical:
            self._critical = True
            mins_left = remaining / 60
            log.warning(f"Time budget critical: {mins_left:.1f} min remaining")
            return (
                f"[SYSTEM] ⚠️ CRITICAL: Only {mins_left:.1f} minutes remaining out of "
                f"{self.budget_seconds / 60:.0f} min budget.\n"
                "STOP building new features. Immediately:\n"
                "1. Verify what you've done so far works correctly.\n"
                "2. Run final checks against the task requirements.\n"
                "3. Fix any broken items — do NOT start anything new."
            )

        if fraction >= self.warn_threshold and not self._warned:
            self._warned = True
            mins_left = remaining / 60
            log.info(f"Time budget warning: {mins_left:.1f} min remaining")
            return (
                f"[SYSTEM] Time check: {mins_left:.1f} minutes remaining out of "
                f"{self.budget_seconds / 60:.0f} min budget. "
                "Start wrapping up your current work and plan for verification."
            )

        return None


# ---------------------------------------------------------------------------
# Recovery state (owned by RecoveryStrategyHook; held by AgentRuntimeState)
# ---------------------------------------------------------------------------


@dataclass
class RecoveryState:
    mode: str = "NORMAL"
    failure_signature: str = ""
    repeat_count: int = 0
    last_successful_action: str = ""
    last_verification_result: str = ""
    tools_in_mode: int = 0  # tool calls since entering current recovery mode


# ---------------------------------------------------------------------------
# Task Tracking (forced decomposition)
# ---------------------------------------------------------------------------


class RecoveryStrategyHook(AgentHook):
    """Classify repeated failures and constrain the next class of actions.

    Progress acknowledgment previously required the update_progress tool.
    That is now handled by automatic state compression in Agent.run:
      - RETHINK blocks action tools until the next state-compress clears requires_update
      - SPEC_RECHECK clears after state compression once the agent has used tools in-mode
    """

    ENV_ERROR_PATTERNS = (
        "command not found",
        "permission denied",
        "no such file or directory",
        "externally-managed-environment",
        "no module named",
        "modulenotfounderror",
    )
    ACTION_TOOLS = {"run_shell", "write_file", "delegate_task"}
    READ_ONLY_PREFIXES = (
        "cat ",
        "ls",
        "pwd",
        "find ",
        "grep ",
        "head ",
        "tail ",
        "sed ",
        "git status",
        "git diff",
        "git log",
        "pytest",
        "python -m pytest",
        "test ",
        "diff ",
        "wc ",
        "which ",
        "env",
        "echo ",
        "printf ",
        "dir",
        "type ",
        "findstr ",
        "where ",
    )
    VERIFICATION_FAILURE_PATTERNS = (
        "assert",
        "failed",
        "failure",
        "mismatch",
        "expected",
        "traceback",
    )

    def __init__(self):
        self._edit_attempts: dict[str, int] = {}

    def _set_mode(self, runtime_state, mode: str) -> None:
        runtime_state.recovery.mode = mode
        runtime_state.recovery.tools_in_mode = 0
        runtime_state.task_board.requires_update = True

    def _clear_mode(self, runtime_state) -> None:
        runtime_state.recovery.mode = "NORMAL"
        runtime_state.recovery.failure_signature = ""
        runtime_state.recovery.repeat_count = 0
        runtime_state.recovery.tools_in_mode = 0
        runtime_state.task_board.requires_update = False

    def _register_failure(self, signature: str, runtime_state) -> None:
        recovery = runtime_state.recovery
        signature = signature.strip()
        if not signature:
            return
        if recovery.failure_signature == signature:
            recovery.repeat_count += 1
        else:
            recovery.failure_signature = signature
            recovery.repeat_count = 1

    def _is_env_failure(self, text: str) -> bool:
        lowered = text.lower()
        return any(pattern in lowered for pattern in self.ENV_ERROR_PATTERNS)

    def _is_read_only_command(self, command: str) -> bool:
        lowered = command.strip().lower()
        return any(lowered.startswith(prefix) for prefix in self.READ_ONLY_PREFIXES)

    def _looks_like_verification_failure(self, text: str) -> bool:
        lowered = text.lower()
        return any(pattern in lowered for pattern in self.VERIFICATION_FAILURE_PATTERNS)

    def observe_tool_result(
        self, tool_name: str, tool_args: dict, result: object, runtime_state
    ) -> None:
        if runtime_state is None:
            return
        outcome = _as_result(result)
        text = outcome.payload_text()
        if outcome.kind == "blocked":
            return

        env_fail = self._is_env_failure(text)
        if outcome.kind == "error" or (outcome.kind == "command_failed" and env_fail):
            self._register_failure(text.split("\n", 1)[0][:300], runtime_state)
            if env_fail and runtime_state.recovery.repeat_count >= 2:
                self._set_mode(runtime_state, "ENV_FIX")
                return
            if outcome.kind == "error" and runtime_state.recovery.repeat_count >= 2:
                self._set_mode(runtime_state, "SPEC_RECHECK")
                return
            return

        if tool_name in self.ACTION_TOOLS and outcome.task_ok:
            runtime_state.recovery.last_successful_action = tool_name

    def observe_verification_failure(self, failure_text: str, runtime_state) -> None:
        if runtime_state is None:
            return
        runtime_state.recovery.last_verification_result = failure_text
        self._register_failure(failure_text, runtime_state)
        if runtime_state.recovery.repeat_count >= 2:
            self._set_mode(runtime_state, "SPEC_RECHECK")

    def observe_edit_attempt(self, path: str, runtime_state) -> None:
        if runtime_state is None:
            return
        self._edit_attempts[path] = self._edit_attempts.get(path, 0) + 1
        if (
            runtime_state.recovery.mode == "SPEC_RECHECK"
            and runtime_state.recovery.failure_signature
            and self._edit_attempts[path] >= 2
        ):
            self._set_mode(runtime_state, "RETHINK")

    def before_tool(
        self,
        tool_name: str,
        tool_args: dict,
        messages: list[dict],
        runtime_state=None,
        agent_name: str | None = None,
    ) -> str | None:
        if agent_name != "builder" or runtime_state is None:
            return None
        mode = runtime_state.recovery.mode
        if mode == "NORMAL":
            return None

        if mode == "ENV_FIX":
            if tool_name in {"write_file", "delegate_task"}:
                return "[blocked] Recovery mode ENV_FIX only allows diagnosis, installation, and environment repair actions."
            return None

        if mode == "SPEC_RECHECK":
            if tool_name in {"write_file", "delegate_task"}:
                return "[blocked] Recovery mode SPEC_RECHECK is read-only. Re-read the task and verification outputs first."
            if tool_name == "run_shell" and not self._is_read_only_command(
                tool_args.get("command", "")
            ):
                return "[blocked] Recovery mode SPEC_RECHECK only allows read-only verification commands."
            return None

        if mode == "RETHINK":
            if (
                tool_name in self.ACTION_TOOLS
                and runtime_state.task_board.requires_update
            ):
                return (
                    "[blocked] Recovery mode RETHINK: stop repeating the same approach. "
                    "Wait for the next turn — automatic state compression will refresh the "
                    "task board, then change strategy before more edits or commands."
                )
            return None

        if mode == "FINAL_VERIFY":
            if tool_name in {
                "delegate_task",
                "web_search",
                "web_fetch",
            }:
                return "[blocked] Recovery mode FINAL_VERIFY only allows direct verification and final fixes."
            return None

        return None

    def post_tool(
        self,
        tool_name: str,
        tool_args: dict,
        result: object,
        messages: list[dict],
        runtime_state=None,
        agent_name: str | None = None,
    ) -> str | None:
        if agent_name != "builder" or runtime_state is None:
            return None
        outcome = _as_result(result)
        self.observe_tool_result(tool_name, tool_args, outcome, runtime_state)
        if outcome.kind in {"error", "blocked"}:
            return None

        if runtime_state.recovery.mode != "NORMAL":
            runtime_state.recovery.tools_in_mode += 1

        if tool_name == "run_shell":
            command = tool_args.get("command", "")
            payload = outcome.payload_text()
            if (
                runtime_state.recovery.mode == "ENV_FIX"
                and command
                and not self._is_read_only_command(command)
            ):
                self._clear_mode(runtime_state)
            elif self._is_read_only_command(
                command
            ) and self._looks_like_verification_failure(payload):
                self.observe_verification_failure(payload, runtime_state)

        if tool_name == "write_file":
            self.observe_edit_attempt(tool_args.get("path", ""), runtime_state)
        return None


# ---------------------------------------------------------------------------
# Error Guidance (structured recovery for weak models)
# ---------------------------------------------------------------------------


class ErrorGuidanceHook(AgentHook):
    """
    Detects common error patterns in tool output and injects specific,
    actionable recovery suggestions.

    Weak models struggle to recover from errors on their own — they often
    retry the same failing command or give up. This hook matches
    error patterns and provides concrete next steps.

    Based on TB2 command-level error analysis:
      - 24.1% of failures: command not found / not on PATH
      -  9.6% of failures: runtime errors in executables
      -  High rate: permission denied, missing dependencies
    """

    # Pattern → (description, recovery suggestion)
    # Patterns are checked in order; first match wins.
    ERROR_PATTERNS: list[tuple[str, str, str]] = [
        # --- Command not found ---
        (
            "command not found",
            "command_not_found",
            "The command is not installed. Try:\n"
            "  apt-get update && apt-get install -y <package>  (for system tools)\n"
            "  pip install <package>  (for Python tools)\n"
            "  which <command> || apt-cache search <keyword>  (to find the right package)\n"
            "If apt-get fails with permission denied, prefix with sudo.",
        ),
        (
            "no such file or directory",
            "file_not_found",
            "A file or directory doesn't exist. Check:\n"
            "  ls -la <parent_directory>  (does the path exist?)\n"
            "  pwd  (are you in the right directory?)\n"
            "  find . -name '<filename>'  (search for the file)",
        ),
        # --- Permission errors ---
        (
            "permission denied",
            "permission_denied",
            "Permission denied. Try:\n"
            "  chmod +x <file>  (if it needs to be executable)\n"
            "  sudo <command>  (if it needs root)\n"
            "  ls -la <file>  (check current permissions)",
        ),
        # --- Python/pip errors ---
        (
            "externally-managed-environment",
            "pip_managed_env",
            "This Python environment is externally managed (PEP 668). Use:\n"
            "  pip install --break-system-packages <package>\n"
            "  or: pip install --user <package>\n"
            "  or: python3 -m venv /tmp/venv && source /tmp/venv/bin/activate",
        ),
        (
            "modulenotfounderror",
            "python_import",
            "A Python module is missing. Install it:\n"
            "  pip install <module_name>\n"
            "  pip install --break-system-packages <module_name>  (if managed env)\n"
            "Check the exact package name — it may differ from the import name.",
        ),
        (
            "no module named",
            "python_import",
            "A Python module is missing. Install it:\n"
            "  pip install <module_name>\n"
            "Check: the pip package name may differ from the import name "
            "(e.g. 'import cv2' → 'pip install opencv-python').",
        ),
        # --- Compilation errors ---
        (
            "fatal error:",
            "compilation",
            "Compilation failed. Check:\n"
            "  1. Read the error — it shows the file and line number.\n"
            "  2. Missing header? Install dev packages: apt-get install -y lib<name>-dev\n"
            "  3. Use: apt-cache search <header_name> to find the right package.",
        ),
        (
            "undefined reference to",
            "linker",
            "Linker error — a symbol is missing. Check:\n"
            "  1. Are you linking all required libraries? (-l<lib> flag)\n"
            "  2. Is the library installed? apt-get install -y lib<name>-dev\n"
            "  3. Check library search path: ldconfig -p | grep <lib>",
        ),
        # --- Git errors ---
        (
            "not a git repository",
            "git",
            "Not in a git repository. Try:\n"
            "  git init  (to create one)\n"
            "  cd <correct_directory>  (you may be in the wrong dir)\n"
            "  find / -name '.git' -type d 2>/dev/null  (find existing repos)",
        ),
        # --- Disk/resource errors ---
        (
            "no space left on device",
            "disk_full",
            "Disk is full. Free space:\n"
            "  df -h  (check disk usage)\n"
            "  du -sh /* 2>/dev/null | sort -rh | head  (find large dirs)\n"
            "  apt-get clean  (clear package cache)\n"
            "  rm -rf /tmp/*  (clear temp files)",
        ),
        (
            "killed",
            "oom",
            "Process was killed (likely out of memory). Try:\n"
            "  free -h  (check available memory)\n"
            "  Reduce memory usage: smaller batch size, fewer workers, etc.\n"
            "  Use swap: fallocate -l 2G /swapfile && mkswap /swapfile && swapon /swapfile",
        ),
    ]

    def __init__(self):
        self._last_guidance_type: str | None = None

    def post_tool(
        self,
        tool_name: str,
        tool_args: dict,
        result: object,
        messages: list[dict],
        runtime_state=None,
        agent_name: str | None = None,
    ) -> str | None:
        if tool_name != "run_shell":
            return None

        outcome = _as_result(result)
        if outcome.task_ok:
            self._last_guidance_type = None
            return None

        result_lower = outcome.payload_text().lower()

        # Skip if no error indicators in the command output
        if (
            "error" not in result_lower
            and "not found" not in result_lower
            and "failed" not in result_lower
            and "traceback" not in result_lower
        ):
            self._last_guidance_type = None
            return None

        for pattern, guidance_type, suggestion in self.ERROR_PATTERNS:
            if pattern in result_lower:
                # Don't repeat the same guidance type consecutively
                if guidance_type == self._last_guidance_type:
                    return None
                self._last_guidance_type = guidance_type
                log.info(f"Error guidance: matched '{guidance_type}'")
                return f"[SYSTEM] Error detected — here's how to fix it:\n{suggestion}"

        self._last_guidance_type = None
        return None
