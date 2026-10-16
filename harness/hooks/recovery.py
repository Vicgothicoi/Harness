"""Failure classification and recovery-mode tool blocks."""
from __future__ import annotations

from dataclasses import dataclass

from harness.hooks.base import AgentHook, _as_result

@dataclass
class RecoveryState:
    mode: str = "NORMAL"
    failure_signature: str = ""
    repeat_count: int = 0
    last_successful_action: str = ""
    last_verification_result: str = ""
    tools_in_mode: int = 0

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
