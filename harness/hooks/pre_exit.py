"""Force a verification pass before the agent stops."""
from __future__ import annotations

import logging

from harness.hooks.base import AgentHook

log = logging.getLogger("harness")

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
