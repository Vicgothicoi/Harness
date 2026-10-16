"""Force the evaluator to write feedback.md before the iteration cap."""
from __future__ import annotations

import logging
from pathlib import Path

from harness import config
from harness.hooks.base import AgentHook, _as_result

log = logging.getLogger("harness")

WRITE_FILE_TOOL_CHOICE = {
    "type": "function",
    "function": {"name": "write_file"},
}


class EvaluatorDeliverableHook(AgentHook):
    """
    Evaluator deliverable gate.

    - Second-to-last iteration: warn that the next turn must write feedback.md.
    - Last iteration: force write_file(feedback.md) from evidence already gathered;
      block any other tool.
    """

    def __init__(
        self,
        deliverable: str | None = None,
        max_iterations: int | None = None,
    ):
        self.deliverable = deliverable or config.FEEDBACK_FILE
        self.max_iterations = (
            int(max_iterations)
            if max_iterations is not None
            else int(config.MAX_AGENT_ITERATIONS)
        )
        self._written = False
        self._forcing = False

    def _is_deliverable_write(self, tool_name: str, tool_args: dict) -> bool:
        if tool_name != "write_file":
            return False
        path = str((tool_args or {}).get("path") or "").replace("\\", "/").strip()
        path = path.lstrip("./")
        return path == self.deliverable or Path(path).name == self.deliverable

    def per_iteration(
        self,
        iteration: int,
        messages: list[dict],
        runtime_state=None,
        agent_name: str | None = None,
    ) -> str | None:
        self._forcing = False
        if runtime_state is not None:
            runtime_state.tool_choice = None

        if self._written:
            return None

        if iteration >= self.max_iterations:
            self._forcing = True
            if runtime_state is not None:
                runtime_state.tool_choice = WRITE_FILE_TOOL_CHOICE
            log.warning(
                "Evaluator deliverable: last iteration %s — forcing write_file(%s)",
                iteration,
                self.deliverable,
            )
            return (
                "[SYSTEM] This is your LAST iteration. Do not start new tests.\n"
                f"You MUST call write_file to save {self.deliverable} NOW, using only "
                "evidence already in this conversation.\n"
                "Follow the output format in your system prompt. Include parseable "
                "scores (criterion X/10 and Average: X/10). For anything not fully "
                "verified, give a conservative score and say so.\n"
                f"The only allowed tool is write_file(path=\"{self.deliverable}\"). "
                "Then stop."
            )

        if iteration == self.max_iterations - 1 and self.max_iterations > 1:
            log.info(
                "Evaluator deliverable: iteration %s/%s — wrap-up warning",
                iteration,
                self.max_iterations,
            )
            return (
                "[SYSTEM] You have one iteration left after this one.\n"
                "Do not start a new testing campaign. Next turn you MUST write "
                f"{self.deliverable} from evidence you already have. Unverified "
                "items get conservative scores and an explicit note."
            )
        return None

    def before_tool(
        self,
        tool_name: str,
        tool_args: dict,
        messages: list[dict],
        runtime_state=None,
        agent_name: str | None = None,
    ) -> str | None:
        if not self._forcing or self._written:
            return None
        if self._is_deliverable_write(tool_name, tool_args):
            return None
        return (
            f'[blocked] Last iteration: only write_file("{self.deliverable}") is '
            f"allowed. Do not call {tool_name}. Write {self.deliverable} now from "
            "evidence you already have."
        )

    def post_tool(
        self,
        tool_name: str,
        tool_args: dict,
        result: object,
        messages: list[dict],
        runtime_state=None,
        agent_name: str | None = None,
    ) -> str | None:
        if not self._is_deliverable_write(tool_name, tool_args):
            return None
        if _as_result(result).task_ok:
            self._written = True
            self._forcing = False
            if runtime_state is not None:
                runtime_state.tool_choice = None
            log.info("Evaluator deliverable: %s written", self.deliverable)
        return None
