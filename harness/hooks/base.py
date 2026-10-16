"""Hook base class. Profiles compose concrete hooks onto an agent."""
from __future__ import annotations

from abc import ABC

from harness.tools.result import ToolResult, wrap_legacy

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
