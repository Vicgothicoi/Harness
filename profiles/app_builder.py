"""
App Builder profile — the original scenario from the Anthropic article.
Plan a product → write code → evaluate → iterate.

Browser UI testing is exposed via BrowserMcpClient on the evaluator only
(AgentConfig.mcp_bridges). Planner/builder do not get browser MCP tools.
"""
from __future__ import annotations

import prompts
from browser_mcp_client import BrowserMcpClient
from hooks import LoopDetectionHook, TimeBudgetHook
from profiles.base import BaseProfile, AgentConfig


class AppBuilderProfile(BaseProfile):

    _DEFAULTS = {
        "task_budget": 1800,
        "loop_file_edit_threshold": 8,
        "loop_command_repeat_threshold": 3,
        "time_warn_threshold": 0.60,
        "time_critical_threshold": 0.85,
    }

    def _get(self, key: str):
        return self.cfg.resolve(key, self.name(), self._DEFAULTS[key])

    def name(self) -> str:
        return "app-builder"

    def description(self) -> str:
        return "Build complete web applications from a one-sentence prompt (Anthropic article scenario)"

    def planner(self) -> AgentConfig:
        return AgentConfig(system_prompt=prompts.PLANNER_SYSTEM)

    def builder(self) -> AgentConfig:
        budget = self._get("task_budget")
        return AgentConfig(
            system_prompt=prompts.BUILDER_SYSTEM,
            time_budget=budget,
            enable_memory=True,
            hooks=[
                LoopDetectionHook(
                    file_edit_threshold=self._get("loop_file_edit_threshold"),
                    command_repeat_threshold=self._get("loop_command_repeat_threshold"),
                ),
                TimeBudgetHook(
                    budget_seconds=budget,
                    warn_threshold=self._get("time_warn_threshold"),
                    critical_threshold=self._get("time_critical_threshold"),
                ),
            ],
        )

    def evaluator(self) -> AgentConfig:
        return AgentConfig(
            system_prompt=prompts.EVALUATOR_SYSTEM,
            mcp_bridges=[BrowserMcpClient(transport="inprocess")],
        )

    def contract_proposer(self) -> AgentConfig:
        return AgentConfig(system_prompt="", enabled=False)
        # return AgentConfig(system_prompt=prompts.CONTRACT_BUILDER_SYSTEM)

    def contract_reviewer(self) -> AgentConfig:
        return AgentConfig(system_prompt="", enabled=False)
        # return AgentConfig(system_prompt=prompts.CONTRACT_REVIEWER_SYSTEM)
