"""
App Builder profile — the original scenario from the Anthropic article.
Plan a product → write code → evaluate → iterate.

Browser UI testing is exposed via BrowserMcpClient on the evaluator only
(AgentConfig.mcp_bridges). Planner/builder do not get browser MCP tools.
"""
from __future__ import annotations

import prompts
from browser_mcp_client import BrowserMcpClient
from profiles.base import BaseProfile, AgentConfig


class AppBuilderProfile(BaseProfile):

    def name(self) -> str:
        return "app-builder"

    def description(self) -> str:
        return "Build complete web applications from a one-sentence prompt (Anthropic article scenario)"

    def planner(self) -> AgentConfig:
        return AgentConfig(system_prompt=prompts.PLANNER_SYSTEM)

    def builder(self) -> AgentConfig:
        return AgentConfig(system_prompt=prompts.BUILDER_SYSTEM)

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
