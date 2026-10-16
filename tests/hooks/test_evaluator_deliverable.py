"""Evaluator last-iteration deliverable gate."""
from __future__ import annotations

import sys
import types
import unittest


def _install_fake_openai_module() -> None:
    openai = types.ModuleType("openai")

    class OpenAI:
        def __init__(self, *args, **kwargs):
            pass

    openai.OpenAI = OpenAI
    sys.modules["openai"] = openai


_install_fake_openai_module()

from harness.hooks.evaluator_deliverable import (
    WRITE_FILE_TOOL_CHOICE,
    EvaluatorDeliverableHook,
)
from harness.orchestration.agent import AgentRuntimeState
from harness.tools.result import result_error, result_ok


class EvaluatorDeliverableHookTests(unittest.TestCase):
    def test_warns_on_second_to_last_iteration(self):
        hook = EvaluatorDeliverableHook(max_iterations=5)
        state = AgentRuntimeState()
        msg = hook.per_iteration(4, [], runtime_state=state)
        self.assertIsNotNone(msg)
        self.assertIn("one iteration left", msg.lower())
        self.assertIsNone(state.tool_choice)

    def test_forces_write_file_on_last_iteration(self):
        hook = EvaluatorDeliverableHook(max_iterations=5)
        state = AgentRuntimeState()
        msg = hook.per_iteration(5, [], runtime_state=state)
        self.assertIsNotNone(msg)
        self.assertIn("LAST iteration", msg)
        self.assertIn("feedback.md", msg)
        self.assertEqual(state.tool_choice, WRITE_FILE_TOOL_CHOICE)

    def test_single_iteration_budget_forces_immediately(self):
        hook = EvaluatorDeliverableHook(max_iterations=1)
        state = AgentRuntimeState()
        msg = hook.per_iteration(1, [], runtime_state=state)
        self.assertIsNotNone(msg)
        self.assertIn("LAST iteration", msg)
        self.assertEqual(state.tool_choice, WRITE_FILE_TOOL_CHOICE)

    def test_mid_run_does_not_intervene(self):
        hook = EvaluatorDeliverableHook(max_iterations=30)
        state = AgentRuntimeState()
        self.assertIsNone(hook.per_iteration(10, [], runtime_state=state))
        self.assertIsNone(state.tool_choice)

    def test_before_tool_blocks_non_feedback_writes_when_forcing(self):
        hook = EvaluatorDeliverableHook(max_iterations=2)
        state = AgentRuntimeState()
        hook.per_iteration(2, [], runtime_state=state)
        blocked = hook.before_tool(
            "browser_evaluate", {"expression": "1"}, [], runtime_state=state
        )
        self.assertIsNotNone(blocked)
        self.assertTrue(blocked.startswith("[blocked]"))

        allowed = hook.before_tool(
            "write_file",
            {"path": "feedback.md", "content": "## QA\n- **Average: 7/10**"},
            [],
            runtime_state=state,
        )
        self.assertIsNone(allowed)

    def test_written_feedback_skips_later_force(self):
        hook = EvaluatorDeliverableHook(max_iterations=3)
        state = AgentRuntimeState()
        hook.post_tool(
            "write_file",
            {"path": "feedback.md", "content": "Average: 8/10"},
            result_ok("Wrote 12 chars"),
            [],
            runtime_state=state,
        )
        self.assertIsNone(hook.per_iteration(3, [], runtime_state=state))
        self.assertIsNone(state.tool_choice)
        self.assertIsNone(
            hook.before_tool("browser_snapshot", {}, [], runtime_state=state)
        )

    def test_failed_write_does_not_clear_force(self):
        hook = EvaluatorDeliverableHook(max_iterations=1)
        state = AgentRuntimeState()
        hook.per_iteration(1, [], runtime_state=state)
        hook.post_tool(
            "write_file",
            {"path": "feedback.md"},
            result_error("[error] disk full"),
            [],
            runtime_state=state,
        )
        blocked = hook.before_tool("run_shell", {"command": "dir"}, [], runtime_state=state)
        self.assertIsNotNone(blocked)


class ExtractScoreTests(unittest.TestCase):
    def test_missing_feedback_is_none_not_zero(self):
        from harness.profiles.app_builder import AppBuilderProfile

        p = AppBuilderProfile()
        self.assertIsNone(p.extract_score(""))
        self.assertIsNone(p.extract_score("   "))
        self.assertIsNone(p.extract_score("no scores in here"))
        self.assertEqual(p.extract_score("**Average: 0/10**"), 0.0)
        self.assertEqual(p.extract_score("Average: 7.5 / 10"), 7.5)


if __name__ == "__main__":
    unittest.main()
