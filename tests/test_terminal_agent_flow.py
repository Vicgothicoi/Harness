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

from agents import Agent
from hooks import RecoveryStrategyHook
from profiles.terminal import TerminalProfile


class AgentRuntimeStateTests(unittest.TestCase):
    def test_agent_creates_runtime_state_once_per_run(self):
        agent = Agent(name="builder", system_prompt="x", use_tools=False)
        state = agent._create_runtime_state("goal text")

        self.assertEqual(state.task_board.goal, "goal text")
        self.assertEqual(state.recovery.mode, "NORMAL")
        self.assertIsNone(state.shell_session)

    def test_terminal_builder_prompt_mentions_stateful_shell_state_memory_and_recovery(self):
        prompt = TerminalProfile().builder().system_prompt

        self.assertIn("persistent shell", prompt.lower())
        self.assertIn("state memory", prompt.lower())
        self.assertNotIn("update_progress", prompt)
        self.assertIn("recovery mode", prompt.lower())

    def test_only_builder_enables_memory(self):
        from profiles.app_builder import AppBuilderProfile

        terminal = TerminalProfile()
        self.assertFalse(terminal.planner().enable_memory)
        self.assertTrue(terminal.builder().enable_memory)
        self.assertFalse(terminal.evaluator().enable_memory)

        app = AppBuilderProfile()
        self.assertFalse(app.planner().enable_memory)
        self.assertTrue(app.builder().enable_memory)
        self.assertFalse(app.evaluator().enable_memory)

    def test_terminal_builder_uses_recovery_hook_without_enforcement(self):
        hooks = TerminalProfile().builder().hooks

        self.assertTrue(any(isinstance(h, RecoveryStrategyHook) for h in hooks))
        names = [type(h).__name__ for h in hooks]
        self.assertNotIn("TaskTrackingEnforcementHook", names)


if __name__ == "__main__":
    unittest.main()
