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

from agents import AgentRuntimeState
from hooks import RecoveryStrategyHook
from tool_result import ShellObservation, from_shell, result_error


class RecoveryStrategyTests(unittest.TestCase):
    def test_repeated_environment_failures_switch_to_env_fix(self):
        state = AgentRuntimeState()
        hook = RecoveryStrategyHook()

        hook.observe_tool_result("run_shell", {"command": "foo"}, "[error] command not found", state)
        hook.observe_tool_result("run_shell", {"command": "foo"}, "[error] command not found", state)

        self.assertEqual(state.recovery.mode, "ENV_FIX")

    def test_command_not_found_exit_is_env_fix(self):
        state = AgentRuntimeState()
        hook = RecoveryStrategyHook()
        failed = from_shell(
            ShellObservation(stderr="foo: command not found", exit_code=127)
        )
        hook.observe_tool_result("run_shell", {"command": "foo"}, failed, state)
        hook.observe_tool_result("run_shell", {"command": "foo"}, failed, state)
        self.assertEqual(state.recovery.mode, "ENV_FIX")

    def test_pytest_failure_is_not_env_fix(self):
        state = AgentRuntimeState()
        hook = RecoveryStrategyHook()
        failed = from_shell(ShellObservation(stdout="FAILED tests/test_x.py", exit_code=1))
        hook.observe_tool_result("run_shell", {"command": "pytest"}, failed, state)
        hook.observe_tool_result("run_shell", {"command": "pytest"}, failed, state)
        self.assertEqual(state.recovery.mode, "NORMAL")
        self.assertNotEqual(state.recovery.last_successful_action, "run_shell")

    def test_protocol_error_still_counts(self):
        state = AgentRuntimeState()
        hook = RecoveryStrategyHook()
        err = result_error("[error] No active shell session")
        hook.observe_tool_result("run_shell", {"command": "ls"}, err, state)
        hook.observe_tool_result("run_shell", {"command": "ls"}, err, state)
        self.assertEqual(state.recovery.mode, "SPEC_RECHECK")

    def test_repeated_same_failure_switches_to_spec_recheck(self):
        state = AgentRuntimeState()
        hook = RecoveryStrategyHook()

        hook.observe_verification_failure("pytest::task_x failed", state)
        hook.observe_verification_failure("pytest::task_x failed", state)

        self.assertEqual(state.recovery.mode, "SPEC_RECHECK")

    def test_repeated_edits_with_same_failure_switch_to_rethink(self):
        state = AgentRuntimeState()
        hook = RecoveryStrategyHook()

        hook.observe_verification_failure("assert result.txt mismatch", state)
        hook.observe_verification_failure("assert result.txt mismatch", state)
        hook.observe_edit_attempt("result.txt", state)
        hook.observe_edit_attempt("result.txt", state)

        self.assertEqual(state.recovery.mode, "RETHINK")
        self.assertTrue(state.task_board.requires_update)


if __name__ == "__main__":
    unittest.main()
