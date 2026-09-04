"""Tests for structured ToolResult / ShellObservation."""
from __future__ import annotations

import unittest

from tool_result import (
    ShellObservation,
    ToolResult,
    from_shell,
    result_blocked,
    result_error,
    result_ok,
    wrap_legacy,
)


class ToolResultTests(unittest.TestCase):
    def test_ok_is_protocol_and_task_success(self):
        r = result_ok("Wrote 10 chars")
        self.assertTrue(r.protocol_ok)
        self.assertTrue(r.task_ok)
        self.assertEqual(r.kind, "ok")

    def test_error_is_neither_protocol_nor_task_success(self):
        r = result_error("[error] File not found: x")
        self.assertFalse(r.protocol_ok)
        self.assertFalse(r.task_ok)
        self.assertEqual(r.kind, "error")
        self.assertEqual(r.payload, "[error] File not found: x")

    def test_blocked_is_not_protocol_ok(self):
        r = result_blocked("[blocked] Recovery mode ENV_FIX ...")
        self.assertFalse(r.protocol_ok)
        self.assertFalse(r.task_ok)
        self.assertEqual(r.kind, "blocked")

    def test_command_failed_ran_but_did_not_succeed(self):
        obs = ShellObservation(stdout="FAILED", stderr="", exit_code=1)
        r = from_shell(obs)
        self.assertEqual(r.kind, "command_failed")
        self.assertTrue(r.protocol_ok)
        self.assertFalse(r.task_ok)
        self.assertEqual(r.exit_code, 1)
        self.assertIs(r.payload, obs)

    def test_timeout_is_protocol_error(self):
        obs = ShellObservation(exit_code=130, timed_out=True)
        r = from_shell(obs)
        self.assertEqual(r.kind, "error")
        self.assertFalse(r.protocol_ok)
        self.assertIsInstance(r.payload, ShellObservation)
        self.assertTrue(r.payload.timed_out)

    def test_zero_exit_is_ok(self):
        r = from_shell(ShellObservation(stdout="ok", exit_code=0))
        self.assertEqual(r.kind, "ok")
        self.assertTrue(r.task_ok)
        self.assertEqual(r.exit_code, 0)

    def test_payload_text_joins_stderr(self):
        r = ToolResult(
            kind="command_failed",
            payload=ShellObservation(stdout="out", stderr="err", exit_code=1),
            exit_code=1,
        )
        self.assertIn("out", r.payload_text())
        self.assertIn("--- STDERR ---", r.payload_text())
        self.assertIn("err", r.payload_text())

    def test_wrap_legacy_prefixes(self):
        self.assertEqual(wrap_legacy("[error] boom").kind, "error")
        self.assertEqual(wrap_legacy("[blocked] no").kind, "blocked")
        self.assertEqual(wrap_legacy("Wrote 3 chars").kind, "ok")
        already = result_ok("x")
        self.assertIs(wrap_legacy(already), already)

    def test_wrap_legacy_shell_uses_exit_code(self):
        r = wrap_legacy(ShellObservation(stderr="fail", exit_code=2))
        self.assertEqual(r.kind, "command_failed")
        self.assertEqual(r.exit_code, 2)

    def test_with_auto_fix_sets_field_not_payload(self):
        r = result_ok("Wrote 1").with_auto_fix("[auto-fix] path")
        self.assertEqual(r.auto_fix, "[auto-fix] path")
        self.assertEqual(r.payload, "Wrote 1")


class ExecuteToolTests(unittest.TestCase):
    def test_unknown_tool_is_error(self):
        from tools import execute_tool

        r = execute_tool("no_such_tool", {})
        self.assertEqual(r.kind, "error")
        self.assertIn("Unknown tool", r.payload)

    def test_empty_shell_command_is_error(self):
        from tools import execute_tool

        r = execute_tool("run_shell", {"command": "  "})
        self.assertEqual(r.kind, "error")
        self.assertIn("Empty command", r.payload)

    def test_run_shell_without_session_is_error(self):
        from tools import execute_tool

        r = execute_tool("run_shell", {"command": "echo hi"})
        self.assertEqual(r.kind, "error")
        self.assertIn("No active shell session", r.payload)


if __name__ == "__main__":
    unittest.main()
