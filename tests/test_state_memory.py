"""Tests for state memory + automatic state compression (replaces update_progress)."""
from __future__ import annotations

import os
import shutil
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock


def _install_fake_openai_module() -> None:
    openai = types.ModuleType("openai")

    class OpenAI:
        def __init__(self, *args, **kwargs):
            pass

    openai.OpenAI = OpenAI
    sys.modules["openai"] = openai


_install_fake_openai_module()

import config
from agents import Agent, AgentRuntimeState, TaskBoard, TraceWriter
from memory.state_memory import (
    STATE_MARKER,
    apply_patch,
    inject_state_summary,
    persist_task_board,
)
from compression.state import compress_state
from hooks import RecoveryStrategyHook


class StateMemoryTests(unittest.TestCase):
    def test_persist_and_inject_round_trip(self):
        temp_dir = os.path.join(os.getcwd(), "workspace", "test-state-memory")
        old_workspace = config.WORKSPACE
        try:
            os.makedirs(temp_dir, exist_ok=True)
            config.WORKSPACE = temp_dir

            board = TaskBoard()
            apply_patch(
                board,
                {
                    "goal": "fix task",
                    "steps": ["inspect", "edit", "verify"],
                    "current_step": "inspect",
                    "completed_steps": [],
                    "blockers": [],
                    "next_action": "read tests",
                },
            )
            path = persist_task_board(board)
            self.assertTrue(path.exists())
            text = path.read_text(encoding="utf-8")
            self.assertIn("fix task", text)
            self.assertIn("inspect", text)
            self.assertEqual(board.update_count, 1)

            messages = [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "task"},
            ]
            messages = inject_state_summary(messages, board)
            self.assertEqual(len(messages), 3)
            self.assertTrue(messages[1]["content"].startswith(STATE_MARKER))

            # Second inject replaces, does not append
            board.next_action = "edit file"
            messages = inject_state_summary(messages, board)
            state_msgs = [
                m for m in messages if isinstance(m.get("content"), str) and m["content"].startswith(STATE_MARKER)
            ]
            self.assertEqual(len(state_msgs), 1)
            self.assertIn("edit file", state_msgs[0]["content"])
        finally:
            config.WORKSPACE = old_workspace
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_compress_state_applies_llm_patch(self):
        temp_dir = os.path.join(os.getcwd(), "workspace", "test-state-compress")
        old_workspace = config.WORKSPACE
        try:
            os.makedirs(temp_dir, exist_ok=True)
            config.WORKSPACE = temp_dir

            board = TaskBoard(goal="old")
            messages = [
                {"role": "user", "content": "do the work"},
                {"role": "assistant", "content": "I will edit result.txt"},
            ]

            def fake_llm(_msgs):
                return (
                    '{"goal":"ship feature","steps":["a","b"],'
                    '"current_step":"a","completed_steps":[],'
                    '"blockers":[],"next_action":"write result.txt"}'
                )

            patch = compress_state(messages, board, fake_llm)
            self.assertIsNotNone(patch)
            self.assertEqual(board.goal, "ship feature")
            self.assertEqual(board.next_action, "write result.txt")
            self.assertTrue(Path(temp_dir, config.PROGRESS_FILE).exists())
        finally:
            config.WORKSPACE = old_workspace
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_checkpoint_writes_handoff_not_progress(self):
        temp_dir = os.path.join(os.getcwd(), "workspace", "test-handoff")
        old_workspace = config.WORKSPACE
        try:
            os.makedirs(temp_dir, exist_ok=True)
            config.WORKSPACE = temp_dir

            board = TaskBoard(goal="keep me")
            apply_patch(
                board,
                {
                    "goal": "keep me",
                    "steps": ["one"],
                    "current_step": "one",
                    "completed_steps": [],
                    "blockers": [],
                    "next_action": "continue",
                },
            )
            persist_task_board(board)
            progress_before = Path(temp_dir, config.PROGRESS_FILE).read_text(encoding="utf-8")

            def fake_llm(_msgs):
                return "## Completed Work\n- foo.py\n"

            from compression.full import create_checkpoint, restore_from_checkpoint

            messages = [{"role": "user", "content": "long history"}]
            checkpoint = create_checkpoint(messages, fake_llm)
            self.assertIn("foo.py", checkpoint)
            self.assertTrue(Path(temp_dir, config.HANDOFF_FILE).exists())
            # progress.md must not be overwritten by checkpoint
            progress_after = Path(temp_dir, config.PROGRESS_FILE).read_text(encoding="utf-8")
            self.assertEqual(progress_before, progress_after)
            self.assertIn("keep me", progress_after)

            restored = restore_from_checkpoint(
                checkpoint, "system", task_board=board
            )
            self.assertEqual(restored[0]["role"], "system")
            # restore itself does not inject memory; Context Builder does
            self.assertFalse(
                any(
                    isinstance(m.get("content"), str)
                    and m["content"].startswith(STATE_MARKER)
                    for m in restored
                )
            )
            from memory.working_memory import build_working_memory
            from memory.project_memory import ProjectMemory
            from memory.long_term_memory import LongTermMemory

            restored = build_working_memory(
                restored,
                task_board=board,
                project_memory=ProjectMemory(),
                long_term_memory=LongTermMemory(),
                load_defaults=False,
            )
            state_msgs = [
                m for m in restored if isinstance(m.get("content"), str) and m["content"].startswith(STATE_MARKER)
            ]
            self.assertEqual(len(state_msgs), 1)
        finally:
            config.WORKSPACE = old_workspace
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_agent_refresh_seeds_then_compresses(self):
        temp_dir = os.path.join(os.getcwd(), "workspace", "test-agent-state")
        old_workspace = config.WORKSPACE
        try:
            os.makedirs(temp_dir, exist_ok=True)
            config.WORKSPACE = temp_dir

            agent = Agent(name="builder", system_prompt="sys", use_tools=False)
            state = agent._create_runtime_state("build a CLI")
            trace = MagicMock(spec=TraceWriter)

            messages = [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "build a CLI"},
            ]
            messages = agent._refresh_state_memory(
                messages, state, "build a CLI", iteration=1, trace=trace
            )
            self.assertGreaterEqual(state.task_board.update_count, 1)
            self.assertTrue(Path(temp_dir, config.PROGRESS_FILE).exists())
            self.assertTrue(any(
                isinstance(m.get("content"), str) and m["content"].startswith(STATE_MARKER)
                for m in messages
            ))

            # Force iteration 2 compression with a stub llm_call_simple
            import agents as agents_mod

            original = agents_mod.llm_call_simple

            def fake_llm(_msgs):
                return (
                    '{"goal":"build a CLI","steps":["design","code","test"],'
                    '"current_step":"code","completed_steps":["design"],'
                    '"blockers":[],"next_action":"implement main"}'
                )

            agents_mod.llm_call_simple = fake_llm
            try:
                messages = agent._refresh_state_memory(
                    messages, state, "build a CLI", iteration=2, trace=trace
                )
            finally:
                agents_mod.llm_call_simple = original

            self.assertEqual(state.task_board.current_step, "code")
            self.assertEqual(state.task_board.next_action, "implement main")
        finally:
            config.WORKSPACE = old_workspace
            shutil.rmtree(temp_dir, ignore_errors=True)


class RecoveryWithoutUpdateProgressTests(unittest.TestCase):
    def test_rethink_blocks_until_requires_update_cleared(self):
        state = AgentRuntimeState()
        hook = RecoveryStrategyHook()

        hook.observe_verification_failure("assert result.txt mismatch", state)
        hook.observe_verification_failure("assert result.txt mismatch", state)
        hook.observe_edit_attempt("result.txt", state)
        hook.observe_edit_attempt("result.txt", state)

        self.assertEqual(state.recovery.mode, "RETHINK")
        self.assertTrue(state.task_board.requires_update)

        blocked = hook.before_tool(
            "run_bash",
            {"command": "pwd"},
            messages=[],
            runtime_state=state,
            agent_name="builder",
        )
        self.assertIsNotNone(blocked)
        self.assertNotIn("update_progress", blocked)

        # Simulate state compression clearing RETHINK
        state.task_board.requires_update = False
        state.recovery.mode = "NORMAL"
        blocked = hook.before_tool(
            "run_bash",
            {"command": "pwd"},
            messages=[],
            runtime_state=state,
            agent_name="builder",
        )
        self.assertIsNone(blocked)

    def test_update_progress_tool_removed(self):
        from tools import TOOL_DISPATCH, TOOL_SCHEMAS

        self.assertNotIn("update_progress", TOOL_DISPATCH)
        names = [s["function"]["name"] for s in TOOL_SCHEMAS]
        self.assertNotIn("update_progress", names)


if __name__ == "__main__":
    unittest.main()
