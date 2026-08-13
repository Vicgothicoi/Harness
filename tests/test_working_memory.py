"""Tests for working memory Context Builder."""
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

from agents import TaskBoard
from memory.long_term_memory import LONG_TERM_MARKER, LongTermMemory
from memory.project_memory import PROJECT_MARKER, ProjectMemory
from memory.state_memory import STATE_MARKER, apply_patch
from memory.working_memory import build_working_memory


class WorkingMemoryTests(unittest.TestCase):
    def test_build_order_state_project_ltm(self):
        board = TaskBoard()
        apply_patch(
            board,
            {
                "goal": "g",
                "steps": ["a"],
                "current_step": "a",
                "completed_steps": [],
                "blockers": [],
                "next_action": "x",
            },
        )
        pm = ProjectMemory(project_id="p", architecture="svc", tech_stack=["py"])
        ltm = LongTermMemory(user_preferences={"style": "concise"})

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "do the task"},
        ]
        out = build_working_memory(
            messages,
            task_board=board,
            project_memory=pm,
            long_term_memory=ltm,
            load_defaults=False,
        )

        self.assertEqual(out[0]["role"], "system")
        self.assertTrue(out[1]["content"].startswith(STATE_MARKER))
        self.assertTrue(out[2]["content"].startswith(PROJECT_MARKER))
        self.assertTrue(out[3]["content"].startswith(LONG_TERM_MARKER))
        self.assertEqual(out[4]["content"], "do the task")

    def test_replace_in_place_no_duplicates(self):
        board = TaskBoard()
        apply_patch(
            board,
            {
                "goal": "g",
                "steps": ["a"],
                "current_step": "a",
                "completed_steps": [],
                "blockers": [],
                "next_action": "x",
            },
        )
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
        ]
        empty_pm = ProjectMemory()
        empty_ltm = LongTermMemory()
        once = build_working_memory(
            messages,
            task_board=board,
            project_memory=empty_pm,
            long_term_memory=empty_ltm,
            load_defaults=False,
        )
        twice = build_working_memory(
            once,
            task_board=board,
            project_memory=empty_pm,
            long_term_memory=empty_ltm,
            load_defaults=False,
        )
        state_count = sum(
            1
            for m in twice
            if isinstance(m.get("content"), str) and m["content"].startswith(STATE_MARKER)
        )
        self.assertEqual(state_count, 1)
        self.assertEqual(len(twice), len(once))


if __name__ == "__main__":
    unittest.main()
