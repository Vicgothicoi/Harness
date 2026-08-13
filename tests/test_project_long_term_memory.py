"""Tests for project memory and long-term memory (no RAG yet)."""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path


def _install_fake_openai_module() -> None:
    openai = types.ModuleType("openai")

    class OpenAI:
        def __init__(self, *args, **kwargs):
            pass

    openai.OpenAI = OpenAI
    sys.modules["openai"] = openai


_install_fake_openai_module()

import config
from memory.inject import upsert_marked_block
from memory.long_term_memory import (
    LONG_TERM_MARKER,
    LongTermMemory,
    inject_long_term_preferences,
    learn_from_task,
)
from memory.project_memory import (
    PROJECT_MARKER,
    ProjectMemory,
    inject_project_summary,
    refresh_project_memory,
    seed_project_memory,
)
from memory.state_memory import STATE_MARKER, apply_patch, inject_state_summary
from agents import TaskBoard


class ProjectMemoryTests(unittest.TestCase):
    def setUp(self):
        self._old_ws = config.WORKSPACE
        self._tmpdir = tempfile.mkdtemp(prefix="pm-test-")
        config.WORKSPACE = self._tmpdir

    def tearDown(self):
        config.WORKSPACE = self._old_ws
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_seed_and_merge_persist(self):
        pm = seed_project_memory("Build a todo app")
        self.assertEqual(pm.source_prompt, "Build a todo app")
        path = Path(self._tmpdir) / config.PROJECT_MEMORY_FILE
        self.assertTrue(path.exists())

        pm.merge_delta(
            {
                "tech_stack": ["vanilla JS"],
                "architecture": "SPA with localStorage",
                "key_files": {"app.js": "state and render"},
                "decisions": [{"what": "use todo.v1 key", "why": "migration"}],
                "known_issues": ["filter flash"],
                "round_summary": {"summary": "scaffold done", "artifacts": ["app.js"], "score": 6.0},
            },
            round_num=1,
        )
        pm.save()
        loaded = ProjectMemory.load()
        self.assertIn("vanilla JS", loaded.tech_stack)
        self.assertEqual(loaded.key_files["app.js"], "state and render")
        self.assertEqual(len(loaded.round_summaries), 1)
        self.assertEqual(loaded.round_summaries[0]["score"], 6.0)

    def test_context_block_and_inject_order(self):
        pm = ProjectMemory(
            source_prompt="todo",
            tech_stack=["js"],
            architecture="spa",
            key_files={"a.js": "core"},
        )
        block = pm.to_context_block()
        self.assertTrue(block.startswith(PROJECT_MARKER))

        board = TaskBoard(goal="g")
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
        messages = inject_state_summary(messages, board)
        messages = inject_project_summary(messages, pm)
        markers = [
            m["content"][:20]
            for m in messages
            if m.get("role") == "user" and isinstance(m.get("content"), str)
        ]
        # system, STATE, PROJECT, original task
        self.assertTrue(messages[1]["content"].startswith(STATE_MARKER))
        self.assertTrue(messages[2]["content"].startswith(PROJECT_MARKER))

    def test_refresh_project_memory_with_fake_llm(self):
        Path(self._tmpdir, "spec.md").write_text("# Task\nBuild x\n", encoding="utf-8")
        Path(self._tmpdir, "app.js").write_text("console.log(1)\n", encoding="utf-8")
        seed_project_memory("Build x")

        def fake_llm(_msgs):
            return json.dumps(
                {
                    "tech_stack": ["node"],
                    "architecture": "single file",
                    "key_files": {"app.js": "entry"},
                    "round_summary": {"summary": "wrote app.js", "artifacts": ["app.js"]},
                }
            )

        pm = refresh_project_memory("Build x", 1, fake_llm, score=7.5)
        self.assertIn("node", pm.tech_stack)
        self.assertEqual(pm.key_files.get("app.js"), "entry")
        self.assertEqual(pm.round_summaries[-1]["score"], 7.5)


class LongTermMemoryTests(unittest.TestCase):
    def setUp(self):
        self._old_dir = config.LONG_TERM_MEMORY_DIR
        self._tmpdir = tempfile.mkdtemp(prefix="ltm-test-")
        config.LONG_TERM_MEMORY_DIR = self._tmpdir
        self._old_ws = config.WORKSPACE
        self._ws = tempfile.mkdtemp(prefix="ltm-ws-")
        config.WORKSPACE = self._ws

    def tearDown(self):
        config.LONG_TERM_MEMORY_DIR = self._old_dir
        config.WORKSPACE = self._old_ws
        shutil.rmtree(self._tmpdir, ignore_errors=True)
        shutil.rmtree(self._ws, ignore_errors=True)

    def test_preference_persist_and_inject(self):
        ltm = LongTermMemory.load()
        ltm.set_preference("ui_language", "zh-CN")
        ltm.save()

        loaded = LongTermMemory.load()
        self.assertEqual(loaded.user_preferences["ui_language"], "zh-CN")
        block = loaded.preferences_context_block()
        self.assertTrue(block.startswith(LONG_TERM_MARKER))
        self.assertIn("zh-CN", block)

        messages = [{"role": "system", "content": "sys"}]
        messages = inject_long_term_preferences(messages, loaded)
        self.assertTrue(messages[1]["content"].startswith(LONG_TERM_MARKER))

    def test_learn_from_task_stores_patterns(self):
        seed_project_memory("todo spa")
        Path(self._ws, config.FEEDBACK_FILE).write_text("Average: 8/10\n", encoding="utf-8")

        def fake_llm(_msgs):
            return json.dumps(
                {
                    "patterns": [
                        {
                            "trigger": "localStorage spa",
                            "lesson": "Use versioned storage keys",
                            "anti_pattern": "save on every render",
                            "confidence": 0.8,
                            "tags": ["frontend"],
                        }
                    ],
                    "anti_patterns": ["hardcode absolute paths"],
                }
            )

        ltm = learn_from_task(
            "todo spa", fake_llm, passed=True, score_history=[8.0], memory_dir=self._tmpdir
        )
        self.assertEqual(len(ltm.patterns), 1)
        self.assertIn("versioned", ltm.patterns[0]["lesson"])
        self.assertTrue(LongTermMemory.path(self._tmpdir).exists())

    def test_remember_preference_tool(self):
        from tools import remember_preference

        result = remember_preference("test_framework", "pytest")
        self.assertIn("Saved preference", result)
        ltm = LongTermMemory.load()
        self.assertEqual(ltm.user_preferences["test_framework"], "pytest")


class InjectHelperTests(unittest.TestCase):
    def test_upsert_replaces_not_appends(self):
        messages = [{"role": "system", "content": "s"}]
        messages = upsert_marked_block(messages, "[X]", "[X]\none")
        messages = upsert_marked_block(messages, "[X]", "[X]\ntwo")
        xs = [m for m in messages if str(m.get("content", "")).startswith("[X]")]
        self.assertEqual(len(xs), 1)
        self.assertIn("two", xs[0]["content"])


if __name__ == "__main__":
    unittest.main()
