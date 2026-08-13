"""Tests for observation + lightweight trace compression."""
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

import config
from compression.observation import compress_observation
from compression.trace import (
    TRACE_MARKER,
    TraceBuffer,
    compress_trace,
    format_trace_summary,
    record_from_tool,
)
from pathlib import Path
import config


class ObservationCompressionTests(unittest.TestCase):
    def test_short_result_unchanged(self):
        out = compress_observation("run_bash", {"command": "ls"}, "ok\n")
        self.assertEqual(out, "ok\n")

    def test_long_result_head_tail(self):
        big = "A" * 20000
        out = compress_observation("read_file", {"path": "big.txt"}, big)
        self.assertIn("[OBSERVATION COMPRESSED]", out)
        self.assertTrue(out.startswith("A"))
        self.assertTrue(out.rstrip().endswith("A"))
        self.assertLess(len(out), len(big))
        self.assertIn("big.txt", out)


class TraceCompressionTests(unittest.TestCase):
    def test_record_from_tool_success_and_failure(self):
        ok = record_from_tool("write_file", {"path": "a.py"}, "Wrote 10 chars", iteration=2)
        self.assertTrue(ok.ok)
        self.assertEqual(ok.detail, "a.py")
        self.assertEqual(ok.iteration, 2)

        bad = record_from_tool("run_bash", {"command": "false"}, "[error] exit 1")
        self.assertFalse(bad.ok)
        self.assertIn("false", bad.detail)

    def test_format_trace_summary(self):
        buf = TraceBuffer()
        buf.add(record_from_tool("run_bash", {"command": "pwd"}, "/tmp"))
        buf.add(record_from_tool("write_file", {"path": "x"}, "[error] fail"))
        text = format_trace_summary(buf.records)
        self.assertTrue(text.startswith(TRACE_MARKER))
        self.assertIn("run_bash", text)
        self.assertIn("FAIL", text)

    def test_compress_trace_replaces_middle_keeps_tool_pairs(self):
        messages = [{"role": "system", "content": "sys"}]
        messages.append({"role": "user", "content": "[STATE MEMORY]\ngoal"})
        # Old turns
        for i in range(6):
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"c{i}",
                            "type": "function",
                            "function": {"name": "run_bash", "arguments": "{}"},
                        }
                    ],
                }
            )
            messages.append(
                {"role": "tool", "tool_call_id": f"c{i}", "content": f"out{i}"}
            )
        # Recent pair
        messages.append(
            {
                "role": "assistant",
                "content": "almost done",
                "tool_calls": [
                    {
                        "id": "recent",
                        "type": "function",
                        "function": {"name": "run_bash", "arguments": "{}"},
                    }
                ],
            }
        )
        messages.append({"role": "tool", "tool_call_id": "recent", "content": "final"})

        buf = TraceBuffer()
        for i in range(7):
            buf.add(record_from_tool("run_bash", {"command": f"c{i}"}, "ok", iteration=i))

        old_len = len(messages)
        # keep_recent small so middle is dropped
        out = compress_trace(messages, buf, keep_recent=2)
        self.assertLess(len(out), old_len)
        self.assertEqual(out[0]["role"], "system")
        # STATE preserved
        self.assertTrue(out[1]["content"].startswith("[STATE MEMORY]"))
        # TRACE SUMMARY present
        trace_msgs = [
            m for m in out if isinstance(m.get("content"), str) and m["content"].startswith(TRACE_MARKER)
        ]
        self.assertEqual(len(trace_msgs), 1)
        # Last tool pair intact
        self.assertEqual(out[-1]["role"], "tool")
        self.assertEqual(out[-1]["tool_call_id"], "recent")
        self.assertEqual(out[-2]["role"], "assistant")
        self.assertTrue(buf.compressed_through > 0)

    def test_compact_messages_removed(self):
        import context

        self.assertFalse(hasattr(context, "compact_messages"))
        self.assertFalse(hasattr(context, "create_checkpoint"))
        self.assertFalse(hasattr(context, "compress_state"))
        self.assertFalse(hasattr(context, "seed_task_board"))

    def test_full_compress_reset_clears_trace_buffer(self):
        import tempfile
        import shutil
        from compression.full import full_compress_reset

        tmp = tempfile.mkdtemp(prefix="full-reset-")
        old = config.WORKSPACE
        try:
            config.WORKSPACE = tmp
            buf = TraceBuffer()
            buf.add(record_from_tool("run_bash", {"command": "ls"}, "ok", iteration=1))
            buf.add(record_from_tool("write_file", {"path": "a"}, "Wrote 1", iteration=2))
            self.assertEqual(len(buf.records), 2)

            def fake_llm(_msgs):
                return "## Completed Work\n- done\n"

            out = full_compress_reset(
                [{"role": "user", "content": "history"}],
                "sys",
                fake_llm,
                trace_buffer=buf,
            )
            self.assertEqual(out[0]["role"], "system")
            self.assertEqual(len(buf.records), 0)
            self.assertEqual(buf.compressed_through, 0)
            self.assertTrue((Path(tmp) / config.HANDOFF_FILE).exists())
        finally:
            config.WORKSPACE = old
            shutil.rmtree(tmp, ignore_errors=True)

if __name__ == "__main__":
    unittest.main()
