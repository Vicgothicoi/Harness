"""read_file start_line windowing."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from harness import config
from harness.tools import execute_tool, read_file


class ReadFileStartLineTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old_ws = config.WORKSPACE
        config.WORKSPACE = self._tmp.name
        self._path = Path(self._tmp.name) / "sample.txt"
        self._path.write_text(
            "\n".join(f"line-{i}" for i in range(1, 6)),
            encoding="utf-8",
        )

    def tearDown(self):
        config.WORKSPACE = self._old_ws
        self._tmp.cleanup()

    def test_default_reads_from_first_line(self):
        result = read_file("sample.txt")
        self.assertEqual(result.kind, "ok")
        self.assertEqual(
            result.payload, "\n".join(f"line-{i}" for i in range(1, 6))
        )

    def test_start_line_skips_earlier_lines(self):
        result = read_file("sample.txt", start_line=3)
        self.assertEqual(result.kind, "ok")
        self.assertEqual(result.payload, "line-3\nline-4\nline-5")

    def test_start_line_string_is_coerced(self):
        result = read_file("sample.txt", start_line="4")
        self.assertEqual(result.kind, "ok")
        self.assertEqual(result.payload, "line-4\nline-5")

    def test_start_line_past_eof_is_error(self):
        result = read_file("sample.txt", start_line=6)
        self.assertEqual(result.kind, "error")
        self.assertIn("past end of file", result.payload)
        self.assertIn("5 lines", result.payload)

    def test_start_line_zero_is_error(self):
        result = read_file("sample.txt", start_line=0)
        self.assertEqual(result.kind, "error")
        self.assertIn("start_line must be >= 1", result.payload)

    def test_execute_tool_passes_start_line(self):
        result = execute_tool("read_file", {"path": "sample.txt", "start_line": 2})
        self.assertEqual(result.kind, "ok")
        self.assertEqual(result.payload, "line-2\nline-3\nline-4\nline-5")


if __name__ == "__main__":
    unittest.main()
