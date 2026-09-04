"""list_files should hide VCS metadata, not harness debug files."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import config
from tools import list_files


class ListFilesFilterTests(unittest.TestCase):
    def test_skips_dot_paths_but_shows_debug_files_if_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = config.WORKSPACE
            try:
                config.WORKSPACE = tmp
                root = Path(tmp)
                (root / "index.html").write_text("<html></html>", encoding="utf-8")
                (root / "messages.txt").write_text("secret", encoding="utf-8")
                (root / "_trace_builder.jsonl").write_text("{}\n", encoding="utf-8")
                git = root / ".git"
                git.mkdir()
                (git / "HEAD").write_text("ref: refs/heads/master\n", encoding="utf-8")

                listing = list_files(".").payload
                self.assertIn("index.html", listing)
                self.assertIn("messages.txt", listing)
                self.assertIn("_trace_builder.jsonl", listing)
                self.assertNotIn(".git", listing)
                self.assertNotIn("HEAD", listing)
            finally:
                config.WORKSPACE = old


if __name__ == "__main__":
    unittest.main()
