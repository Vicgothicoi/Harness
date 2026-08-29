"""BrowserSession marshals Playwright work onto one dedicated thread."""
from __future__ import annotations

import threading
import time
import unittest

from browser_session import BrowserSession


class BrowserSessionThreadTests(unittest.TestCase):
    def setUp(self):
        self.session = BrowserSession()

    def tearDown(self):
        self.session.shutdown()

    def test_invoke_runs_on_dedicated_worker_not_caller(self):
        main_id = threading.get_ident()
        worker_ids: list[int] = []

        def record() -> int:
            ident = threading.get_ident()
            worker_ids.append(ident)
            return ident

        first = self.session._invoke(record)
        second = self.session._invoke(record)

        self.assertEqual(first, second)
        self.assertNotEqual(first, main_id)
        self.assertEqual(worker_ids, [first, first])

    def test_concurrent_callers_share_the_same_worker_thread(self):
        barrier = threading.Barrier(3)
        seen: list[int] = []
        lock = threading.Lock()

        def record() -> int:
            ident = threading.get_ident()
            with lock:
                seen.append(ident)
            return ident

        def from_other_thread() -> None:
            barrier.wait()
            self.session._invoke(record)

        threads = [
            threading.Thread(target=from_other_thread),
            threading.Thread(target=from_other_thread),
        ]
        for t in threads:
            t.start()
        barrier.wait()
        self.session._invoke(record)
        for t in threads:
            t.join(timeout=5)

        self.assertEqual(len(seen), 3)
        self.assertEqual(len(set(seen)), 1)
        self.assertNotEqual(seen[0], threading.get_ident())

    def test_nested_invoke_on_worker_does_not_deadlock(self):
        def inner() -> str:
            return "inner"

        def outer() -> str:
            return f"outer-{self.session._invoke(inner)}"

        self.assertEqual(self.session._invoke(outer), "outer-inner")

    def test_jobs_are_serialized_on_the_worker(self):
        order: list[int] = []

        def hold(n: int) -> int:
            time.sleep(0.05)
            order.append(n)
            return n

        results: list[int] = []

        def call(n: int) -> None:
            results.append(self.session._invoke(hold, n))

        threads = [threading.Thread(target=call, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        self.assertEqual(sorted(results), [0, 1, 2, 3])
        self.assertEqual(order, results)


class EvaluatorPromptFallbackTests(unittest.TestCase):
    def test_evaluator_must_write_feedback_when_browser_unavailable(self):
        from prompts import EVALUATOR_SYSTEM

        text = EVALUATOR_SYSTEM.lower()
        self.assertIn("browser unavailable", text)
        self.assertIn("feedback.md", text)
        self.assertIn("do not install, repair, debug", text)
        self.assertIn("code review", text)


if __name__ == "__main__":
    unittest.main()
