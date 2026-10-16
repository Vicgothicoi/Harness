"""Inject time-budget warnings into the agent loop."""
from __future__ import annotations

import logging
import time

from harness.hooks.base import AgentHook

log = logging.getLogger("harness")

class TimeBudgetHook(AgentHook):
    """
    Injects time awareness into the agent loop.

    At configurable thresholds (default: 60% and 85% of budget),
    warns the agent about remaining time and nudges it toward
    wrapping up and verifying.

    Can track time from harness start (not just agent start) by calling
    sync_start_time() before the agent runs. This ensures the budget
    accounts for time already spent on planning/setup.
    """

    def __init__(
        self,
        budget_seconds: float,
        warn_threshold: float = 0.60,
        critical_threshold: float = 0.85,
    ):
        self.budget_seconds = budget_seconds
        self.warn_threshold = warn_threshold
        self.critical_threshold = critical_threshold
        self.start_time = time.time()
        self._warned = False
        self._critical = False

    def sync_start_time(self, harness_start: float):
        """Set start time to harness start, so budget includes planning/setup time."""
        self.start_time = harness_start

    def per_iteration(
        self,
        iteration: int,
        messages: list[dict],
        runtime_state=None,
        agent_name: str | None = None,
    ) -> str | None:
        elapsed = time.time() - self.start_time
        fraction = elapsed / self.budget_seconds
        remaining = self.budget_seconds - elapsed

        if remaining <= 0:
            if not self._critical:
                self._critical = True
                log.warning("Time budget EXPIRED")
                return (
                    "[SYSTEM] ⚠️ TIME IS UP. You have exceeded the time budget.\n"
                    "STOP immediately. Save whatever you have and finish NOW."
                )
            return None

        if fraction >= self.critical_threshold and not self._critical:
            self._critical = True
            mins_left = remaining / 60
            log.warning(f"Time budget critical: {mins_left:.1f} min remaining")
            return (
                f"[SYSTEM] ⚠️ CRITICAL: Only {mins_left:.1f} minutes remaining out of "
                f"{self.budget_seconds / 60:.0f} min budget.\n"
                "STOP building new features. Immediately:\n"
                "1. Verify what you've done so far works correctly.\n"
                "2. Run final checks against the task requirements.\n"
                "3. Fix any broken items — do NOT start anything new."
            )

        if fraction >= self.warn_threshold and not self._warned:
            self._warned = True
            mins_left = remaining / 60
            log.info(f"Time budget warning: {mins_left:.1f} min remaining")
            return (
                f"[SYSTEM] Time check: {mins_left:.1f} minutes remaining out of "
                f"{self.budget_seconds / 60:.0f} min budget. "
                "Start wrapping up your current work and plan for verification."
            )

        return None
