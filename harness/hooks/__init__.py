"""Composable hooks. Profiles attach the ones they need.

before_tool, post_tool, per_iteration, and pre_exit return an optional
message to inject. None means no intervention.
"""

from harness.hooks.base import AgentHook
from harness.hooks.error_guidance import ErrorGuidanceHook
from harness.hooks.evaluator_deliverable import EvaluatorDeliverableHook
from harness.hooks.loop import LoopDetectionHook
from harness.hooks.pre_exit import PreExitVerificationHook
from harness.hooks.recovery import RecoveryState, RecoveryStrategyHook
from harness.hooks.time_budget import TimeBudgetHook

__all__ = [
    "AgentHook",
    "ErrorGuidanceHook",
    "EvaluatorDeliverableHook",
    "LoopDetectionHook",
    "PreExitVerificationHook",
    "RecoveryState",
    "RecoveryStrategyHook",
    "TimeBudgetHook",
]
