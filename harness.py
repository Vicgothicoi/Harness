#!/usr/bin/env python3
"""
Harness — profile-driven multi-agent architecture for autonomous task execution.

Reproduces the design from Anthropic's "Harness design for long-running
application development" using a pure Python + OpenAI-compatible API approach.

The core loop (Plan → Build → Evaluate → Iterate) is generic.
Profiles define the scenario-specific behavior (prompts, tools, scoring).

Built-in profiles:
  app-builder  — Build web apps from a prompt (original Anthropic article scenario)
  terminal     — Solve terminal/CLI tasks (Terminal-Bench-2 style)
  swe-bench    — Fix GitHub issues in real repos
  reasoning    — Knowledge-intensive QA (MMMU-Pro style)

Usage:
  python harness.py "Build a DAW in the browser"                    # default: app-builder
  python harness.py --profile terminal "Fix the broken git merge"
  python harness.py --profile swe-bench "Fix issue #123"
  python harness.py --profile reasoning "Calculate the orbital period of..."
  python harness.py --list-profiles
"""
from __future__ import annotations

import logging
import os
import re
import sys
import time
from pathlib import Path

import config
from agents import Agent, llm_call_simple
from skills import SkillRegistry
from profiles import get_profile, list_profiles
from profiles.base import BaseProfile
from memory.project_memory import refresh_project_memory, seed_project_memory
from memory.long_term_memory import learn_from_task

log = logging.getLogger("harness")


class Harness:
    """
    Generic orchestration loop driven by a Profile.
    """

    def __init__(self, profile: BaseProfile):
        self.profile = profile
        self.skill_registry = SkillRegistry()
        skill_catalog = self.skill_registry.build_catalog_prompt()

        # Build agents from profile config
        planner_cfg = profile.planner()
        builder_cfg = profile.builder()
        evaluator_cfg = profile.evaluator()
        proposer_cfg = profile.contract_proposer()
        reviewer_cfg = profile.contract_reviewer()

        self.planner = Agent(
            "planner", planner_cfg.system_prompt + skill_catalog,
            use_tools=True, extra_tool_schemas=planner_cfg.extra_tool_schemas,
            hooks=planner_cfg.hooks, time_budget=planner_cfg.time_budget,
            mcp_bridges=planner_cfg.mcp_bridges,
        ) if planner_cfg.enabled else None

        self.builder = Agent(
            "builder", builder_cfg.system_prompt + skill_catalog,
            use_tools=True, extra_tool_schemas=builder_cfg.extra_tool_schemas,
            hooks=builder_cfg.hooks, time_budget=builder_cfg.time_budget,
            mcp_bridges=builder_cfg.mcp_bridges,
        )

        self.evaluator = Agent(
            "evaluator", evaluator_cfg.system_prompt,
            use_tools=True, extra_tool_schemas=evaluator_cfg.extra_tool_schemas,
            hooks=evaluator_cfg.hooks, time_budget=evaluator_cfg.time_budget,
            mcp_bridges=evaluator_cfg.mcp_bridges,
        ) if evaluator_cfg.enabled else None

        self.contract_proposer = Agent(
            "contract_proposer", proposer_cfg.system_prompt, use_tools=True,
            hooks=proposer_cfg.hooks,
            mcp_bridges=proposer_cfg.mcp_bridges,
        ) if proposer_cfg.enabled else None

        self.contract_reviewer = Agent(
            "contract_reviewer", reviewer_cfg.system_prompt, use_tools=True,
            hooks=reviewer_cfg.hooks,
            mcp_bridges=reviewer_cfg.mcp_bridges,
        ) if reviewer_cfg.enabled else None

    def run(self, user_prompt: str) -> None:
        # Create a unique project subdirectory under workspace
        # (skip if HARNESS_FLAT_WORKSPACE is set — used for benchmarks
        #  where outputs must land directly in the workspace root)
        if os.environ.get("HARNESS_FLAT_WORKSPACE"):
            Path(config.WORKSPACE).mkdir(parents=True, exist_ok=True)
        else:
            from datetime import datetime
            # slug = re.sub(r'[^a-z0-9]+', '-', user_prompt.lower().strip())[:40].strip('-')
            slug = re.sub(r'[\\/:*?"<>|]', "_", user_prompt).strip()[:20]
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            project_name = f"{timestamp}_{slug}"
            project_dir = os.path.join(config.WORKSPACE, project_name)

            config.WORKSPACE = os.path.abspath(project_dir)
            Path(config.WORKSPACE).mkdir(parents=True, exist_ok=True)

        log.info(f"Profile: {self.profile.name()}")
        log.info(f"Project directory: {config.WORKSPACE}")

        # Initialize git
        git_dir = Path(config.WORKSPACE) / ".git"
        if not git_dir.exists():
            os.system(f"cd {config.WORKSPACE} && git init && git add -A 2>/dev/null; git commit -m 'init' --allow-empty 2>/dev/null")

        total_start = time.time()
        max_rounds = self.profile.max_rounds() or config.MAX_HARNESS_ROUNDS
        threshold = self.profile.pass_threshold() or config.PASS_THRESHOLD

        # ---- Resolve dynamic time allocation ----
        allocation = self.profile.resolve_time_allocation(user_prompt)
        skip_planner = not allocation.get("planner_enabled", True)
        skip_evaluator = not allocation.get("evaluator_enabled", True)
        log.info(f"Time allocation: planner={allocation['planner']:.0%} "
                 f"builder={allocation['builder']:.0%} "
                 f"evaluator={allocation['evaluator']:.0%} "
                 f"(planner={'skip' if skip_planner else 'on'}, "
                 f"evaluator={'skip' if skip_evaluator else 'on'})")

        # ---- Phase 1: Planning ----
        if self.planner and not skip_planner:
            log.info("=" * 60)
            log.info("PHASE 1: PLANNING")
            log.info("=" * 60)
            phase_start = time.time()

            self.planner.run(
                f"Create a plan for the following task:\n\n"
                f"{user_prompt}\n\n"
                f"Save the plan to spec.md."
            )

            log.info(f"Planning completed in {time.time() - phase_start:.0f}s")
        else:
            # No planner — write prompt directly as spec
            spec_path = Path(config.WORKSPACE) / config.SPEC_FILE
            spec_path.write_text(f"# Task\n\n{user_prompt}\n", encoding="utf-8")
            log.info("No planner — wrote prompt directly to spec.md")

        # Seed project memory for this workspace (updated again after each build)
        seed_project_memory(user_prompt)
        log.info(f"Project memory seeded at {config.PROJECT_MEMORY_FILE}")



        # ---- Phase 2: Build → Evaluate loop ----
        score_history: list[float] = []
        passed = False

        for round_num in range(1, max_rounds + 1):

            # ---- Contract negotiation (if enabled) ----
            if self.contract_proposer and self.contract_reviewer:
                log.info("=" * 60)
                log.info(f"ROUND {round_num}/{max_rounds}: CONTRACT NEGOTIATION")
                log.info("=" * 60)
                contract_start = time.time()
                self._negotiate_contract(round_num)
                log.info(f"Contract negotiation completed in {time.time() - contract_start:.0f}s")

            # ---- Build ----
            log.info("=" * 60)
            log.info(f"ROUND {round_num}/{max_rounds}: BUILD")
            log.info("=" * 60)
            build_start = time.time()

            # Sync time budget to harness start so builder knows total elapsed time
            from hooks import TimeBudgetHook
            for hook in self.builder.hooks:
                if isinstance(hook, TimeBudgetHook):
                    hook.sync_start_time(total_start)
                    # Let the profile resolve task-specific timeout
                    task_timeout = self.profile.resolve_task_timeout(user_prompt)
                    if task_timeout:
                        hook.budget_seconds = task_timeout
                        log.info(f"Time budget set to {task_timeout}s from task metadata")

            feedback_path = Path(config.WORKSPACE) / config.FEEDBACK_FILE
            prev_feedback = feedback_path.read_text(encoding="utf-8") if feedback_path.exists() else ""

            build_task = self.profile.format_build_task(
                user_prompt, round_num, prev_feedback, score_history,
            )

            self.builder.run(build_task)
            log.info(f"Build round {round_num} completed in {time.time() - build_start:.0f}s")

            # ---- Evaluate (if enabled) ----
            round_score: float | None = None
            if self.evaluator and not skip_evaluator:
                log.info("=" * 60)
                log.info(f"ROUND {round_num}/{max_rounds}: EVALUATE")
                log.info("=" * 60)
                eval_start = time.time()

                self.evaluator.run(
                    f"This is evaluation round {round_num}.\n"
                    f"Read spec.md to understand the task.\n"
                    f"Examine the work done and test it thoroughly.\n"
                    f"Score each criterion honestly. Write your evaluation to feedback.md."
                )

                log.info(f"Evaluation round {round_num} completed in {time.time() - eval_start:.0f}s")

                # Check score
                feedback_text = ""
                if feedback_path.exists():
                    feedback_text = feedback_path.read_text(encoding="utf-8")
                round_score = self.profile.extract_score(feedback_text)
                score_history.append(round_score)
                log.info(f"Round {round_num} average score: {round_score:.1f} / 10  (threshold: {threshold})")
                log.info(f"Score history: {score_history}")

                if round_score >= threshold:
                    log.info(f"PASSED at round {round_num}.")
                    passed = True
            else:
                log.info("No evaluator — single-pass execution.")
                passed = True  # treat single-pass completion as done for LTM learn

            # Update project memory after each build (+ eval) round
            try:
                refresh_project_memory(
                    user_prompt, round_num, llm_call_simple, score=round_score
                )
            except Exception as e:
                log.warning(f"Project memory refresh failed: {e}")

            if passed or not self.evaluator or skip_evaluator:
                break

        else:
            log.warning(f"Did not pass after {max_rounds} rounds.")

        # Long-term memory: task-complete summary (no RAG retrieval yet)
        try:
            learn_from_task(
                user_prompt,
                llm_call_simple,
                passed=passed,
                score_history=score_history,
            )
        except Exception as e:
            log.warning(f"Long-term memory learn_from_task failed: {e}")

        total_duration = time.time() - total_start
        log.info("=" * 60)
        log.info(f"HARNESS COMPLETE — total time: {total_duration / 60:.1f} minutes")
        log.info(f"Output in: {config.WORKSPACE}")
        log.info("=" * 60)

    def _negotiate_contract(self, round_num: int, max_iterations: int = 3) -> None:
        proposer = self.contract_proposer
        reviewer = self.contract_reviewer
        if proposer is None or reviewer is None:
            return

        proposer.run(
            f"This is round {round_num}.\n"
            f"Read spec.md. If feedback.md exists, read it too.\n"
            f"Propose a sprint contract for this round. Write it to contract.md."
        )

        for i in range(max_iterations):
            log.info(f"[contract] Review iteration {i + 1}/{max_iterations}")

            reviewer.run(
                f"Review the sprint contract in contract.md for round {round_num}.\n"
                f"Read spec.md for context. Read feedback.md if it exists.\n"
                f"If acceptable, write APPROVED at the top and save to contract.md.\n"
                f"If changes needed, write revision requests and save updated contract."
            )

            contract_path = Path(config.WORKSPACE) / "contract.md"
            if contract_path.exists():
                contract_text = contract_path.read_text(encoding="utf-8")
                if "APPROVED" in contract_text.upper()[:200]:
                    log.info("[contract] Contract approved.")
                    return

            if i < max_iterations - 1:
                log.info("[contract] Contract needs revision...")
                proposer.run(
                    f"The reviewer requested changes. Read contract.md and revise."
                )

        log.warning("[contract] Max iterations reached, proceeding with current contract.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    from logger import setup_logging
    setup_logging(verbose="--verbose" in sys.argv or "-v" in sys.argv)

    # Parse flags
    args = [a for a in sys.argv[1:] if a not in ("--verbose", "-v")]

    # --list-profiles
    if "--list-profiles" in args:
        print("Available profiles:\n")
        for p in list_profiles():
            print(f"  {p['name']:15s} {p['description']}")
        sys.exit(0)

    # --profile <name>
    profile_name = "app-builder"
    if "--profile" in args:
        idx = args.index("--profile")
        if idx + 1 < len(args):
            profile_name = args[idx + 1]
            args = args[:idx] + args[idx + 2:]
        else:
            print("Error: --profile requires a name")
            sys.exit(1)

    if not config.API_KEY:
        print("Error: Set OPENAI_API_KEY in .env or environment.")
        sys.exit(1)

    if len(args) < 1:
        print("Usage: python harness.py [--profile <name>] \"<task>\" [--verbose]")
        print()
        print("Profiles:")
        for p in list_profiles():
            print(f"  {p['name']:15s} {p['description']}")
        print()
        print("Examples:")
        print('  python harness.py "Build a DAW in the browser"')
        print('  python harness.py --profile terminal "Fix the broken symlinks in /tmp"')
        print('  python harness.py --profile swe-bench "Fix the TypeError in parse_config()"')
        print('  python harness.py --profile reasoning "What is the escape velocity of Mars?"')
        sys.exit(1)

    user_prompt = " ".join(args)

    try:
        profile = get_profile(profile_name)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    log.info(f"Prompt: {user_prompt}")
    log.info(f"Profile: {profile_name}")
    log.info(f"Model: {config.MODEL}")
    log.info(f"Base URL: {config.BASE_URL}")
    log.info(f"Workspace: {config.WORKSPACE}")

    # Preflight — verify API connection with retries for rate limits
    # Skip in benchmark mode (HARNESS_FLAT_WORKSPACE) to avoid wasting an API
    # call and hitting rate limits when many containers start simultaneously.
    if os.environ.get("HARNESS_FLAT_WORKSPACE"):
        log.info("Benchmark mode — skipping API preflight check.")
        preflight_ok = True
    else:
        log.info("Verifying API connection...")
        from agents import get_client
        import random
        preflight_ok = False
        for attempt in range(8):
            try:
                resp = get_client().chat.completions.create(
                    model=config.MODEL,
                    messages=[{"role": "user", "content": "Say OK"}],
                    max_tokens=5,
                )
                log.info(f"API OK — model responded: {resp.choices[0].message.content}")
                preflight_ok = True
                break
            except Exception as e:
                err_str = str(e)
                if "rate_limit" in err_str or "429" in err_str:
                    # Exponential backoff with jitter to avoid thundering herd
                    base_wait = min(2 ** (attempt + 1), 60)
                    jitter = random.uniform(0, base_wait * 0.5)
                    wait = base_wait + jitter
                    log.warning(f"API rate limited (attempt {attempt+1}/8), waiting {wait:.1f}s...")
                    time.sleep(wait)
                else:
                    log.error(f"API preflight failed: {e}")
                    break

    if not preflight_ok:
        print(f"\nCannot connect to API. Check your .env:\n"
              f"  OPENAI_API_KEY  — is it valid?\n"
              f"  OPENAI_BASE_URL — is {config.BASE_URL} correct?\n"
              f"  HARNESS_MODEL   — does {config.MODEL} exist on this provider?")
        sys.exit(1)

    harness = Harness(profile)
    try:
        harness.run(user_prompt)
    except KeyboardInterrupt:
        log.warning("Interrupted by user.")
        sys.exit(130)
    except Exception as e:
        log.error(f"Harness crashed with unhandled exception: {e}", exc_info=True)
        # Exit 1 signals failure to Harbor, but at least we log the cause
        sys.exit(1)


if __name__ == "__main__":
    main()
