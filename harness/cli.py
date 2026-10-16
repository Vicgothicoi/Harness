"""Command-line entry for the harness."""
from __future__ import annotations

import logging
import os
import sys
import time

from harness import config
from harness.orchestration.loop import Harness
from harness.profiles import get_profile, list_profiles

log = logging.getLogger("harness")

def main():
    from harness.log import setup_logging

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
            args = args[:idx] + args[idx + 2 :]
        else:
            print("Error: --profile requires a name")
            sys.exit(1)

    if not config.API_KEY:
        print("Error: Set OPENAI_API_KEY in .env or environment.")
        sys.exit(1)

    if len(args) < 1:
        print('Usage: python harness.py [--profile <name>] "<task>" [--verbose]')
        print()
        print("Profiles:")
        for p in list_profiles():
            print(f"  {p['name']:15s} {p['description']}")
        print()
        print("Examples:")
        print('  python harness.py "Build a DAW in the browser"')
        print(
            '  python harness.py --profile terminal "Fix the broken symlinks in /tmp"'
        )
        print(
            '  python harness.py --profile swe-bench "Fix the TypeError in parse_config()"'
        )
        print(
            '  python harness.py --profile reasoning "What is the escape velocity of Mars?"'
        )
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
    log.info(f"Workspace root: {config.WORKSPACE_ROOT}")

    # Preflight — verify API connection with retries for rate limits
    # Skip in benchmark mode (HARNESS_FLAT_WORKSPACE) to avoid wasting an API
    # call and hitting rate limits when many containers start simultaneously.
    if os.environ.get("HARNESS_FLAT_WORKSPACE"):
        log.info("Benchmark mode — skipping API preflight check.")
        preflight_ok = True
    else:
        log.info("Verifying API connection...")
        from harness.orchestration.llm import get_client
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
                    log.warning(
                        f"API rate limited (attempt {attempt+1}/8), waiting {wait:.1f}s..."
                    )
                    time.sleep(wait)
                else:
                    log.error(f"API preflight failed: {e}")
                    break

    if not preflight_ok:
        print(
            f"\nCannot connect to API. Check your .env:\n"
            f"  OPENAI_API_KEY  — is it valid?\n"
            f"  OPENAI_BASE_URL — is {config.BASE_URL} correct?\n"
            f"  HARNESS_MODEL   — does {config.MODEL} exist on this provider?"
        )
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
