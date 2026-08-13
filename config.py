"""
Harness configuration.
Uses OpenAI-compatible API so it works with any provider.

Setup:
  cp .env.template .env   # then fill in your real values
"""

import os
from pathlib import Path


def _load_dotenv():
    """Load .env file if it exists. No third-party dependency needed."""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        # Shell env vars take priority over .env
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

# --- API ---
API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL = os.environ.get("HARNESS_MODEL", "gpt-4o")

# --- Token budgets ---
# Trace compression (rules-based) kicks in above COMPRESS_THRESHOLD.
# Full reset + handoff.md kicks in above RESET_THRESHOLD (or anxiety).
COMPRESS_THRESHOLD = int(os.environ.get("COMPRESS_THRESHOLD", "80000"))
RESET_THRESHOLD = int(os.environ.get("RESET_THRESHOLD", "150000"))

# --- Harness loop ---
MAX_HARNESS_ROUNDS = int(os.environ.get("MAX_HARNESS_ROUNDS", "5"))
PASS_THRESHOLD = float(os.environ.get("PASS_THRESHOLD", "7.0"))

# --- Agent limits ---
MAX_AGENT_ITERATIONS = int(os.environ.get("MAX_AGENT_ITERATIONS", "60"))
MAX_TOOL_ERRORS = 5  # consecutive tool errors before abort

# --- Paths ---
WORKSPACE = os.path.abspath(os.environ.get("HARNESS_WORKSPACE", "./workspace"))
SPEC_FILE = "spec.md"
FEEDBACK_FILE = "feedback.md"
CONTRACT_FILE = "contract.md"
PROGRESS_FILE = "progress.md"
HANDOFF_FILE = "handoff.md"
PROJECT_MEMORY_FILE = "project_memory.json"
LONG_TERM_MEMORY_FILE = "long_term_memory.json"

# --- State / project / long-term memory ---
STATE_CONTEXT_MAX_CHARS = int(os.environ.get("STATE_CONTEXT_MAX_CHARS", "1500"))
PROJECT_CONTEXT_MAX_CHARS = int(os.environ.get("PROJECT_CONTEXT_MAX_CHARS", "2000"))
LONG_TERM_PREFS_MAX_CHARS = int(os.environ.get("LONG_TERM_PREFS_MAX_CHARS", "800"))
# Global long-term memory directory (outside workspace). Empty = ~/.harness/memory
LONG_TERM_MEMORY_DIR = os.environ.get("LONG_TERM_MEMORY_DIR", "")

# --- Observation / trace compression ---
OBSERVATION_MAX_CHARS = int(os.environ.get("OBSERVATION_MAX_CHARS", "8000"))
TRACE_KEEP_RECENT = int(os.environ.get("TRACE_KEEP_RECENT", "8"))
TRACE_SUMMARY_MAX_CHARS = int(os.environ.get("TRACE_SUMMARY_MAX_CHARS", "4000"))
