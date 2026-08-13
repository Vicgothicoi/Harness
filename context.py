"""
Context metering helpers — token counting and context-anxiety detection.

Compression strategies live under `compression/`:
  1. observation.py — per tool result
  2. trace.py — ActionRecord → [TRACE SUMMARY]
  3. state.py — TaskBoard / progress.md
  4. full.py — handoff.md + session reset
"""
from __future__ import annotations

import re
import logging

import config

log = logging.getLogger("harness")

# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

_encoder = None
_use_tiktoken = False

try:
    import tiktoken
    _use_tiktoken = True
except ImportError:
    pass


def _get_encoder():
    global _encoder
    if not _use_tiktoken:
        return None
    if _encoder is None:
        try:
            _encoder = tiktoken.encoding_for_model(config.MODEL)
        except Exception:
            _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def count_tokens(messages: list[dict]) -> int:
    """Rough token count for a message list.
    Uses tiktoken if available, otherwise estimates ~4 chars per token."""
    enc = _get_encoder()
    total = 0
    for msg in messages:
        content = msg.get("content") or ""
        if isinstance(content, list):
            content = " ".join(
                block.get("text", "") for block in content if isinstance(block, dict)
            )
        text = str(content)
        if enc:
            total += len(enc.encode(text)) + 4
        else:
            total += len(text) // 4 + 4
        for tc in msg.get("tool_calls", []):
            args = str(tc.get("function", {}).get("arguments", ""))
            if enc:
                total += len(enc.encode(args))
            else:
                total += len(args) // 4
    return total


# ---------------------------------------------------------------------------
# Context anxiety detection
# ---------------------------------------------------------------------------

_ANXIETY_PATTERNS = [
    r"(?i)let me wrap up",
    r"(?i)i('ll| will) finalize",
    r"(?i)that should be (enough|sufficient)",
    r"(?i)i('ll| will) stop here",
    r"(?i)due to (context |token )?limit",
    r"(?i)running (low on|out of) (context|space|tokens)",
    r"(?i)to (save|conserve) (context|space|tokens)",
    r"(?i)i('ve| have) covered the (main|key|essential)",
    r"(?i)in the interest of (time|space|brevity)",
]


def detect_anxiety(messages: list[dict]) -> bool:
    """
    Check recent assistant messages for signs of context anxiety —
    the model trying to wrap up work prematurely because it thinks
    it's running out of context space.
    """
    recent_texts = []
    for msg in reversed(messages[-10:]):
        if msg.get("role") == "assistant" and msg.get("content"):
            recent_texts.append(msg["content"])
        if len(recent_texts) >= 3:
            break

    combined = " ".join(recent_texts)
    matches = sum(1 for p in _ANXIETY_PATTERNS if re.search(p, combined))
    if matches >= 2:
        log.warning(f"Context anxiety detected ({matches} signals found)")
        return True
    return False
