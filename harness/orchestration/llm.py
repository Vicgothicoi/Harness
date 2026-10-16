"""OpenAI-compatible client used by the agent loop and summarizers."""
from __future__ import annotations

import logging
import time

from openai import OpenAI

from harness import config

log = logging.getLogger("harness")

_client: OpenAI | None = None

def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=config.API_KEY,
            base_url=config.BASE_URL,
            timeout=300.0,  # 5 min per request
            max_retries=2,
        )
    return _client

def llm_call_simple(messages: list[dict]) -> str:
    """Simple LLM call without tools — used for summarization"""
    import random

    for attempt in range(4):
        try:
            resp = get_client().chat.completions.create(
                model=config.MODEL,
                messages=messages,
                max_tokens=200000,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            err_str = str(e)
            if ("rate_limit" in err_str.lower() or "429" in err_str) and attempt < 3:
                wait = min(2 ** (attempt + 1), 30) + random.uniform(0, 3)
                log.warning(
                    f"llm_call_simple rate limited, waiting {wait:.1f}s (attempt {attempt+1}/4)"
                )
                time.sleep(wait)
                continue
            log.error(f"llm_call_simple failed: {e}")
            # Return a minimal summary rather than crashing
            return "[context summarization failed — continuing with truncated context]"
    return "[context summarization failed after retries]"
