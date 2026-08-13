"""Helpers for injecting replaceable marked memory blocks into message lists."""
from __future__ import annotations


def upsert_marked_block(
    messages: list[dict],
    marker: str,
    content: str,
    *,
    after_markers: tuple[str, ...] = (),
) -> list[dict]:
    """
    Ensure exactly one user message starting with `marker` exists.

    Inserts after the last message whose content starts with any of
    `after_markers` (if present), else after the system prompt, else at front.
    """
    if not content or not content.strip():
        # Remove stale block if content is empty
        return [
            m
            for m in messages
            if not (
                m.get("role") == "user"
                and isinstance(m.get("content"), str)
                and m["content"].startswith(marker)
            )
        ]

    block = content if content.startswith(marker) else f"{marker}\n{content}"
    msg = {"role": "user", "content": block}

    cleaned = [
        m
        for m in messages
        if not (
            m.get("role") == "user"
            and isinstance(m.get("content"), str)
            and m["content"].startswith(marker)
        )
    ]

    insert_at = 0
    if cleaned and cleaned[0].get("role") == "system":
        insert_at = 1

    if after_markers:
        last_after = insert_at - 1
        for i, m in enumerate(cleaned):
            c = m.get("content")
            if m.get("role") == "user" and isinstance(c, str):
                if any(c.startswith(am) for am in after_markers):
                    last_after = i
        if last_after >= insert_at - 1:
            insert_at = last_after + 1

    return cleaned[:insert_at] + [msg] + cleaned[insert_at:]
