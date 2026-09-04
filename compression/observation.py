"""
Observation compression — Tools may apply a hard cap to avoid OOM; this layer decides what the model sees.
Rules-only (no LLM).
"""
from __future__ import annotations

import re

import config
from tool_result import ShellObservation, ToolResult

_TOOL_MAX = {
    "run_shell": 10000,
    "read_file": 10000,
    "web_fetch": 6000,
    "web_search": 4000,
    "delegate_task": 6000,
    # Browser Testing MCP tools
    "browser_snapshot": 4000,
    "browser_console": 3000,
    "browser_evaluate": 3000,
}
_SKIP_COMPRESSION = {"read_skill_file"}

_ERROR_PATTERN = re.compile(
    r"(?i)(error|fail|assert|exception|traceback|warning|not found|denied|refused|fatal)",
)


def _limit_for(tool_name: str) -> int:
    global_max = int(config.OBSERVATION_MAX_CHARS)
    tool_max = _TOOL_MAX.get(tool_name)
    if tool_max is None:
        return global_max
    return min(int(tool_max), global_max)


def _source_hint(tool_args: dict | None) -> str:
    if not isinstance(tool_args, dict):
        return ""
    path = (
        tool_args.get("path")
        or tool_args.get("command")
        or tool_args.get("url")
        or tool_args.get("query")
    )
    if path:
        return f" source={str(path)[:120]}"
    return ""


def _as_text(result: object) -> str:
    if isinstance(result, ShellObservation):
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        if stderr:
            return (stdout + "\n\n--- STDERR ---\n" + stderr).strip()
        return stdout
    if result is None:
        return ""
    if not isinstance(result, str):
        return str(result)
    return result


def _status_prefix(result: ToolResult) -> str:
    """Header for non-shell ToolResult metadata. Shell headers live in _compress_shell."""
    if isinstance(result.payload, ShellObservation):
        return ""
    if result.exit_code is not None and result.kind != "ok":
        return f"exit={result.exit_code}"
    return ""


def _shell_status_prefix(obs: ShellObservation) -> str:
    parts: list[str] = []
    if obs.timed_out:
        parts.append("timed_out=true")
    if obs.exit_code != 0:
        parts.append(f"exit={obs.exit_code}")
    return " ".join(parts)


def compress_observation(
    tool_name: str,
    tool_args: dict | None,
    result: object,
) -> str:
    """
    Compress a single tool result for the conversation window.
    """
    auto_fix = None
    header = ""
    if isinstance(result, ToolResult):
        auto_fix = result.auto_fix
        header = _status_prefix(result)
        result = result.payload

    if tool_name in _SKIP_COMPRESSION:
        out = _as_text(result)
    else:
        limit = _limit_for(tool_name)
        if tool_name == "run_shell":
            out = _compress_shell(result, limit, tool_args)
        else:
            text = _as_text(result)
            out = (
                text
                if len(text) <= limit
                else _compress_prefix_lines(text, limit, tool_name, tool_args)
            )

    if header:
        out = header + "\n" + out
    if auto_fix:
        return f"{auto_fix}\n\n{out}"
    return out


def _compress_shell(
    result: object, limit: int, tool_args: dict | None
) -> str:
    if isinstance(result, ShellObservation):
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        if not stdout and not stderr:
            body = "(no output)"
        else:
            body = _smart_truncate_output(stdout, stderr, limit)
        prefix = _shell_status_prefix(result)
        if prefix:
            return prefix + "\n" + body
        return body

    text = _as_text(result)
    if text.startswith("[error]"):
        if len(text) <= limit:
            return text
        return _compress_prefix_lines(text, limit, "run_shell", tool_args)
    if len(text) <= limit:
        return text
    return _smart_truncate_output(text.strip(), "", limit)


def _smart_truncate_output(stdout: str, stderr: str, limit: int) -> str:
    """Preserve stderr and error-like lines from the middle of stdout.

    Strategy:
    - Keep tail for stderr.
    - Use head + important-middle + tail for stdout.
    """
    combined = (stdout + "\n" + stderr).strip() if stderr else stdout
    if len(combined) <= limit:
        return combined

    stderr_budget = min(len(stderr), int(limit * 0.4))
    stdout_budget = limit - stderr_budget

    if len(stderr) > stderr_budget:
        stderr = "...[stderr truncated]\n" + stderr[-(stderr_budget - 30) :]

    if len(stdout) <= stdout_budget:
        truncated_stdout = stdout
    else:
        head_size = int(stdout_budget * 0.40)
        tail_size = int(stdout_budget * 0.40)
        middle_budget = stdout_budget - head_size - tail_size - 200

        head = stdout[:head_size]
        tail = stdout[-tail_size:]
        middle = stdout[head_size:-tail_size] if tail_size else stdout[head_size:]
        important_lines = []
        if middle and middle_budget > 0:
            for line in middle.splitlines():
                if _ERROR_PATTERN.search(line):
                    important_lines.append(line)

        important_section = "\n".join(important_lines)
        if len(important_section) > middle_budget:
            important_section = important_section[:middle_budget]

        if important_section:
            middle_part = (
                f"\n\n[...{len(middle)} chars omitted — key lines extracted:]\n"
                + important_section
                + "\n[...end extracted lines]\n\n"
            )
        else:
            middle_part = (
                f"\n\n[TRUNCATED — {len(middle)} chars omitted from middle]\n\n"
            )
        truncated_stdout = head + middle_part + tail

    if stderr:
        return truncated_stdout + "\n\n--- STDERR ---\n" + stderr
    return truncated_stdout


def _compress_prefix_lines(
    text: str,
    limit: int,
    tool_name: str,
    tool_args: dict | None,
) -> str:
    """Keep a numbered prefix; drop the rest."""
    hint = _source_hint(tool_args)
    lines = text.splitlines()
    if not lines:
        lines = [text]
    total_lines = len(lines)

    footer = (
        f"\n[OBSERVATION COMPRESSED] showing first {{kept}} of {total_lines} lines, "
        f"omitted ~{{omitted}} chars "
        f"(tool={tool_name}{hint}; limit={limit}). "
        "Re-read with a narrower path/command if you need the rest.\n"
    )
    footer_est = len(footer.format(kept=total_lines, omitted=len(text)))
    budget = max(limit - footer_est, 64)

    out_parts: list[str] = []
    used = 0
    kept = 0
    for i, line in enumerate(lines, 1):
        numbered = f"{i}| {line}\n"
        if used + len(numbered) > budget:
            if kept == 0:
                prefix = f"{i}| "
                room = max(budget - len(prefix) - 1, 16)
                out_parts.append(prefix + line[:room] + "\n")
                kept = 1
            break
        out_parts.append(numbered)
        used += len(numbered)
        kept += 1

    kept_raw = "\n".join(lines[:kept])
    omitted = max(len(text) - len(kept_raw), 0)
    return "".join(out_parts) + footer.format(kept=kept, omitted=omitted)
