"""
Tool definitions and execution for agents.
Each tool is an OpenAI function-calling schema + a Python implementation.
Agents operate inside config.WORKSPACE to keep generated code isolated.
"""

from __future__ import annotations

import os
from pathlib import Path

import config
from compression.observation import ShellObservation

_RESULT_HARD_CAP = int(getattr(config, "TOOL_RESULT_HARD_CAP", 1_000_000))
_LIST_FILES_MAX = 200
_SKILL_FILE_HARD_CAP = 60_000


def _hard_cap(text: str, cap: int = _RESULT_HARD_CAP) -> str:
    if len(text) <= cap:
        return text
    return text[:cap]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve(path: str) -> Path:
    """Resolve a relative path inside the workspace. Prevent escaping."""
    p = Path(config.WORKSPACE, path).resolve()
    ws = Path(config.WORKSPACE).resolve()
    if not str(p).startswith(str(ws)):
        raise ValueError(f"Path escapes workspace: {path}")
    return p


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def read_file(path: str) -> str:
    p = _resolve(path)
    if not p.exists():
        return f"[error] File not found: {path}"
    try:
        with p.open("r", encoding="utf-8", errors="replace") as f:
            return f.read(_RESULT_HARD_CAP)
    except OSError as e:
        return f"[error] {e}"


def read_skill_file(path: str) -> str:
    """Read a file from the skills directory (outside workspace). Path must be relative to project root."""
    project_root = Path(__file__).parent
    p = (project_root / path).resolve()
    # Must stay within the skills directory
    skills_dir = (project_root / "skills").resolve()
    if not str(p).startswith(str(skills_dir)):
        return f"[error] Path must be inside skills/ directory: {path}"
    if not p.exists():
        return f"[error] Skill file not found: {path}"
    return _hard_cap(
        p.read_text(encoding="utf-8", errors="replace"),
        _SKILL_FILE_HARD_CAP,
    )


def write_file(path: str, content: str) -> str:
    if not path or not path.strip():
        return "[error] Empty file path"
    p = _resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} chars to {path}"


def remember_preference(key: str, value: str) -> str:
    """Explicitly store a user preference in global long-term memory."""
    from memory.long_term_memory import LongTermMemory

    key = (key or "").strip()
    value = (value or "").strip()
    if not key:
        return "[error] remember_preference requires a non-empty key"
    if not value:
        return "[error] remember_preference requires a non-empty value"
    ltm = LongTermMemory.load()
    ltm.set_preference(key, value)
    path = ltm.save()
    return f"Saved preference {key}={value!r} to {path}"


def list_files(directory: str = ".") -> str:
    p = _resolve(directory)
    if not p.is_dir():
        return f"[error] Not a directory: {directory}"
    entries = []
    ws = Path(config.WORKSPACE).resolve()
    for item in sorted(p.rglob("*")):
        if not item.is_file():
            continue
        rel = item.relative_to(ws)
        if any(part.startswith(".") for part in rel.parts):
            continue
        entries.append(str(rel))
    if not entries:
        return "(empty)"
    if len(entries) > _LIST_FILES_MAX:
        return (
            "\n".join(entries[:_LIST_FILES_MAX])
            + f"\n...[list stopped at {_LIST_FILES_MAX} files]"
        )
    return "\n".join(entries)


def run_shell(
    command: str,
    timeout: int = 300,
    runtime_state=None,
    agent_name: str | None = None,
) -> str | ShellObservation:
    """Run a shell command inside the agent's persistent shell session."""
    if runtime_state is None or runtime_state.shell_session is None:
        return "[error] No active shell session for run_shell"
    try:
        shell_result = runtime_state.shell_session.run(command, timeout=timeout)
        if shell_result.timed_out:
            return (
                f"[error] Command timed out after {timeout}s. "
                f"If this command legitimately needs more time (e.g. compilation, training), "
                f"retry with a larger timeout parameter."
            )
        return ShellObservation(
            stdout=_hard_cap(shell_result.stdout or ""),
            stderr=_hard_cap(shell_result.stderr or ""),
        )
    except Exception as e:
        return f"[error] {e}"


# ---------------------------------------------------------------------------
# Sub-agent delegation (context isolation)
# ---------------------------------------------------------------------------


def delegate_task(task: str, role: str = "assistant") -> str:
    """
    Spawn a sub-agent in a completely isolated context to handle a subtask.

    The sub-agent gets a clean context window — it does NOT inherit the parent's
    conversation history. It has access to the same workspace and tools.
    Only the structured result comes back to the parent.

    Use this for:
    - Exploring/reading many files without polluting your context
    - Running a series of bash commands and summarizing results
    - Any "dirty work" that would bloat your context window

    The sub-agent's internal reasoning is invisible to the caller.
    """
    # Lazy import to avoid circular dependency
    from agents import Agent

    sub = Agent(
        name=f"sub_{role}",
        system_prompt=(
            f"You are a sub-agent with the role: {role}. "
            f"Complete the assigned task and provide a concise, structured summary of your findings. "
            f"You have access to the workspace files and bash. "
            f"Focus only on the task — do not do extra work.\n"
            f"When done, respond with a clear summary of:\n"
            f"1. What you found or did\n"
            f"2. Key results or artifacts created\n"
            f"3. Any issues encountered"
        ),
        use_tools=True,
        enable_memory=False,
    )

    result = sub.run(task)

    if not result:
        return "[sub-agent returned no output]"
    return _hard_cap(result)


def _run_shell_description() -> str:
    if os.name == "nt":
        return (
            "Run a command in the persistent Windows cmd.exe session in the workspace "
            "(cwd and environment are kept across calls). Use cmd syntax: dir, type, "
            "findstr, cd, set. Chain commands with & (sequential in cmd). "
            "Background: start /B. Do not use bash-only commands (ls, cat, pwd, &&, export). "
            "For long-running work, increase timeout. Stderr is preserved separately."
        )
    return (
        "Run a command in the persistent bash session in the workspace "
        "(cwd and environment are kept across calls). Use for installing deps, "
        "running builds, starting servers, running tests, etc. For long-running "
        "commands (compilation, training), increase the timeout parameter. "
        "For background services (VMs, servers), use '... &' and a separate command "
        "to check readiness. Stderr is preserved separately for easier debugging."
    )


# ---------------------------------------------------------------------------
# OpenAI function-calling schemas
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file in the workspace.",
            "parameters": {
                "type": "object",
                "required": ["path"],
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path inside workspace",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_skill_file",
            "description": "Read a skill file from the skills/ directory. Use this to load a skill's SKILL.md or any sub-files referenced within it. Path should be relative to project root (e.g. 'skills/frontend-design/SKILL.md').",
            "parameters": {
                "type": "object",
                "required": ["path"],
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to skill file (e.g. 'skills/frontend-design/SKILL.md')",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file in the workspace.",
            "parameters": {
                "type": "object",
                "required": ["path", "content"],
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path inside workspace",
                    },
                    "content": {
                        "type": "string",
                        "description": "File content to write",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember_preference",
            "description": (
                "Save a durable user preference into global long-term memory "
                "(survives across projects). Use when the user explicitly states "
                "a lasting preference (language, tooling, style)."
            ),
            "parameters": {
                "type": "object",
                "required": ["key", "value"],
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Preference name, e.g. ui_language",
                    },
                    "value": {
                        "type": "string",
                        "description": "Preference value, e.g. zh-CN",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List all files in a directory recursively.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Relative directory path (default: root)",
                        "default": ".",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": _run_shell_description(),
            "parameters": {
                "type": "object",
                "required": ["command"],
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to run",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default 300). Increase for long builds/training.",
                        "default": 300,
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delegate_task",
            "description": (
                "Spawn a sub-agent in a completely isolated context to handle a subtask. "
                "The sub-agent gets a clean context window and does NOT see your conversation history. "
                "Only its structured result comes back. Use this for: "
                "(1) exploring/reading many files without bloating your context, "
                "(2) running a series of bash commands and getting a summary, "
                "(3) any 'dirty work' that would waste your context budget. "
                "The sub-agent has access to the same workspace and tools."
            ),
            "parameters": {
                "type": "object",
                "required": ["task"],
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Detailed description of the subtask to delegate",
                    },
                    "role": {
                        "type": "string",
                        "description": "Role for the sub-agent (e.g. 'codebase_explorer', 'test_runner', 'dependency_installer')",
                        "default": "assistant",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information. Use when you need documentation, examples, or domain knowledge not available locally. Returns titles, URLs, and snippets.",
            "parameters": {
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {
                        "type": "integer",
                        "description": "Max results to return (default 5)",
                        "default": 5,
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch and read the text content of a web page. Use after web_search to read a specific page in detail.",
            "parameters": {
                "type": "object",
                "required": ["url"],
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"},
                },
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Tool-call pre-validation & auto-correction
# ---------------------------------------------------------------------------


def _validate_and_fix(name: str, arguments: dict) -> tuple[dict, str | None]:
    """
    Pre-validate tool arguments and auto-correct common mistakes.
    Returns (fixed_arguments, warning_message_or_None).

    This is a lightweight heuristic layer — no LLM calls.
    Catches the most common tool-call errors from weaker models:
      - Empty/missing required arguments
      - Absolute paths that should be relative
      - Obvious typos in common patterns
    """
    warning = None

    if name == "write_file":
        path = arguments.get("path", "")
        content = arguments.get("content")

        # Empty path
        if not path or not path.strip():
            return arguments, "[auto-fix] Empty file path. You must specify a path."

        # Absolute path → make relative to workspace
        if path.startswith("/"):
            import re

            # Strip common workspace prefixes
            for prefix in ["/app/", "/home/user/", "/workspace/"]:
                if path.startswith(prefix):
                    arguments["path"] = path[len(prefix) :]
                    warning = f"[auto-fix] Converted absolute path '{path}' to relative '{arguments['path']}'"
                    break

        # Missing content
        if content is None:
            arguments["content"] = ""
            warning = "[auto-fix] Missing 'content' argument — writing empty file."

    elif name == "read_file":
        path = arguments.get("path", "")

        # Absolute path → relative
        if path.startswith("/"):
            for prefix in ["/app/", "/home/user/", "/workspace/"]:
                if path.startswith(prefix):
                    arguments["path"] = path[len(prefix) :]
                    warning = f"[auto-fix] Converted absolute path '{path}' to relative '{arguments['path']}'"
                    break

    elif name == "run_shell":
        command = arguments.get("command", "")

        # Empty command
        if not command or not command.strip():
            return (
                arguments,
                "[auto-fix] Empty command. You must specify a command to run.",
            )

        # Detect interactive commands that will hang
        interactive_cmds = ["vim", "nano", "vi", "less", "more", "top", "htop"]
        first_word = command.strip().split()[0] if command.strip() else ""
        if first_word in interactive_cmds:
            viewer = "type" if os.name == "nt" else "cat/head/tail"
            return arguments, (
                f"[auto-fix] '{first_word}' is an interactive command that will hang. "
                f"Use non-interactive alternatives: "
                f"for editing use write_file, for viewing use {viewer}."
            )

    elif name == "list_files":
        directory = arguments.get("directory", ".")
        if directory.startswith("/"):
            for prefix in ["/app/", "/home/user/", "/workspace/"]:
                if directory.startswith(prefix):
                    arguments["directory"] = directory[len(prefix) :] or "."
                    warning = f"[auto-fix] Converted absolute path '{directory}' to relative '{arguments['directory']}'"
                    break

    return arguments, warning


# ---------------------------------------------------------------------------
# Web search (lightweight, no external deps)
# ---------------------------------------------------------------------------


def web_search(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo and return text results.
    Uses DDG's lite HTML endpoint — no API key needed, works in any container.
    """
    import urllib.request
    import urllib.parse
    import re
    import html as html_mod

    try:
        encoded = urllib.parse.urlencode({"q": query})
        url = f"https://lite.duckduckgo.com/lite/?{encoded}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=15)
        raw = resp.read().decode("utf-8", errors="replace")

        # Extract result links (DDG lite uses rel="nofollow" for result links)
        links = re.findall(
            r'<a[^>]*rel="nofollow"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', raw, re.DOTALL
        )

        # Extract snippets (text in <td> cells that aren't links/navigation)
        cells = re.findall(r"<td[^>]*>(.*?)</td>", raw, re.DOTALL)
        snippets = []
        for cell in cells:
            text = re.sub(r"<[^>]+>", "", cell).strip()
            if len(text) > 50 and not text.startswith("http"):
                snippets.append(text)

        results = []
        for i, (href, title) in enumerate(links):
            if i >= max_results:
                break
            title = html_mod.unescape(re.sub(r"<[^>]+>", "", title).strip())
            # Decode DDG redirect URL
            real_url = href
            m = re.search(r"uddg=([^&]+)", href)
            if m:
                real_url = urllib.parse.unquote(m.group(1))
            snippet = snippets[i] if i < len(snippets) else ""
            results.append(f"{i+1}. {title}\n   {real_url}\n   {snippet[:200]}\n")

        if results:
            return f"Search results for: {query}\n\n" + "\n".join(results)

        return f"No results found for: {query}"

    except Exception as e:
        return f"[error] Web search failed: {e}"


def web_fetch(url: str) -> str:
    """Fetch the content of a web page and return as text."""
    import urllib.request
    import re

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            },
        )
        resp = urllib.request.urlopen(req, timeout=15)
        html = resp.read(_RESULT_HARD_CAP).decode("utf-8", errors="replace")

        # Strip HTML tags, keep text (conversation truncation is observation's job)
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        return _hard_cap(text) or "(empty page)"

    except Exception as e:
        return f"[error] Web fetch failed: {e}"


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

TOOL_DISPATCH = {
    "read_file": read_file,
    "read_skill_file": read_skill_file,
    "write_file": write_file,
    "remember_preference": remember_preference,
    "list_files": list_files,
    "run_shell": run_shell,
    "delegate_task": delegate_task,
    "web_search": web_search,
    "web_fetch": web_fetch,
}


def execute_tool(
    name: str, arguments: dict, runtime_state=None, agent_name: str | None = None
) -> str | ShellObservation:
    """Execute a tool by name with pre-validation and auto-correction."""
    fn = TOOL_DISPATCH.get(name)
    if fn is None:
        return f"[error] Unknown tool: {name}"

    # Pre-validate and auto-correct arguments
    arguments, fix_warning = _validate_and_fix(name, arguments)

    # If validation returned a blocking error (no fix possible), return it
    if fix_warning and fix_warning.startswith("[auto-fix] Empty"):
        return fix_warning
    if fix_warning and "interactive command" in fix_warning:
        return fix_warning

    try:
        if name == "run_shell":
            result = fn(**arguments, runtime_state=runtime_state, agent_name=agent_name)
        else:
            result = fn(**arguments)
    except Exception as e:
        result = f"[error] {type(e).__name__}: {e}"

    # Prepend the auto-fix warning so the model knows what was corrected
    if fix_warning:
        if isinstance(result, ShellObservation):
            result = ShellObservation(
                stdout=f"{fix_warning}\n\n{result.stdout}",
                stderr=result.stderr,
            )
        else:
            result = f"{fix_warning}\n\n{result}"

    return result
