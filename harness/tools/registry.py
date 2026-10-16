"""OpenAI tool schemas and dispatch."""
from __future__ import annotations

import os

from harness.tools.delegate import delegate_task
from harness.tools.files import (
    list_files,
    read_file,
    read_skill_file,
    remember_preference,
    write_file,
)
from harness.tools.result import ToolResult, result_error, wrap_legacy
from harness.tools.shell import run_shell
from harness.tools.validate import _validate_and_fix
from harness.tools.web import web_fetch, web_search

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
) -> ToolResult:
    """Execute a tool by name with pre-validation and auto-correction."""
    fn = TOOL_DISPATCH.get(name)
    if fn is None:
        return result_error(f"[error] Unknown tool: {name}")

    # Pre-validate and auto-correct arguments
    arguments, fix_warning = _validate_and_fix(name, arguments)

    # If validation returned a blocking error (no fix possible), return it
    if fix_warning and fix_warning.startswith("[auto-fix] Empty"):
        return result_error(fix_warning)
    if fix_warning and "interactive command" in fix_warning:
        return result_error(fix_warning)

    try:
        if name == "run_shell":
            result = fn(**arguments, runtime_state=runtime_state, agent_name=agent_name)
        else:
            result = fn(**arguments)
    except Exception as e:
        result = result_error(f"[error] {type(e).__name__}: {e}")

    if not isinstance(result, ToolResult):
        result = wrap_legacy(result)

    return result.with_auto_fix(fix_warning)
