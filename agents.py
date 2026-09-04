"""
Agent implementation — the core while loop with tool use.
Uses OpenAI-compatible chat completions API with function calling.
"""

from __future__ import annotations

import json
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from openai import OpenAI

import config
import tools
import context
from hooks import RecoveryState
from memory.working_memory import (
    build_working_memory,
    build_working_memory_from_runtime,
)
from memory.state_memory import (
    TaskBoard,
    board_says_stop,
    persist_task_board,
    seed_task_board,
)
from compression.observation import compress_observation
from compression.trace import TraceBuffer, compress_trace, record_from_tool
from compression.state import compress_state
from compression.full import full_compress_reset
from shell_session import PersistentShellSession

log = logging.getLogger("harness")

ACTION_TOOLS = {"run_shell", "write_file", "delegate_task"}


def _reinject_memory_blocks(
    messages: list[dict], runtime_state: "AgentRuntimeState"
) -> list[dict]:
    """Re-project memory layers"""
    return build_working_memory_from_runtime(messages, runtime_state)


def _truncate(s: object, n: int) -> str:
    text = s if isinstance(s, str) else str(s)
    return text[:n] + "..." if len(text) > n else text


# ---------------------------------------------------------------------------
# Trace writer — records every agent event to a JSONL file
# ---------------------------------------------------------------------------


class TraceWriter:
    """Appends structured events to a JSONL trace file outside the agent workspace.

    Each line is a JSON object with: timestamp, agent, event_type, and data.
    Trace file: {LOG}/_trace_{agent_name}.jsonl
    """

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self._start_time = time.time()
        trace_dir = Path(config.LOG)
        trace_dir.mkdir(parents=True, exist_ok=True)
        self._path = trace_dir / f"_trace_{agent_name}.jsonl"

    def _write(self, event_type: str, data: dict):
        try:
            entry = {
                "t": round(time.time() - self._start_time, 2),
                "agent": self.agent_name,
                "event": event_type,
                **data,
            }
            line = json.dumps(entry, ensure_ascii=False)[:10000]
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            # Also print to stderr so Harbor logs capture it
            import sys

            print(f"[TRACE] {line}", file=sys.stderr)
        except Exception:
            pass  # never let tracing break the agent

    def iteration(self, n: int, tokens: int):
        self._write("iteration", {"n": n, "tokens": tokens})

    def llm_response(
        self, content: str | None, tool_calls: list | None, finish_reason: str | None
    ):
        self._write(
            "llm_response",
            {
                "content": (content or "")[:500],
                "tool_calls": [tc["function"]["name"] for tc in (tool_calls or [])],
                "finish_reason": finish_reason,
            },
        )

    def tool_call(self, name: str, args: dict, result: str):
        self._write(
            "tool_call",
            {
                "tool": name,
                "args": _truncate(json.dumps(args, ensure_ascii=False), 300),
                "result": _truncate(result, 500),
            },
        )

    def hook_inject(self, source: str, hook: str, message: str):
        self._write(
            "hook",
            {
                "source": source,
                "hook": hook,
                "message": message[:300],
            },
        )

    def context_event(self, event_type: str, reason: str = ""):
        self._write("context", {"type": event_type, "reason": reason})

    def error(self, error_type: str, message: str):
        self._write("error", {"type": error_type, "message": message[:500]})

    def finish(self, reason: str, iterations: int):
        self._write("finish", {"reason": reason, "iterations": iterations})

    def dump_messages(self, iteration: int, messages: list[dict], tokens: int):
        """Append one LLM-request context snapshot (JSONL, not truncated)."""
        try:
            path = Path(config.LOG) / f"_messages_{self.agent_name}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "iteration": iteration,
                "tokens": tokens,
                "messages": messages,
            }
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# LLM client (singleton)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Agent runtime state
# ---------------------------------------------------------------------------


@dataclass
class AgentRuntimeState:
    shell_session: PersistentShellSession | None = None
    task_board: TaskBoard = field(default_factory=TaskBoard)
    recovery: RecoveryState = field(default_factory=RecoveryState)
    trace_buffer: TraceBuffer = field(default_factory=TraceBuffer)
    action_tool_count: int = 0
    stop_signaled: bool = False


# ---------------------------------------------------------------------------
# Core agent loop
# ---------------------------------------------------------------------------


class Agent:
    """
    A single agent with a system prompt and tool access.

    This is the 'managed agent loop' from the architecture:
    - while loop with llm.call(prompt)
    - tool execution
    - context lifecycle (observation / trace / state / reset)

    Skills are handled via progressive disclosure:
    - Level 1: skill catalog (name + description) is baked into system_prompt
    - Level 2: agent decides to read_skill_file("skills/.../SKILL.md") on its own
    - Level 3: SKILL.md references sub-files, agent reads those too
    No external code decides which skills to load — the agent does.
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        use_tools: bool = True,
        time_budget: float | None = None,
        extra_tool_schemas: list[dict] | None = None,
        hooks: list | None = None,
        mcp_bridges: list | None = None,
        enable_memory: bool = False,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.use_tools = use_tools
        self.extra_tool_schemas = extra_tool_schemas or []
        self.hooks = hooks or []
        self.time_budget = time_budget
        self.mcp_bridges = mcp_bridges or []  # list[McpBridge]
        self.enable_memory = enable_memory
        self._mcp_schemas: list[dict] = []

    def _connect_mcp_bridges(self) -> None:
        """Connect configured MCP bridges and cache their OpenAI tool schemas."""
        self._mcp_schemas = []
        for bridge in self.mcp_bridges:
            try:
                bridge.connect()
                self._mcp_schemas.extend(bridge.list_openai_schemas())
            except Exception as e:
                log.error(f"[{self.name}] MCP bridge connect failed: {e}")

    def _close_mcp_bridges(self) -> None:
        for bridge in self.mcp_bridges:
            try:
                bridge.close()
            except Exception as e:
                log.warning(f"[{self.name}] MCP bridge close failed: {e}")
        self._mcp_schemas = []

    def _execute_tool(
        self,
        fn_name: str,
        fn_args: dict,
        runtime_state: AgentRuntimeState,
    ) -> str:
        """Dispatch to an MCP bridge when it owns the tool; else local tools."""
        for bridge in self.mcp_bridges:
            if bridge.has_tool(fn_name):
                return bridge.call_tool(fn_name, fn_args)
        return tools.execute_tool(
            fn_name, fn_args, runtime_state=runtime_state, agent_name=self.name
        )

    def _create_runtime_state(self, task: str) -> AgentRuntimeState:
        return AgentRuntimeState(task_board=TaskBoard(goal=task))

    def _refresh_state_memory(
        self,
        messages: list[dict],
        runtime_state: AgentRuntimeState,
        task: str,
        iteration: int,
        trace: TraceWriter,
    ) -> list[dict]:
        """
        State compression.

        Iteration 1 seeds the board from the task (no LLM). Later iterations
        run LLM state compression every loop and persist progress.md.
        """
        board = runtime_state.task_board
        if iteration <= 1 or board.update_count == 0:
            seed_task_board(board, task)
            trace.context_event("state_seed", f"goal={board.goal[:80]}")
        elif board_says_stop(board):
            persist_task_board(board)
            trace.context_event("state_stop_skip_compress", board.next_action[:80])
        else:
            patch = compress_state(messages, board, llm_call_simple)
            if patch is None:
                # Keep prior board; still persist so progress.md stays available
                persist_task_board(board)
                trace.context_event("state_compress_failed", f"iter={iteration}")
            else:
                trace.context_event(
                    "state_compress",
                    f"step={board.current_step} updates={board.update_count}",
                )
                # Auto-clear recovery gates that previously required update_progress
                recovery = runtime_state.recovery
                if recovery.mode == "SPEC_RECHECK" and recovery.tools_in_mode > 0:
                    recovery.mode = "NORMAL"
                    recovery.failure_signature = ""
                    recovery.repeat_count = 0
                    recovery.tools_in_mode = 0
                    log.info("Recovery: SPEC_RECHECK cleared after state compression")
                if recovery.mode == "RETHINK":
                    board.requires_update = False
                    recovery.mode = "NORMAL"
                    recovery.failure_signature = ""
                    recovery.repeat_count = 0
                    recovery.tools_in_mode = 0
                    log.info("Recovery: RETHINK cleared after state compression")

        return build_working_memory(messages, task_board=board, load_defaults=True)

    def run(self, task: str) -> str:
        """
        Execute the agent loop until the model stops calling tools
        or we hit the iteration limit.

        Returns the final assistant text response.
        Writes a JSONL trace file to {LOG}/_trace_{name}.jsonl.
        Memory/compression layers run only when enable_memory is True (builder).
        """
        trace = TraceWriter(self.name)
        runtime_state = self._create_runtime_state(task)

        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task},
        ]

        client = get_client()
        consecutive_errors = 0
        last_text = ""

        self._connect_mcp_bridges()

        try:
            for iteration in range(1, config.MAX_AGENT_ITERATIONS + 1):
                stopped = False
                if self.enable_memory:
                    messages = self._refresh_state_memory(
                        messages, runtime_state, task, iteration, trace
                    )
                    stopped = board_says_stop(runtime_state.task_board)
                    if stopped and not runtime_state.stop_signaled:
                        runtime_state.stop_signaled = True
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "[SYSTEM] Task board next_action is STOP. "
                                    "Do not call any tools. Give a brief summary and finish."
                                ),
                            }
                        )
                        trace.context_event(
                            "state_stop", runtime_state.task_board.next_action[:80]
                        )
                        log.info(f"[{self.name}] Task board says STOP — wrapping up.")

                # --- Hooks: per-iteration ---
                for hook in self.hooks:
                    inject = hook.per_iteration(
                        iteration,
                        messages,
                        runtime_state=runtime_state,
                        agent_name=self.name,
                    )
                    if inject:
                        messages.append({"role": "user", "content": inject})
                        trace.hook_inject(type(hook).__name__, "per_iteration", inject)

                # --- Context lifecycle check ---
                token_count = context.count_tokens(messages)
                log.info(f"[{self.name}] iteration={iteration}  tokens≈{token_count}")
                trace.iteration(iteration, token_count)

                if self.enable_memory and (
                    token_count > config.RESET_THRESHOLD or context.detect_anxiety(
                        messages
                    )
                ):
                    reason = (
                        "anxiety detected"
                        if token_count <= config.RESET_THRESHOLD
                        else f"tokens {token_count} > threshold"
                    )
                    log.warning(
                        f"[{self.name}] Full-context compress (reset) triggered ({reason})..."
                    )
                    trace.context_event("full_compress", reason)
                    messages = full_compress_reset(
                        messages,
                        self.system_prompt,
                        llm_call_simple,
                        task_board=runtime_state.task_board,
                        trace_buffer=runtime_state.trace_buffer,
                    )
                    messages = _reinject_memory_blocks(messages, runtime_state)
                elif self.enable_memory and token_count > config.COMPRESS_THRESHOLD:
                    log.info(
                        f"[{self.name}] Trace-compressing context (role={self.name})..."
                    )
                    trace.context_event("trace_compress", f"tokens={token_count}")
                    messages = compress_trace(messages, runtime_state.trace_buffer)
                    messages = _reinject_memory_blocks(messages, runtime_state)

                # --- LLM call ---
                kwargs = dict(
                    model=config.MODEL,
                    messages=messages,
                    max_tokens=200000,
                )
                if self.use_tools:
                    kwargs["tools"] = (
                        tools.TOOL_SCHEMAS + self.extra_tool_schemas + self._mcp_schemas
                    )
                    kwargs["tool_choice"] = "none" if stopped else "auto"

                try:
                    trace.dump_messages(iteration, messages, token_count)
                except Exception:
                    pass

                try:
                    response = client.chat.completions.create(**kwargs)
                except Exception as e:
                    err_str = str(e)
                    trace.error("api_error", err_str)

                    if "rate_limit" in err_str.lower() or "429" in err_str:
                        import random

                        # 指数退避 + 上限控制 + 随机抖动
                        wait = min(2 ** (consecutive_errors + 2), 120) + random.uniform(
                            0, 5
                        )
                        log.warning(
                            f"[{self.name}] Rate limited, waiting {wait:.1f}s..."
                        )
                        time.sleep(wait)
                        continue

                    log.error(f"[{self.name}] API error: {e}")
                    consecutive_errors += 1
                    if consecutive_errors >= config.MAX_TOOL_ERRORS:
                        log.error(f"[{self.name}] Too many API errors, aborting.")
                        trace.finish("api_errors", iteration)
                        break
                    time.sleep(2**consecutive_errors)
                    continue

                consecutive_errors = 0

                # --- Guard against empty choices ---
                if not response.choices:
                    log.warning(
                        f"[{self.name}] API returned empty choices. Retrying..."
                    )
                    trace.error("empty_choices", "API returned no choices")
                    consecutive_errors += 1
                    if consecutive_errors >= config.MAX_TOOL_ERRORS:
                        log.error(f"[{self.name}] Too many empty responses, aborting.")
                        trace.finish("empty_choices", iteration)
                        break
                    time.sleep(2)
                    continue

                choice = response.choices[0]
                msg = choice.message

                # --- Append assistant message to history ---
                assistant_msg = {"role": "assistant", "content": msg.content}
                if msg.tool_calls:
                    assistant_msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                messages.append(assistant_msg)

                # --- Trace the LLM response ---
                trace.llm_response(
                    msg.content, assistant_msg.get("tool_calls"), choice.finish_reason
                )

                # --- If model produced text, capture it ---
                if msg.content:
                    last_text = msg.content
                    log.info(f"[{self.name}] assistant: {msg.content[:200]}...")

                if stopped and msg.tool_calls:
                    log.info(
                        f"[{self.name}] Ignoring tool calls because task board says STOP."
                    )
                    trace.finish("state_stop", iteration)
                    break

                # --- Hooks: pre-exit ---
                if not msg.tool_calls:
                    if stopped:
                        log.info(f"[{self.name}] Finished (task board STOP).")
                        trace.finish("state_stop", iteration)
                        break
                    forced_continue = False
                    for hook in self.hooks:
                        inject = hook.pre_exit(
                            messages, runtime_state=runtime_state, agent_name=self.name
                        )
                        if inject:
                            messages.append({"role": "user", "content": inject})
                            trace.hook_inject(type(hook).__name__, "pre_exit", inject)
                            forced_continue = True
                            break
                    if forced_continue:
                        continue
                    log.info(f"[{self.name}] Finished (no more tool calls).")
                    trace.finish("no_tool_calls", iteration)
                    break

                # --- Execute tool calls ---
                for tc in msg.tool_calls:
                    fn_name = tc.function.name
                    try:
                        fn_args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        log.warning(
                            f"[{self.name}] Bad JSON in tool call {fn_name}: {tc.function.arguments[:200]}"
                        )
                        trace.error(
                            "bad_json", f"{fn_name}: {tc.function.arguments[:200]}"
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": f"[error] Invalid JSON arguments: {tc.function.arguments[:200]}",
                            }
                        )
                        continue

                    # --- Hooks: before-tool ---
                    blocked = None
                    for hook in self.hooks:
                        blocked = hook.before_tool(
                            fn_name,
                            fn_args,
                            messages,
                            runtime_state=runtime_state,
                            agent_name=self.name,
                        )
                        if blocked:
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tc.id,
                                    "content": blocked,
                                }
                            )
                            trace.hook_inject(
                                type(hook).__name__, "before_tool", blocked
                            )
                            break
                    if blocked:
                        continue

                    if fn_name == "run_shell" and runtime_state.shell_session is None:
                        runtime_state.shell_session = PersistentShellSession(
                            config.WORKSPACE
                        )

                    log.info(
                        f"[{self.name}] tool: {fn_name}({_truncate(str(fn_args), 120)})"
                    )
                    raw = self._execute_tool(fn_name, fn_args, runtime_state)
                    result = compress_observation(fn_name, fn_args, raw)
                    log.debug(f"[{self.name}] tool result: {_truncate(result, 200)}")
                    trace.tool_call(fn_name, fn_args, result)

                    if self.enable_memory:
                        runtime_state.trace_buffer.add(
                            record_from_tool(
                                fn_name, fn_args, result, iteration=iteration
                            )
                        )

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        }
                    )

                    if (
                        fn_name in ACTION_TOOLS
                        and not result.startswith("[error]")
                        and not result.startswith("[blocked]")
                    ):
                        runtime_state.action_tool_count += 1
                        runtime_state.task_board.needs_final_update = True

                    # --- Hooks: post-tool ---
                    for hook in self.hooks:
                        inject = hook.post_tool(
                            fn_name,
                            fn_args,
                            result,
                            messages,
                            runtime_state=runtime_state,
                            agent_name=self.name,
                        )
                        if inject:
                            messages.append({"role": "user", "content": inject})
                            trace.hook_inject(type(hook).__name__, "post_tool", inject)
                            break

                # --- Check finish reason ---
                if choice.finish_reason == "stop":  # 模型正常退出
                    log.info(f"[{self.name}] Finished (stop).")
                    trace.finish("stop", iteration)
                    break

                if choice.finish_reason == "length":  # 超过长度限制
                    log.warning(f"[{self.name}] Output truncated (max_tokens hit).")
                    trace.error("length_truncated", "max_tokens hit")
                    # If tool calls were present, they were already executed above.
                    # Only tell the model they weren't executed if none were parsed
                    # (i.e. the truncation cut off the tool call JSON itself).
                    if msg.tool_calls:
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "[SYSTEM] Your response was truncated (token limit), but your tool calls "
                                    "WERE executed successfully. The results are above. "
                                    "If you had more tool calls planned, continue with the remaining ones now. "
                                    "Do NOT re-run the tools that already executed."
                                ),
                            }
                        )
                    else:
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "[SYSTEM] Your last response was cut off because it exceeded the token limit. "
                                    "No tool calls were executed. "
                                    "Please retry, but split large files into smaller parts:\n"
                                    "1. Write the first half of the file with write_file\n"
                                    "2. Then write the second half as a separate file or append\n"
                                    "Or simplify the implementation to fit in one response."
                                ),
                            }
                        )

            else:
                log.warning(
                    f"[{self.name}] Hit max iterations ({config.MAX_AGENT_ITERATIONS})."
                )
                trace.finish("max_iterations", config.MAX_AGENT_ITERATIONS)
        finally:
            if runtime_state.shell_session is not None:
                runtime_state.shell_session.close()
            self._close_mcp_bridges()

        return last_text
