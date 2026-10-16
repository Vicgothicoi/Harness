"""Managed agent loop: LLM call, tools, hooks, and context lifecycle."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

from harness import config
from harness.compression import meter as context
from harness.compression.full import full_compress_reset
from harness.compression.observation import compress_observation
from harness.compression.state import compress_state
from harness.compression.trace import TraceBuffer, compress_trace, record_from_tool
from harness.hooks import RecoveryState
from harness.memory.state_memory import (
    TaskBoard,
    board_says_stop,
    persist_task_board,
    seed_task_board,
)
from harness.memory.working_memory import (
    build_working_memory,
    build_working_memory_from_runtime,
)
from harness.orchestration.llm import get_client, llm_call_simple
from harness.orchestration.trace import TraceWriter, _truncate
from harness.tools.registry import TOOL_SCHEMAS, execute_tool
from harness.tools.result import ToolResult, result_error, wrap_legacy
from harness.tools.shell import PersistentShellSession

log = logging.getLogger("harness")

ACTION_TOOLS = {"run_shell", "write_file", "delegate_task"}

def _reinject_memory_blocks(
    messages: list[dict], runtime_state: "AgentRuntimeState"
) -> list[dict]:
    """Re-project memory layers"""
    return build_working_memory_from_runtime(messages, runtime_state)

@dataclass
class AgentRuntimeState:
    shell_session: PersistentShellSession | None = None
    task_board: TaskBoard = field(default_factory=TaskBoard)
    recovery: RecoveryState = field(default_factory=RecoveryState)
    trace_buffer: TraceBuffer = field(default_factory=TraceBuffer)
    action_tool_count: int = 0
    stop_signaled: bool = False
    # Hook-set OpenAI tool_choice for this iteration (None = auto / none).
    tool_choice: str | dict | None = None

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
    ) -> ToolResult:
        """Dispatch to an MCP bridge when it owns the tool; else local tools."""
        for bridge in self.mcp_bridges:
            if bridge.has_tool(fn_name):
                return wrap_legacy(bridge.call_tool(fn_name, fn_args))
        return execute_tool(
            fn_name, fn_args, runtime_state=runtime_state, agent_name=self.name
        )

    def _commit_tool_result(
        self,
        *,
        fn_name: str,
        fn_args: dict,
        raw: ToolResult,
        tool_call_id: str,
        messages: list[dict],
        runtime_state: AgentRuntimeState,
        iteration: int,
        trace: TraceWriter,
        executed: bool,
    ) -> None:
        rendered = compress_observation(fn_name, fn_args, raw)
        log.debug(f"[{self.name}] tool result: {_truncate(rendered, 200)}")
        trace.tool_call(fn_name, fn_args, rendered)
        if self.enable_memory:
            runtime_state.trace_buffer.add(
                record_from_tool(fn_name, fn_args, raw, iteration=iteration)
            )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": rendered,
            }
        )
        if executed and fn_name in ACTION_TOOLS and raw.task_ok:
            runtime_state.action_tool_count += 1
            runtime_state.task_board.needs_final_update = True
        for hook in self.hooks:
            inject = hook.post_tool(
                fn_name,
                fn_args,
                raw,
                messages,
                runtime_state=runtime_state,
                agent_name=self.name,
            )
            if inject:
                messages.append({"role": "user", "content": inject})
                trace.hook_inject(type(hook).__name__, "post_tool", inject)
                break

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
                runtime_state.tool_choice = None
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
                        TOOL_SCHEMAS + self.extra_tool_schemas + self._mcp_schemas
                    )
                    if stopped:
                        kwargs["tool_choice"] = "none"
                    elif runtime_state.tool_choice is not None:
                        kwargs["tool_choice"] = runtime_state.tool_choice
                    else:
                        kwargs["tool_choice"] = "auto"

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
                        self._commit_tool_result(
                            fn_name=fn_name,
                            fn_args={},
                            raw=result_error(
                                f"[error] Invalid JSON arguments: {tc.function.arguments[:200]}"
                            ),
                            tool_call_id=tc.id,
                            messages=messages,
                            runtime_state=runtime_state,
                            iteration=iteration,
                            trace=trace,
                            executed=False,
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
                            trace.hook_inject(
                                type(hook).__name__, "before_tool", blocked
                            )
                            self._commit_tool_result(
                                fn_name=fn_name,
                                fn_args=fn_args,
                                raw=wrap_legacy(blocked),
                                tool_call_id=tc.id,
                                messages=messages,
                                runtime_state=runtime_state,
                                iteration=iteration,
                                trace=trace,
                                executed=False,
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
                    self._commit_tool_result(
                        fn_name=fn_name,
                        fn_args=fn_args,
                        raw=wrap_legacy(raw),
                        tool_call_id=tc.id,
                        messages=messages,
                        runtime_state=runtime_state,
                        iteration=iteration,
                        trace=trace,
                        executed=True,
                    )

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
