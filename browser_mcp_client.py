"""
MCP bridge client for the Browser Testing server.

Implements the McpBridge protocol expected by AgentConfig.mcp_bridges:
  connect() / list_openai_schemas() / call_tool() / close()

Default transport is in-process (Client(mcp)) for same-repo use.
Pass transport="stdio" to spawn browser_mcp_server.py as a subprocess.

Other MCP servers can ship their own client classes that satisfy McpBridge;
AgentConfig.mcp_bridges is intentionally a heterogeneous list of bridges.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

log = logging.getLogger("harness")

Transport = Literal["inprocess", "stdio"]


@runtime_checkable
class McpBridge(Protocol):
    """Minimal contract for any MCP bridge hung on AgentConfig.mcp_bridges."""

    def connect(self) -> None:
        """Open the MCP session and cache tool schemas."""
        ...

    def list_openai_schemas(self) -> list[dict]:
        """Return OpenAI function-calling schemas for this server's tools."""
        ...

    def call_tool(self, name: str, arguments: dict | None = None) -> str:
        """Invoke a tool and return a plain-text result for the agent loop."""
        ...

    def close(self) -> None:
        """Tear down the MCP session (idempotent)."""
        ...

    def has_tool(self, name: str) -> bool:
        """Whether this bridge owns the given tool name."""
        ...


def _mcp_tool_to_openai_schema(tool: Any) -> dict:
    """Convert an MCP Tool object into an OpenAI tools[] entry."""
    params = getattr(tool, "inputSchema", None) or {
        "type": "object",
        "properties": {},
    }
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": (getattr(tool, "description", None) or "").strip(),
            "parameters": params,
        },
    }


def _result_to_text(result: Any) -> str:
    """Normalize a FastMCP CallToolResult (or similar) to a string."""
    data = getattr(result, "data", None)
    if isinstance(data, str):
        return data
    if data is not None:
        return str(data)

    content = getattr(result, "content", None) or []
    texts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if text:
            texts.append(text)
        elif isinstance(block, dict) and block.get("text"):
            texts.append(str(block["text"]))
    if texts:
        return "\n".join(texts)

    if result is None:
        return "(no output)"
    return str(result)


class BrowserMcpClient:
    """Sync facade over FastMCP Client for browser_mcp_server."""

    def __init__(
        self,
        transport: Transport = "inprocess",
        server_script: str | Path | None = None,
    ):
        self.transport: Transport = transport
        self.server_script = Path(
            server_script or Path(__file__).parent / "browser_mcp_server.py"
        ).resolve()

        self._loop: asyncio.AbstractEventLoop | None = None
        self._client: Any = None
        self._schemas: list[dict] = []
        self._tool_names: set[str] = set()
        self._connected = False

    # ------------------------------------------------------------------
    # McpBridge API
    # ------------------------------------------------------------------

    def connect(self) -> None:
        if self._connected:
            return
        self._loop = asyncio.new_event_loop()
        try:
            self._loop.run_until_complete(self._async_connect())
            self._connected = True
            log.info(
                "BrowserMcpClient connected (%s) — %d tools",
                self.transport,
                len(self._tool_names),
            )
        except Exception:
            self._cleanup_loop()
            raise

    def list_openai_schemas(self) -> list[dict]:
        if not self._connected:
            self.connect()
        return list(self._schemas)

    def call_tool(self, name: str, arguments: dict | None = None) -> str:
        if not self._connected or self._loop is None or self._client is None:
            return "[error] BrowserMcpClient is not connected. Call connect() first."
        if name not in self._tool_names:
            return f"[error] Unknown browser MCP tool: {name}"
        try:
            return self._loop.run_until_complete(
                self._async_call_tool(name, arguments or {})
            )
        except Exception as e:
            return f"[error] MCP call_tool({name}) failed: {type(e).__name__}: {e}"

    def close(self) -> None:
        if not self._connected:
            self._cleanup_loop()
            return
        assert self._loop is not None
        try:
            self._loop.run_until_complete(self._async_close())
        except Exception as e:
            log.warning("BrowserMcpClient close error: %s", e)
        finally:
            self._connected = False
            self._client = None
            self._schemas = []
            self._tool_names = set()
            self._cleanup_loop()
            log.info("BrowserMcpClient closed")

    def has_tool(self, name: str) -> bool:
        return name in self._tool_names

    @property
    def tool_names(self) -> set[str]:
        return set(self._tool_names)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _cleanup_loop(self) -> None:
        if self._loop is not None:
            try:
                self._loop.close()
            except Exception:
                pass
            self._loop = None

    def _build_client(self) -> Any:
        from fastmcp import Client

        if self.transport == "inprocess":
            from browser_mcp_server import mcp

            return Client(mcp)
        if self.transport == "stdio":
            if not self.server_script.is_file():
                raise FileNotFoundError(
                    f"Browser MCP server script not found: {self.server_script}"
                )
            return Client(str(self.server_script))
        raise ValueError(f"Unsupported transport: {self.transport!r}")

    async def _async_connect(self) -> None:
        client = self._build_client()
        await client.__aenter__()
        self._client = client
        tools = await client.list_tools()
        self._schemas = [_mcp_tool_to_openai_schema(t) for t in tools]
        self._tool_names = {t.name for t in tools}

    async def _async_call_tool(self, name: str, arguments: dict) -> str:
        assert self._client is not None
        # Prefer raw protocol result so tool-level errors become strings,
        # not raised exceptions that break the agent loop.
        if hasattr(self._client, "call_tool_mcp"):
            raw = await self._client.call_tool_mcp(name, arguments)
            if getattr(raw, "isError", False):
                return f"[error] {_result_to_text(raw)}"
            return _result_to_text(raw)
        result = await self._client.call_tool(name, arguments)
        return _result_to_text(result)

    async def _async_close(self) -> None:
        if self._client is None:
            return
        # Best-effort session cleanup before dropping the connection.
        for tool_name in ("browser_close", "stop_dev_server"):
            if tool_name in self._tool_names:
                try:
                    await self._async_call_tool(tool_name, {})
                except Exception:
                    pass
        await self._client.__aexit__(None, None, None)
