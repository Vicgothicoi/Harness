"""
Browser Testing MCP server — session / observe / action tools only.

Exposes a stateful Playwright browser so MCP clients can drive UI tests
incrementally (open → snapshot → click/fill → screenshot → close).

Usage:
  # stdio transport (client spawns this process)
  python browser_mcp_server.py

  # SSE/HTTP transport (client connects over network)
  python browser_mcp_server.py --transport sse --port 8000

Environment:
  HARNESS_WORKSPACE   workspace directory (default: ./workspace)
"""

from __future__ import annotations

import argparse
import atexit

from fastmcp import FastMCP

from browser_session import BrowserSession

# ---------------------------------------------------------------------------
# Server instance
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "browser-testing",
    instructions=(
        "Stateful browser testing tools powered by Playwright. "
        "Typical flow: start_dev_server (optional) → browser_open → "
        "browser_goto → browser_snapshot / browser_click / browser_fill → "
        "browser_screenshot → browser_close → stop_dev_server. "
        "Screenshots are written relative to HARNESS_WORKSPACE."
    ),
)

# ---------------------------------------------------------------------------
# Shared browser session (one per MCP process)
# ---------------------------------------------------------------------------

_session = BrowserSession()


def _cleanup() -> None:
    _session.shutdown()


atexit.register(_cleanup)

# ---------------------------------------------------------------------------
# Session tools
# ---------------------------------------------------------------------------


@mcp.tool()
def browser_open(url: str | None = None, headless: bool = True) -> str:
    """Launch a Chromium browser session. Optionally navigate to url."""
    return _session.open(url=url, headless=headless)


@mcp.tool()
def browser_goto(url: str) -> str:
    """Navigate the current page to a URL. Requires browser_open first."""
    return _session.goto(url)


@mcp.tool()
def browser_close() -> str:
    """Close the browser session. Does not stop the dev server."""
    return _session.close()


@mcp.tool()
def start_dev_server(
    start_command: str,
    port: int = 5173,
    startup_wait: int = 8,
) -> str:
    """Start a background dev server in the workspace (e.g. 'npm run dev')."""
    return _session.start_dev_server(
        start_command, port=port, startup_wait=startup_wait
    )


@mcp.tool()
def stop_dev_server() -> str:
    """Stop the background dev server started by start_dev_server."""
    return _session.stop_dev_server()


# ---------------------------------------------------------------------------
# Observe tools
# ---------------------------------------------------------------------------


@mcp.tool()
def browser_snapshot(max_chars: int = 2000) -> str:
    """Return current URL, title, and visible body text."""
    return _session.snapshot(max_chars=max_chars)


@mcp.tool()
def browser_screenshot(path: str = "_screenshot.png") -> str:
    """Save a viewport screenshot into the workspace."""
    return _session.screenshot(path=path)


@mcp.tool()
def browser_console() -> str:
    """Return console errors captured since browser_open."""
    return _session.console()


# ---------------------------------------------------------------------------
# Action tools
# ---------------------------------------------------------------------------


@mcp.tool()
def browser_click(selector: str) -> str:
    """Click an element matched by a CSS selector."""
    return _session.click(selector)


@mcp.tool()
def browser_fill(selector: str, value: str) -> str:
    """Fill an input matched by a CSS selector with the given value."""
    return _session.fill(selector, value)


@mcp.tool()
def browser_wait(delay_ms: int = 1000) -> str:
    """Wait for delay_ms milliseconds on the current page."""
    return _session.wait(delay_ms)


@mcp.tool()
def browser_evaluate(expression: str) -> str:
    """Evaluate a JavaScript expression in the page and return the result."""
    return _session.evaluate(expression)


@mcp.tool()
def browser_scroll(pixels: int = 500) -> str:
    """Scroll the page vertically by the given pixel amount."""
    return _session.scroll(pixels)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Browser Testing MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport type (default: stdio)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host for SSE transport (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for SSE transport (default: 8000)",
    )
    args = parser.parse_args()

    if args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        mcp.run()  # stdio
