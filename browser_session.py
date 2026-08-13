"""
Stateful browser session for testing web apps via Playwright.

Used by the Browser Testing MCP server. Keeps one Chromium page alive across
tool calls so the agent can open → observe → act → observe incrementally.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import config

try:
    from playwright.sync_api import sync_playwright

    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False


class BrowserSession:
    """Persistent headless browser + optional background dev server."""

    def __init__(self, workspace: str | Path | None = None):
        self.workspace = Path(workspace or config.WORKSPACE).resolve()
        self._playwright = None
        self._browser = None
        self._page = None
        self._console_errors: list[str] = []
        self._dev_server_proc: subprocess.Popen | None = None

    # ------------------------------------------------------------------
    # Session
    # ------------------------------------------------------------------

    def _ensure_playwright(self) -> str | None:
        if not HAS_PLAYWRIGHT:
            return (
                "[error] Playwright not installed. "
                "Install with: pip install playwright && python -m playwright install chromium"
            )
        return None

    def _require_page(self) -> str | None:
        err = self._ensure_playwright()
        if err:
            return err
        if self._page is None:
            return "[error] No browser session. Call browser_open first."
        return None

    def open(self, url: str | None = None, headless: bool = True) -> str:
        """Launch Chromium and optionally navigate to a URL."""
        err = self._ensure_playwright()
        if err:
            return err

        if self._browser is not None:
            self.close()

        try:
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(headless=headless)
            self._page = self._browser.new_page(viewport={"width": 1280, "height": 720})
            self._console_errors = []
            self._page.on(
                "console",
                lambda msg: (
                    self._console_errors.append(msg.text)
                    if msg.type == "error"
                    else None
                ),
            )
            if url:
                return self.goto(url)
            return "Browser opened (no URL yet — call browser_goto)"
        except Exception as e:
            self.close()
            return f"[error] Failed to open browser: {e}"

    def goto(self, url: str) -> str:
        """Navigate the current page to a URL."""
        err = self._require_page()
        if err:
            return err
        assert self._page is not None
        try:
            self._page.goto(url, timeout=15000)
            return f"Navigated to {url} — title: {self._page.title()}"
        except Exception as e:
            return f"[error] Navigation failed: {e}"

    def close(self) -> str:
        """Close the browser session (does not stop the dev server)."""
        errors: list[str] = []
        try:
            if self._browser is not None:
                self._browser.close()
        except Exception as e:
            errors.append(f"browser close: {e}")
        try:
            if self._playwright is not None:
                self._playwright.stop()
        except Exception as e:
            errors.append(f"playwright stop: {e}")
        self._browser = None
        self._page = None
        self._playwright = None
        self._console_errors = []
        if errors:
            return "[error] Closed with issues: " + "; ".join(errors)
        return "Browser closed"

    def start_dev_server(
        self,
        start_command: str,
        port: int = 5173,
        startup_wait: int = 8,
    ) -> str:
        """Start a background dev server in the workspace."""
        if self._dev_server_proc is not None and self._dev_server_proc.poll() is None:
            return f"Dev server already running (pid={self._dev_server_proc.pid})"
        self._dev_server_proc = subprocess.Popen(
            start_command,
            shell=True,
            cwd=str(self.workspace),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(startup_wait)
        if self._dev_server_proc.poll() is not None:
            stderr = self._dev_server_proc.stderr.read().decode(errors="replace")[:2000]
            self._dev_server_proc = None
            return f"[error] Dev server exited immediately: {stderr}"
        return f"Dev server started (pid={self._dev_server_proc.pid}, port={port})"

    def stop_dev_server(self) -> str:
        """Stop the background dev server."""
        if self._dev_server_proc is None:
            return "No dev server running"
        self._dev_server_proc.terminate()
        try:
            self._dev_server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._dev_server_proc.kill()
        self._dev_server_proc = None
        return "Dev server stopped"

    # ------------------------------------------------------------------
    # Observe
    # ------------------------------------------------------------------

    def snapshot(self, max_chars: int = 2000) -> str:
        """Return URL, title, and visible body text."""
        err = self._require_page()
        if err:
            return err
        assert self._page is not None
        try:
            text = self._page.inner_text("body")
            if len(text) > max_chars:
                text = text[:max_chars] + f"\n...[{len(text) - max_chars} chars truncated]"
            return (
                f"URL: {self._page.url}\n"
                f"Title: {self._page.title()}\n"
                f"Visible text:\n{text}"
            )
        except Exception as e:
            return f"[error] Snapshot failed: {e}"

    def screenshot(self, path: str = "_screenshot.png") -> str:
        """Save a viewport screenshot into the workspace."""
        err = self._require_page()
        if err:
            return err
        assert self._page is not None
        try:
            ss_path = self.workspace / path
            ss_path.parent.mkdir(parents=True, exist_ok=True)
            self._page.screenshot(path=str(ss_path), full_page=False)
            return f"Screenshot saved to {path}"
        except Exception as e:
            return f"[error] Screenshot failed: {e}"

    def console(self) -> str:
        """Return captured browser console errors since open."""
        err = self._require_page()
        if err:
            return err
        if not self._console_errors:
            return "No console errors captured"
        lines = [f"Console errors ({len(self._console_errors)}):"]
        for msg in self._console_errors[:20]:
            lines.append(f"  - {msg[:300]}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def click(self, selector: str) -> str:
        err = self._require_page()
        if err:
            return err
        assert self._page is not None
        try:
            self._page.click(selector, timeout=5000)
            return f"Clicked: {selector}"
        except Exception as e:
            return f"[error] click('{selector}'): {e}"

    def fill(self, selector: str, value: str) -> str:
        err = self._require_page()
        if err:
            return err
        assert self._page is not None
        try:
            self._page.fill(selector, value, timeout=5000)
            preview = value if len(value) <= 50 else value[:50] + "..."
            return f"Filled '{selector}' with {preview!r}"
        except Exception as e:
            return f"[error] fill('{selector}'): {e}"

    def wait(self, delay_ms: int = 1000) -> str:
        err = self._require_page()
        if err:
            return err
        assert self._page is not None
        try:
            self._page.wait_for_timeout(delay_ms)
            return f"Waited {delay_ms}ms"
        except Exception as e:
            return f"[error] wait: {e}"

    def evaluate(self, expression: str) -> str:
        err = self._require_page()
        if err:
            return err
        assert self._page is not None
        try:
            result = self._page.evaluate(expression)
            return f"JS eval result: {str(result)[:2000]}"
        except Exception as e:
            return f"[error] evaluate: {e}"

    def scroll(self, pixels: int = 500) -> str:
        err = self._require_page()
        if err:
            return err
        assert self._page is not None
        try:
            self._page.evaluate(f"window.scrollBy(0, {int(pixels)})")
            return f"Scrolled by {pixels}px"
        except Exception as e:
            return f"[error] scroll: {e}"
