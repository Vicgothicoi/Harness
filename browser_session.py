"""
Stateful browser session for testing web apps via Playwright.

Used by the Browser Testing MCP server. Keeps one Chromium page alive across
tool calls so the agent can open → observe → act → observe incrementally.

sync Playwright must run on the same OS thread that called
``sync_playwright().start()``. FastMCP in-process often dispatches each tool
onto a different worker, so every Playwright call is marshalled onto one
long-lived dedicated thread via a command queue.
"""

from __future__ import annotations

import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

import config

try:
    from playwright.sync_api import sync_playwright

    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

_STOP = object()
_INVOKE_TIMEOUT_S = 120.0


class BrowserSession:
    """Persistent headless browser + optional background dev server."""

    def __init__(self, workspace: str | Path | None = None):
        self.workspace = Path(workspace or config.WORKSPACE).resolve()
        self._playwright = None
        self._browser = None
        self._page = None
        self._console_errors: list[str] = []
        self._dev_server_proc: subprocess.Popen | None = None

        self._cmd_queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._worker_ident: int | None = None

    # ------------------------------------------------------------------
    # Dedicated Playwright thread
    # ------------------------------------------------------------------

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(
                target=self._worker_loop,
                name="playwright-session",
                daemon=True,
            )
            self._thread.start()

    def _worker_loop(self) -> None:
        self._worker_ident = threading.get_ident()
        try:
            while True:
                job = self._cmd_queue.get()
                if job is _STOP:
                    try:
                        self._close_impl()
                    except Exception:
                        pass
                    try:
                        self._stop_dev_server_impl()
                    except Exception:
                        pass
                    break
                fn, args, kwargs, event, box = job
                try:
                    box.append(fn(*args, **kwargs))
                except Exception as e:
                    box.append(e)
                finally:
                    event.set()
        finally:
            self._worker_ident = None

    def _invoke(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run ``fn`` on the Playwright worker thread and return its result."""
        self._ensure_worker()
        if (
            self._worker_ident is not None
            and threading.get_ident() == self._worker_ident
        ):
            return fn(*args, **kwargs)

        event = threading.Event()
        box: list[Any] = []
        self._cmd_queue.put((fn, args, kwargs, event, box))
        if not event.wait(timeout=_INVOKE_TIMEOUT_S):
            return (
                "[error] Playwright worker timed out "
                f"(>{int(_INVOKE_TIMEOUT_S)}s) waiting for {fn.__name__}"
            )
        if not box:
            return f"[error] Playwright worker returned no result for {fn.__name__}"
        result = box[0]
        if isinstance(result, Exception):
            return f"[error] {type(result).__name__}: {result}"
        return result

    def _stop_worker(self) -> None:
        with self._lock:
            thread = self._thread
            if thread is None or not thread.is_alive():
                self._thread = None
                return
            self._cmd_queue.put(_STOP)
        thread.join(timeout=15)
        with self._lock:
            if self._thread is thread:
                self._thread = None

    def shutdown(self) -> None:
        """Close browser, stop the dev server, and join the worker thread."""
        try:
            self.close()
        except Exception:
            pass
        try:
            self.stop_dev_server()
        except Exception:
            pass
        self._stop_worker()

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
        return self._invoke(self._open_impl, url, headless)

    def _open_impl(self, url: str | None, headless: bool) -> str:
        err = self._ensure_playwright()
        if err:
            return err

        if self._browser is not None:
            self._close_impl()

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
                return self._goto_impl(url)
            return "Browser opened (no URL yet — call browser_goto)"
        except Exception as e:
            self._close_impl()
            return f"[error] Failed to open browser: {e}"

    def goto(self, url: str) -> str:
        """Navigate the current page to a URL."""
        return self._invoke(self._goto_impl, url)

    def _goto_impl(self, url: str) -> str:
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
        return self._invoke(self._close_impl)

    def _close_impl(self) -> str:
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
        return self._invoke(
            self._start_dev_server_impl, start_command, port, startup_wait
        )

    def _start_dev_server_impl(
        self, start_command: str, port: int, startup_wait: int
    ) -> str:
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
        return self._invoke(self._stop_dev_server_impl)

    def _stop_dev_server_impl(self) -> str:
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
        return self._invoke(self._snapshot_impl, max_chars)

    def _snapshot_impl(self, max_chars: int) -> str:
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
        return self._invoke(self._screenshot_impl, path)

    def _screenshot_impl(self, path: str) -> str:
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
        return self._invoke(self._console_impl)

    def _console_impl(self) -> str:
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
        return self._invoke(self._click_impl, selector)

    def _click_impl(self, selector: str) -> str:
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
        return self._invoke(self._fill_impl, selector, value)

    def _fill_impl(self, selector: str, value: str) -> str:
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
        return self._invoke(self._wait_impl, delay_ms)

    def _wait_impl(self, delay_ms: int) -> str:
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
        return self._invoke(self._evaluate_impl, expression)

    def _evaluate_impl(self, expression: str) -> str:
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
        return self._invoke(self._scroll_impl, pixels)

    def _scroll_impl(self, pixels: int) -> str:
        err = self._require_page()
        if err:
            return err
        assert self._page is not None
        try:
            self._page.evaluate(f"window.scrollBy(0, {int(pixels)})")
            return f"Scrolled by {pixels}px"
        except Exception as e:
            return f"[error] scroll: {e}"
