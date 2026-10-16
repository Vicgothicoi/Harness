"""Web search and page fetch tools."""
from __future__ import annotations

from harness.tools.result import ToolResult, result_error, result_ok
from harness.tools.workspace import RESULT_HARD_CAP, hard_cap

def web_search(query: str, max_results: int = 5) -> ToolResult:
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
            return result_ok(
                f"Search results for: {query}\n\n" + "\n".join(results)
            )

        return result_ok(f"No results found for: {query}")

    except Exception as e:
        return result_error(f"[error] Web search failed: {e}")

def web_fetch(url: str) -> ToolResult:
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
        html = resp.read(RESULT_HARD_CAP).decode("utf-8", errors="replace")

        # Strip HTML tags, keep text (conversation truncation is observation's job)
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        return result_ok(hard_cap(text) or "(empty page)")

    except Exception as e:
        return result_error(f"[error] Web fetch failed: {e}")
