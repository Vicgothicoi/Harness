from browser_mcp_client import BrowserMcpClient

c = BrowserMcpClient(transport="inprocess")
c.connect()
print("tools:", sorted(c.tool_names))
print(c.call_tool("browser_open", {"url": "https://example.com"}))
print(c.call_tool("browser_snapshot", {"max_chars": 500}))
print(c.call_tool("browser_close", {}))
c.close()