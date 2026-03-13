import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

SERVERS = {
    "Demo Server": {
        "transport": "stdio",
        "command": "uv",
        "args": [
            "run",
            "--with",
            "fastmcp",
            "fastmcp",
            "run",
            "D:\\Udemy\\fast-mcp-demo-server\\main.py"
        ]
    }
}

async def main():
    print("Hello from fast-mcp-demo-client!")
    client = MultiServerMCPClient(connections=SERVERS)
    tools = await client.get_tools()
    print("Tools available from the server:")
    for tool in tools:
        print(f"- {tool.name}: {tool.description}")

if __name__ == "__main__":
    asyncio.run(main())
