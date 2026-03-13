import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv(override=True)

# Ensure the API token is set
if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    raise ValueError("HUGGINGFACEHUB_API_TOKEN environment variable is not set. Please set it in your .env file or environment.")
else:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

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

    llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2")
    model = ChatHuggingFace(llm=llm)

    model_with_tool = model.bind_tools(tools)

    prompt = "add 25 and 35 using add_numbers and return result"
    response = await model_with_tool.ainvoke(prompt)
    print(f"Response from model: {response}")    

if __name__ == "__main__":
    asyncio.run(main())
