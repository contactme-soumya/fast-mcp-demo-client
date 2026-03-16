import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import ToolMessage
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

    print(f"response tool calls: {response.tool_calls}")

    # If no tool calls, manually execute the tool
    if not response.tool_calls:
        # Find the add_numbers tool
        add_tool = next((tool for tool in tools if tool.name == "add_numbers"), None)
        if add_tool:
            # Call the tool with args
            tool_result = await add_tool.ainvoke({"a": 25, "b": 35})
            print(f"Tool result: {tool_result}")
            
            # Now, pass the result back to the LLM
            follow_up_prompt = f"The result of adding 25 and 35 is {tool_result}. Please explain this result."
            follow_up_response = await model.ainvoke(follow_up_prompt)
            print(f"Follow-up response content: {follow_up_response.content}")
        else:
            print("add_numbers tool not found")
           
    else:
        # Handle actual tool calls if any
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool = next((t for t in tools if t.name == tool_name), None)
            if tool:
                result = await tool.ainvoke(tool_args)
                print(f"Tool {tool_name} result: {result}")
                # Pass result back to LLM
                follow_up = f"Tool {tool_name} with args {tool_args} returned: {result}"
                follow_up_response = await model.ainvoke(follow_up)
                print(f"Follow-up: {follow_up_response.content}")

    print("---------going to form tool message to send back output to LLM----------")
    tool_message = ToolMessage(content=follow_up_response, tool_call_id=follow_up_response.id)
    final_response = await model.ainvoke([follow_up_prompt,follow_up_response, tool_message]) 
    print(f"Final response: {final_response.content}")        

 

if __name__ == "__main__":
    asyncio.run(main())
