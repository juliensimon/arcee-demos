from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent

from openai import OpenAI
import asyncio

API_KEY = "GtYzhob9zthYMxoq3a-wt1aP7yd3Ll2gRBNm04b-IJ1rOVOBUwmAuqp7Shso4aZDo01rYBFE9UXAfhf2CqX1v7TRjyo"

server_params = StdioServerParameters(
    command="anymcp",
    args=[
        "connect",
        "--debug",
        "--token",
        API_KEY,
        "https://mcp.arcee.ai/mcp/anymcp-b6543719/sse"
    ]
)
model = OpenAI(
    api_key=API_KEY,
    base_url="https://conductor.arcee.ai/v1",
)


async def setup_mcp_connection():
    """Establish the MCP connection and return session."""
    async with stdio_client(server_params) as (read, write):
        session = ClientSession(read, write)
        await session.initialize()
        return session


async def run_mcp_agent(session):
    """Run the agent with the provided session."""
    # Get tools
    tools = await load_mcp_tools(session)
    
    # Create and run the agent
    agent = create_react_agent(model, tools)
    prompt = "Find a villa in San Diego for 2 people for 2 nights starting on 2025-05-10"
    agent_response = await agent.ainvoke({"messages": prompt})
    print(agent_response)


async def main():
    # Setup MCP connection
    session = await setup_mcp_connection()
    
    # Run the agent with the session
    await run_mcp_agent(session)


if __name__ == "__main__":
    asyncio.run(main())
