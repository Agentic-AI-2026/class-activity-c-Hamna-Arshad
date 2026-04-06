import asyncio, sys, os
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from graph import build_graph, run_graph
from dotenv import load_dotenv
load_dotenv()
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.environ["GROQ_API_KEY"], temperature=0)

mcp_client = MultiServerMCPClient({
    "math":    {"command": sys.executable, "args": ["Tools/math_server.py"],   "transport": "stdio"},
    "search":  {"command": sys.executable, "args": ["Tools/search_server.py"], "transport": "stdio"},
    "weather": {"url": "http://localhost:8000/mcp", "transport": "streamable_http"},
})

async def main():
    tools = []
    for server in ["math", "search", "weather"]:
        tools.extend(await mcp_client.get_tools(server_name=server))
    tools_map = {t.name: t for t in tools}

    app = build_graph(llm, tools, tools_map)

    query = ("What is the weather in Lahore and who is the current Prime Minister of Pakistan? "
             "Now tell me about the founding year of Microsoft and then calculate the sum of that year and the current year. What "
             "temperature in Fahrenheit corresponds to the current average temperature in Lahore?")

    answer = await run_graph(app, query)
    print(answer)

if __name__ == "__main__":
    asyncio.run(main())