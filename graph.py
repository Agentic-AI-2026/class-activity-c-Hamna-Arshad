import asyncio
from typing import TypedDict, Annotated, List, Optional
import operator
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, END

REACT_SYSTEM = """You are a ReAct agent. Follow this loop strictly:
Thought => Action (tool call) => Observation => Thought => ...
Give complete answer with reasoning for each step like:

Thought: ....
Action: tool_name with args {...}
Observation: result from tool call


for each of the tool calls.

At the end, summarizwe all the observations and give a final answer to the user's question.
."""

class AgentState(TypedDict):
    input: str
    agent_scratchpad: str
    final_answer: Optional[str]
    steps: Annotated[List[dict], operator.add]
    messages: List

def react_node(state, llm_with_tools):
    response = llm_with_tools.invoke(state["messages"])
    if response.tool_calls:
        return {**state, "messages": state["messages"] + [response], "final_answer": None}
    return {**state, "messages": state["messages"] + [response], "final_answer": response.content}

async def tool_node(state, tools_map):
    last_ai: AIMessage = state["messages"][-1]
    tool_messages, new_steps = [], []
    for tc in last_ai.tool_calls:
        result = await tools_map[tc["name"]].ainvoke(tc["args"])
        tool_messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
        new_steps.append({"action": tc["name"], "observation": str(result)})
    return {**state, "messages": state["messages"] + tool_messages, "steps": new_steps}

def router(state):
    return "end" if state.get("final_answer") else "tool_node"

def build_graph(llm, tools, tools_map):
    llt = llm.bind_tools(tools)

    def sync_tool_node(state):
        return asyncio.run(tool_node(state, tools_map))

    graph = StateGraph(AgentState)
    graph.add_node("react_node", lambda s: react_node(s, llt))
    graph.add_node("tool_node",  sync_tool_node)
    graph.set_entry_point("react_node")
    graph.add_conditional_edges("react_node", router, {"tool_node": "tool_node", "end": END})
    graph.add_edge("tool_node", "react_node")
    return graph.compile()

async def run_graph(app, user_input):
    state = {
        "input": user_input,
        "agent_scratchpad": "",
        "final_answer": None,
        "steps": [],
        "messages": [SystemMessage(content=REACT_SYSTEM), HumanMessage(content=user_input)],
    }
    result = await app.ainvoke(state)
    return result["final_answer"]