from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from config import llm_with_tools, memory, tools
from state import State



def build_graph() -> StateGraph:
    """
    Builds and compiles a StateGraph for managing chatbot and tool interactions.

    This function constructs a state graph with nodes for a chatbot and tools, 
    sets up conditional and direct edges between them, and specifies the starting node. 
    The graph is compiled with a checkpointer for state management.

    Returns:
        StateGraph: The compiled state graph ready for execution.
    """
    graph_builder = StateGraph(State)

    def chatbot(state: State) -> dict:
        message = llm_with_tools.invoke(state["messages"])
        return {"message": [message]}
    graph_builder.add_node("chatbot", chatbot)

    tool_node = ToolNode(tools=tools)

    graph_builder.add_node("tools", tool_node)

    graph_builder.add_conditional_edges("chatbot", tools_condition)
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")

    return graph_builder.compile(checkpointer=memory)