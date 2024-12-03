from typing import Dict, Any
from langgraph.graph import StateGraph
from my_agent.utils.state import AgentState
from my_agent.generate_topics_agent import generate_topics_graph
from my_agent.store_doc_by_url_agent import store_doc_by_url_graph

def select_flow(state: Dict[str, Any]) -> Dict[str, Any]:
    """Route to the appropriate agent based on selection"""
    # Pass through the URL while selecting the agent
    flow_type = state.get("config", {}).get("flow_type", "store_doc_by_url")
    return {
        "flow": flow_type,
        "url": state["url"]  # Ensure URL is passed through
    }

# Create the main graph
main_workflow = StateGraph(dict)

# Add the flow selection node
main_workflow.add_node("select", select_flow)

# Add the flows as nodes
main_workflow.add_node("generate_topics", generate_topics_graph)
main_workflow.add_node("store_doc_by_url", store_doc_by_url_graph)

# Add conditional edges based on flow selection
main_workflow.add_conditional_edges(
    "select",
    lambda x: x["flow"],
    {
        "generate_topics": "generate_topics",
        "store_doc_by_url": "store_doc_by_url"
    }
)

# Set entry point
main_workflow.set_entry_point("select")

# Compile the graph
graph = main_workflow.compile()
