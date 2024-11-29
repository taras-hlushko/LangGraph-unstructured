from langgraph.graph import StateGraph
from my_agent.utils.nodes import fetch_and_process, display_results, save_to_astra
from my_agent.utils.state import AgentState

# Define the graph with state schema
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("process", fetch_and_process)
workflow.add_node("save", save_to_astra)
workflow.add_node("display", display_results)

# Define edges
workflow.add_edge("process", "save")
workflow.add_edge("save", "display")

# Set entry point
workflow.set_entry_point("process")

# Compile the graph
graph = workflow.compile()
