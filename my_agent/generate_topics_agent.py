from langgraph.graph import StateGraph
from my_agent.utils.state import AgentState
from my_agent.utils.nodes import display_results, generate_curriculum_topics

# Define the graph with state schema
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("generate_curriculum_topics", generate_curriculum_topics)
workflow.add_node("display", display_results)

# Define edges
workflow.add_edge("generate_curriculum_topics", "display")

# Set entry point
workflow.set_entry_point("generate_curriculum_topics")

# Compile the graph
generate_topics_graph = workflow.compile() 