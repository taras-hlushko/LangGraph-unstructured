from typing import TypedDict, Optional, List, Any

class AgentState(TypedDict, total=False):
    url: str  # The input URL
    elements: Optional[List[Any]]  # Raw elements from Unstructured.io
    summarized_elements: Optional[List[dict]]  # Elements with their summaries
    topic_summaries: Optional[str]  # Curriculum topics JSON
    error: Optional[str]  # Any error messages
