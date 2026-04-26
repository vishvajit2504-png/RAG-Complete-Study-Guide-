from langgraph.graph import StateGraph, END
from state import GraphState
from nodes import retrieve, grade_documents, rewrite_query, web_search, generate


def decide_after_grading(state: GraphState) -> str:
    """
    Conditional edge function.
    Returns the name of the NEXT NODE to route to.
    
    Logic:
    - No relevant docs at all → web_search (strongest fallback)
    - Some irrelevant docs found → rewrite_query (try again with better query)
    - All docs relevant → generate (go straight to answer)
    """
    web_search_needed = state["web_search_needed"]
    has_docs = len(state["documents"]) > 0
    
    if not has_docs:
        print("--- DECISION: All irrelevant → Web Search ---")
        return "web_search"
    elif web_search_needed == "yes":
        print("--- DECISION: Mixed relevance → Rewrite Query ---")
        return "rewrite_query"
    else:
        print("--- DECISION: All relevant → Generate ---")
        return "generate"


def build_graph():
    workflow = StateGraph(GraphState)
    
    # Register nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate", generate)
    
    # Entry point
    workflow.set_entry_point("retrieve")
    
    # Linear edge: retrieve → grade
    workflow.add_edge("retrieve", "grade_documents")
    
    # Conditional edge after grading
    workflow.add_conditional_edges(
        "grade_documents",           # from this node...
        decide_after_grading,        # ...call this function to get next node name
        {                            # map return values to node names
            "web_search": "web_search",
            "rewrite_query": "rewrite_query",
            "generate": "generate",
        }
    )
    
    # After rewrite → retrieve again (the "corrective" loop!)
    workflow.add_edge("rewrite_query", "retrieve")
    
    # After web search → generate
    workflow.add_edge("web_search", "generate")
    
    # Generate → END
    workflow.add_edge("generate", END)
    
    return workflow.compile()