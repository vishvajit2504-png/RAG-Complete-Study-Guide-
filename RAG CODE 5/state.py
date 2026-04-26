from typing import TypedDict, List
from langchain_core.documents import Document

class GraphState(TypedDict):
    """
    Shared state passed between all graph nodes.
    
    Attributes:
        question: The user's original question
        documents: Retrieved (and possibly filtered) documents
        generation: The final LLM answer
        web_search_needed: Whether to trigger Tavily fallback
    """
    question: str
    documents: List[Document]
    generation: str
    web_search_needed: str