from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from state import GraphState
from chains import retrieval_grader, question_rewriter, rag_chain
from retriever import build_retriever

retriever = build_retriever()
web_search_tool = TavilySearchResults(max_results=3)


def retrieve(state: GraphState) -> GraphState:
    """Node 1: Retrieve docs from vector store."""
    print("--- NODE: RETRIEVE ---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


def grade_documents(state: GraphState) -> GraphState:
    """
    Node 2: Grade each retrieved document.
    Sets web_search_needed='yes' if any doc is irrelevant.
    Filters out irrelevant docs from state.
    """
    print("--- NODE: GRADE DOCUMENTS ---")
    question = state["question"]
    documents = state["documents"]
    
    filtered_docs = []
    web_search_needed = "no"
    
    for doc in documents:
        score = retrieval_grader.invoke({
            "question": question,
            "document": doc.page_content
        })
        if score.binary_score == "yes":
            print(f"  ✅ RELEVANT: {doc.page_content[:80]}...")
            filtered_docs.append(doc)
        else:
            print(f"  ❌ IRRELEVANT: {doc.page_content[:80]}...")
            web_search_needed = "yes"  # at least one bad doc → flag it
    
    return {
        "documents": filtered_docs,
        "question": question,
        "web_search_needed": web_search_needed
    }


def rewrite_query(state: GraphState) -> GraphState:
    """Node 3: Rewrite question for better retrieval."""
    print("--- NODE: REWRITE QUERY ---")
    question = state["question"]
    better_question = question_rewriter.invoke({"question": question})
    print(f"  Original: {question}")
    print(f"  Rewritten: {better_question}")
    return {"question": better_question, "documents": state["documents"]}


def web_search(state: GraphState) -> GraphState:
    """Node 4: Tavily fallback search. Appends web results to documents."""
    print("--- NODE: WEB SEARCH ---")
    question = state["question"]
    results = web_search_tool.invoke({"query": question})
    
    # Convert Tavily results to Document objects so generate() can use them uniformly
    web_docs = []
    for r in results:
        if isinstance(r, dict):
            web_docs.append(Document(
                page_content=r.get("content", r.get("snippet", "")),
                metadata={"source": r.get("url", r.get("href", "")), "type": "web"}
            ))
        else:
            web_docs.append(Document(page_content=str(r), metadata={"type": "web"}))
    
    # Combine with any relevant docs we still have
    combined = state["documents"] + web_docs
    return {"documents": combined, "question": question}


def generate(state: GraphState) -> GraphState:
    """Node 5: Generate final answer from filtered context."""
    print("--- NODE: GENERATE ---")
    question = state["question"]
    documents = state["documents"]
    
    context = "\n\n".join(doc.page_content for doc in documents)
    generation = rag_chain.invoke({"context": context, "question": question})
    
    return {"generation": generation, "documents": documents, "question": question}