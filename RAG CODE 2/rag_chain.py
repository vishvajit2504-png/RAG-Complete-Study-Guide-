import logging

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from config import get_settings

from retriever import get_retriever, retrieve_with_scores, filter_by_score
from llm import get_llm, get_rag_prompt
from embedder import embed_and_store, get_embeddings,get_vector_store
from document_loader import load_documents_from_folder
from text_splitter import split_documents
import os
settings = get_settings()
logger = logging.getLogger(__name__)


def format_context(docs)-> str:
    if not docs:
        return "No relevant information found in the knowledge base."
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source","unknown")
        chunk  = doc.metadata.get("chunk","?")
        formatted.append(f"Source:{source}, Chunk: {chunk}\n Content:{doc.page_content}")
    logger.info(f"Formatted textfrom documents  {formatted}into context for LLM.")
    return "\n\n---\n\n".join(formatted)

def build_rag_chain():
    embeddings = get_embeddings()
    vectorstore = get_vector_store(embeddings)
    retriever = get_retriever(vectorstore)
    rag_prompt = get_rag_prompt()
    llm = get_llm()
    
    chain = (
        {
            "context": retriever | RunnableLambda(format_context),
            "question": RunnablePassthrough()  
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    logger.info(f"RAG chain built with mode: mmr")
    return chain

def query_with_sources(query:str, mode: str = "similarity"):

    vectorstore = get_vector_store(get_embeddings())
    logger.info(f"Retrieval mode: {mode}")
    scored_results = retrieve_with_scores(query, vectorstore)
    print("\nRetrieved documents with scores:")
    filtered_docs = filter_by_score(scored_results)
    print(f"\n{len(filtered_docs)} documents passed the similarity threshold ")
    if len(filtered_docs) == 0:
        logger.warning("No relevant documents found for the query.")
        return {
            "answer":  "I don't have enough information to answer that.",
            "sources": [],
            "chunks_used": 0
        }
    context = format_context(filtered_docs)
    logger.info(f"Context prepared with  {context} documents for LLM.")
    logger.info(f"Invoking LLM with query: {query}")
    rag_prompt = get_rag_prompt()
    llm = get_llm()
    chain  = rag_prompt | llm | StrOutputParser()
    answer = chain.invoke({"context":context,"question":query})
    logger.info(f"LLM generated answer: {answer}")
    sources = list({doc.metadata.get("source") for doc in filtered_docs})

    return {
        "answer": answer,
        "sources" : sources,
        "chunks_used": len(filtered_docs)   
    }

if __name__ == "__main__":

    docs_path = os.path.join(os.path.dirname(__file__), "papers")
    docs   = load_documents_from_folder(docs_path)
    chunks = split_documents(docs)
    embed_and_store(chunks)

    # Test simple chain
    chain  = build_rag_chain()
    answer = chain.invoke("What caused the attention is all you need paper to be so influential?")
    print("Simple chain answer:\n", answer)

    print("\n" + "="*50 + "\n")

    # Test with sources
    result = query_with_sources("What caused the attention is all you need paper to be so influential?")
    print("Answer:", result["answer"])
    print("Sources:", result["sources"])
    print("Chunks used:", result["chunks_used"])