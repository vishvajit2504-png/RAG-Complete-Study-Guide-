import logging
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from embedder import embed_and_store
from document_loader import load_documents_from_folder
from text_splitter import split_documents
import os
from config import get_settings
from embedder import get_vector_store, get_embeddings

settings = get_settings()
logger = logging.getLogger(__name__)

def get_base_retriever(vectorstore: Chroma = None):

    if vectorstore is None:
        vectorstore = get_vectorstore()

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.retrieval_top_k}
    )
    logger.info(f"Base retriever ready — top_k={settings.retrieval_top_k}")
    return retriever

def get_mmr_retriever(vectorstore: Chroma = None):

    if vectorstore is None:
        vectorstore = get_vectorstore()

    retriver = vectorstore.as_retriever(
        search_type = "mmr",
        search_kwargs={
            "k": settings.retrieval_top_k,
            "fetch_k":settings.retrieval_top_k*3,
            "lambda_mult":0.7 
        }
    )
    logger.info(f"MMR retriever ready — top_k={settings.retrieval_top_k}, fetch_k={settings.retrieval_top_k*3}, lambda_mult=0.7")
    return retriver

def retrieve_with_scores(query:str, vectorstore:Chroma= None) -> List[tuple[Document, float]]:

    if vectorstore is None:
        vectorstore = get_vectorstore()

    results = vectorstore.similarity_search_with_score(query, k=settings.retrieval_top_k)
    for doc, score in results:
        logger.info(f"Score: {score:.4f} | Source: {doc.metadata.get('source')} | "
                    f"Chunk: {doc.metadata.get('chunk_index')}") 
    return results

def filter_by_score(results : List[tuple[Document,float]],threshold:float = None ) -> List[Document]:

    threshold = threshold or settings.similarity_threshold

    filtered = [doc for doc, score in results if score >= threshold]
    dropped = len(results) - len(filtered)

    if dropped:
        logger.info(f"Filtered out {dropped} low-quality chunk(s) below threshold {threshold}")

    return filtered

def get_retriever(mode : str = "mmr" , vectorstore: Chroma = None):

    if vectorstore is None:
        vectorstore = get_vectorstore()

    if mode == "mmr":
        return get_mmr_retriever(vectorstore)
    else:
        return get_base_retriever(vectorstore)
    
if __name__ == "__main__":
    docs_path = os.path.join(os.path.dirname(__file__), "papers")
    docs = load_documents_from_folder(docs_path)
    chunks = split_documents(docs)
    vs = embed_and_store(chunks)

    # Test scored retrieval
    query = "What is Attention is all you need about?"
    results = retrieve_with_scores(query, vs)
    filtered = filter_by_score(results)

    print(f"\nTop chunks after filtering:")
    print(len(filtered))
    for doc in filtered:
        print(doc.metadata)
        print(doc.page_content)
        print("---")