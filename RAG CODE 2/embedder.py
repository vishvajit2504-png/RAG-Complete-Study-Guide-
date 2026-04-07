# embedder.py
import logging
from typing import List

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from document_loader import load_documents_from_folder
from text_splitter import split_documents

from config import get_settings
import os

settings = get_settings()
logger = logging.getLogger(__name__)

def get_embeddings() -> OllamaEmbeddings:
   return OllamaEmbeddings(
      base_url = settings.ollama_base_url,
        model = settings.ollama_embedding_model
   )

def get_vector_store(embeddings: OllamaEmbeddings) -> Chroma:

   if embeddings is None:
     embeddings = get_embeddings()
   return Chroma(
      collection_name = settings.chroma_collection_name,
      embedding_function = embeddings,
      persist_directory = settings.chroma_persist_dir
   )
    
    
def embed_and_store(chunks: List[Document]) -> Chroma:
   
   if not chunks:
      raise ValueError("Chunks cannot be None")
   

   
   embeddings = get_embeddings()
   vector_store = get_vector_store(embeddings)

   existing_count = vector_store._collection.count()

   if existing_count >0:
      logger.info(
            f"ChromaDB already has {existing_count} vectors. "
            f"Skipping re-embedding. Delete ./chroma_db to re-index."
        )
      return vector_store
   
   logger.info(f"Embedding {len(chunks)} chunks with '{settings.ollama_embedding_model}'...")
   vector_store.add_documents(documents=chunks)
   logger.info(f"Stored {len(chunks)} vectors in ChromaDB at '{settings.chroma_persist_dir}'")

   return vector_store

def similarity_search(query:str,vector_store: Chroma, top_k:int =settings.retrieval_top_k) -> List[Document]:
   if vector_store is None:
      raise ValueError("Vector store cannot be None")
   if not query:
      raise ValueError("Query cannot be empty")
   
   results = vector_store.similarity_search(query, k=top_k)

   logger.info(f"Retrieved {len(results)} relevant documents for query: '{query}'")    
   return results  

if __name__ == "__main__":

    docs_path = os.path.join(os.path.dirname(__file__), "papers")
    docs = load_documents_from_folder(docs_path)
    chunks = split_documents(docs)

    vs = embed_and_store(chunks)

    # Test retrieval
    results = similarity_search("What is the recall reason?", vs)
    for r in results:
        print(r.metadata)
        print(r.page_content[:200])
        print("---")