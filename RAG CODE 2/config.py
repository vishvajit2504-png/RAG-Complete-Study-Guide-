# config.py
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):

    # --- Ollama ---
    ollama_base_url: str = "http://localhost:11434"
    ollama_embedding_model: str = "nomic-embed-text"   # best local embedding model
    ollama_chat_model: str = "llama3.2:1b"                # or mistral, phi3, gemma2

    # --- ChromaDB ---
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "rag_documents"

    # --- Chunking ---
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # --- Retrieval ---
    retrieval_top_k: int = 5
    similarity_threshold: float = 0.60

    # --- FastAPI ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()