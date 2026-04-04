import os
import json
import time
import numpy as np
import faiss
import fitz          # PyMuPDF
import requests
from pathlib import Path
import requests

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
OLLAMA_BASE_URL   = "http://localhost:11434"
EMBED_MODEL       = "nomic-embed-text"
CHAT_MODEL        = "llama3.2"        # change to "mistral" if preferred
EMBED_DIM         = 768               # nomic-embed-text output dimension
CHUNK_SIZE        = 500               # characters
CHUNK_OVERLAP     = 50                # characters
TOP_K             = 5                 # chunks to retrieve
PAPERS_DIR        = "papers"          # folder with your PDFs
INDEX_DIR         = "rag_index"       # where we save FAISS index


def check_ollama_running():
    """Verify Ollama is up before we start."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        models = [m["name"] for m in r.json().get("models", [])]
        print(f"Ollama running. Available models: {models}")
        
        # Warn if required models are missing
        for required in [EMBED_MODEL, CHAT_MODEL]:
            if not any(required in m for m in models):
                print(f" Model '{required}' not found. Run: ollama pull {required}")
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Ollama is not running. Start it with: ollama serve"
        )
    
check_ollama_running()