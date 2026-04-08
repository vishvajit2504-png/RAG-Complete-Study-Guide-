# main.py
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import shutil
import os

from config import get_settings
from document_loader import load_documents_from_folder
from text_splitter import split_documents
from embedder import embed_and_store, get_vector_store, get_embeddings
from rag_chain import query_with_sources

settings = get_settings()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Startup / Shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load vectorstore once at startup — not on every request."""
    logger.info("Starting up — loading ChromaDB...")
    app.state.vectorstore = get_vector_store(get_embeddings())
    logger.info("ChromaDB loaded. API ready.")
    yield
    logger.info("Shutting down.")

app = FastAPI(
    title="Multi-Doc RAG API",
    description="Local RAG pipeline — Ollama + ChromaDB + LangChain",
    version="1.0.0",
    lifespan=lifespan
)

# ── Request / Response Schemas ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    mode: str = "mmr"          # "mmr" or "similarity"

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    chunks_used: int

class IngestResponse(BaseModel):
    message: str
    documents_loaded: int
    chunks_created: int

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Quick liveness check."""
    return {"status": "ok", "model": settings.ollama_chat_model}


@app.post("/ingest", response_model=IngestResponse)
def ingest_documents(folder_path: str = "./sample_docs"):
    """
    Load, chunk, embed, and store all documents from a folder.
    Call this once before querying.
    """
    try:
        logger.info(f"Ingestion started from: {folder_path}")

        docs   = load_documents_from_folder(folder_path)
        chunks = split_documents(docs)
        embed_and_store(chunks)

        return IngestResponse(
            message="Ingestion complete.",
            documents_loaded=len(docs),
            chunks_created=len(chunks)
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Ask a question — returns answer + sources + chunk count.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        logger.info(f"Query received: '{request.question}' mode={request.mode}")
        result = query_with_sources(request.question, request.mode)

        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            chunks_used=result["chunks_used"]
        )

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a single document directly via API.
    Saves to ./sample_docs/ then triggers ingestion.
    """
    allowed = {".pdf", ".txt", ".docx"}
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {allowed}"
        )

    save_path = f"./sample_docs/{file.filename}"
    os.makedirs("./sample_docs", exist_ok=True)

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info(f"Uploaded: {file.filename}")

    # Auto-ingest the uploaded file
    docs   = load_documents_from_folder("./sample_docs")
    chunks = split_documents(docs)
    embed_and_store(chunks)

    return {"message": f"'{file.filename}' uploaded and ingested.", "chunks_created": len(chunks)}


@app.delete("/reset")
def reset_vectorstore():
    """
    Delete ChromaDB collection — forces full re-ingestion next time.
    Use when switching documents or embedding models.
    """
    try:
        vs = get_vector_store(get_embeddings())
        vs.delete_collection()
        logger.info("ChromaDB collection deleted.")
        return {"message": "Vector store reset. Re-ingest documents before querying."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.api_host, port=settings.api_port, reload=True)