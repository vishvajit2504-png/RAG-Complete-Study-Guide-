# main.py

import json
import os
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from config import get_settings

from vectorless_pipeline import (
    run_pipeline,
    index_document,
    get_cache_paths,
    cache_exists,
    load_cache
)
from reasoning_retriever import retrieve
from context_extractor import extract_context


# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Vectorless RAG API — Phase 3",
    description = "Reasoning-based document retrieval without vector embeddings",
    version     = "1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"]
)


# ── Request / Response models ─────────────────────────────────────────────────

class IndexRequest(BaseModel):
    pdf_path:      str          = Field(..., example="data/docs/sample.pdf")
    model:         str          = Field(get_settings().ollama_chat_model, example="")
    force_reindex: bool         = Field(False)
    cache_dir:     str          = Field("data/cache")


class QueryRequest(BaseModel):
    pdf_path:              str  = Field(..., example="data/docs/sample.pdf")
    query:                 str  = Field(..., example="What caused the recall?")
    model:                 str  = Field(get_settings().ollama_chat_model)
    cache_dir:             str  = Field("data/cache")
    max_nodes:             int  = Field(3, ge=1, le=10)
    max_chars_per_section: int  = Field(4000, ge=500, le=10000)
    max_total_tokens:      int  = Field(6000, ge=1000, le=20000)


class FullPipelineRequest(BaseModel):
    pdf_path:              str  = Field(..., example="data/docs/sample.pdf")
    query:                 str  = Field(..., example="What caused the recall?")
    model:                 str  = Field(get_settings().ollama_chat_model)
    cache_dir:             str  = Field("data/cache")
    max_nodes:             int  = Field(3, ge=1, le=10)
    force_reindex:         bool = Field(False)


class IndexResponse(BaseModel):
    status:      str
    pdf_path:    str
    total_nodes: int
    cache_paths: dict


class QueryResponse(BaseModel):
    query:    str
    answer:   str
    metadata: dict
    trace:    dict


class HealthResponse(BaseModel):
    status:  str
    version: str


# ── Shared state (in-memory cache of loaded trees) ────────────────────────────

# Avoids reloading JSON from disk on every query
# Key: pdf_path, Value: (tree, node_map) tuple
_tree_cache: dict = {}


def get_or_load_tree(pdf_path: str, cache_dir: str) -> tuple:
    """
    Returns tree + node_map from in-memory cache if available.
    Falls back to loading from disk cache.
    Raises HTTPException if neither exists.
    """
    if pdf_path in _tree_cache:
        return _tree_cache[pdf_path]

    cache_paths = get_cache_paths(pdf_path, cache_dir)

    if not cache_exists(cache_paths):
        raise HTTPException(
            status_code = 404,
            detail      = (
                f"Document '{pdf_path}' has not been indexed yet. "
                f"Call POST /index first."
            )
        )

    tree, node_map = load_cache(cache_paths)
    _tree_cache[pdf_path] = (tree, node_map)
    return tree, node_map


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Sanity check — confirms API is running."""
    return {"status": "ok", "version": "1.0.0"}


@app.post("/index", response_model=IndexResponse)
def index_pdf(request: IndexRequest):
    """
    Indexes a PDF document:
      - Builds hierarchical tree index
      - Summarizes all nodes with LLM
      - Caches results to disk

    Safe to call multiple times — uses cache unless force_reindex=True.
    """
    if not os.path.exists(request.pdf_path):
        raise HTTPException(
            status_code = 404,
            detail      = f"PDF not found: {request.pdf_path}"
        )

    cache_paths = get_cache_paths(request.pdf_path, request.cache_dir)

    # Use cache if available and not forced
    if not request.force_reindex and cache_exists(cache_paths):
        tree, node_map = load_cache(cache_paths)
        return IndexResponse(
            status      = "cached",
            pdf_path    = request.pdf_path,
            total_nodes = len(node_map),
            cache_paths = cache_paths
        )

    # Run full indexing
    try:
        tree, node_map = index_document(
            pdf_path    = request.pdf_path,
            cache_paths = cache_paths,
            model       = request.model
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Store in memory cache
    _tree_cache[request.pdf_path] = (tree, node_map)

    return IndexResponse(
        status      = "indexed",
        pdf_path    = request.pdf_path,
        total_nodes = len(node_map),
        cache_paths = cache_paths
    )


@app.post("/query", response_model=QueryResponse)
def query_document(request: QueryRequest):
    """
    Queries an already-indexed document.
    Requires POST /index to have been called first.

    Pipeline:
      Load cached tree → LLM reasons → extract context → generate answer
    """
    try:
        tree, node_map = get_or_load_tree(request.pdf_path, request.cache_dir)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        # Retrieval
        retrieval_result = retrieve(
            query     = request.query,
            tree      = tree,
            node_map  = node_map,
            model     = request.model,
            max_nodes = request.max_nodes
        )

        # Context extraction
        extraction_result = extract_context(
            retrieval_result,
            max_chars_per_section = request.max_chars_per_section,
            max_total_tokens      = request.max_total_tokens
        )

        # Generation
        from vectorless_pipeline import build_generation_prompt, call_ollama
        prompt = build_generation_prompt(
            query          = request.query,
            context_string = extraction_result["context_string"]
        )
        answer = call_ollama(prompt, model=request.model)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return QueryResponse(
        query    = request.query,
        answer   = answer,
        metadata = extraction_result["metadata"],
        trace    = extraction_result["trace"]
    )


@app.post("/query/full", response_model=QueryResponse)
def full_pipeline(request: FullPipelineRequest):
    """
    One-shot endpoint — indexes if needed, then queries.
    Best for first-time use or testing.
    Equivalent to calling /index then /query sequentially.
    """
    if not os.path.exists(request.pdf_path):
        raise HTTPException(
            status_code = 404,
            detail      = f"PDF not found: {request.pdf_path}"
        )

    try:
        result = run_pipeline(
            pdf_path      = request.pdf_path,
            query         = request.query,
            model         = request.model,
            cache_dir     = request.cache_dir,
            max_nodes     = request.max_nodes,
            force_reindex = request.force_reindex
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return QueryResponse(
        query    = result["query"],
        answer   = result["answer"],
        metadata = result["metadata"],
        trace    = result["trace"]
    )


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)