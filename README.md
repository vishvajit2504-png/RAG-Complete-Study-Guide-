# RAG Complete Study Guide

A concise, hands-on collection of examples and notes for building Retrieval-Augmented Generation (RAG) systems.

This repository gathers sample code, document loaders, and text-splitting utilities to help you learn and prototype RAG pipelines.

## Contents

- `RAG CODE 1/` — Example RAG notebook and index files (includes `Rag-file.py`, `rag_index/`, and `papers/`).
- `RAG CODE 2/` — Production-oriented helpers (`document_loader.py`, `text_splitter.py`, `config.py`) and notes.
- `requirements.txt` — Python dependencies used across examples.

## Quick start

1. Create and activate a Python virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Open the example code in `RAG CODE 1/` or the production helpers in `RAG CODE 2/` and follow the README inside each folder.

## Notes

- Use `RAG CODE 1/` for exploratory experiments and `RAG CODE 2/` for a more structured implementation.
- See `RAG CODE 2/README.md` for details on the document loader and text splitter implementations.

---

## Consolidated summary (RAG CODE 1 + RAG CODE 2)

This repository contains two complementary sets of examples:

- `RAG CODE 1/` — Exploration and teaching-focused scripts:
	- `Rag-file.py`: hands-on example demonstrating embeddings, chunking, indexing, and retrieval.
	- `rag_index/`: pre-built index files (FAISS) and a `chunks.json` sample used for quick experiments.
	- Focuses on: embedding calls to a local Ollama endpoint, batch embedding normalization (unit-length vectors for cosine similarity), PDF loading, fixed chunk size (~500) with overlap, and building/searching indexes.

- `RAG CODE 2/` — Production-oriented helpers and a small FastAPI service:
	- `document_loader.py`: robust loaders by extension (`.pdf`, `.txt`, `.docx`) that attach `source` metadata.
	- `text_splitter.py`: `RecursiveCharacterTextSplitter` usage with configurable `chunk_size` and `chunk_overlap`, attaches `chunk_index` to chunks.
	- `embedder.py`: `OllamaEmbeddings` + `Chroma` vector store helpers; `embed_and_store` avoids re-embedding when vectors exist.
	- `retriever.py`: scored retrieval helpers, `retrieve_with_scores` logs raw scores, and `filter_by_score` now handles both similarity (higher=better) and distance (lower=better) semantics.
	- `rag_chain.py`: builds the RAG chain and exposes `query_with_sources(question, mode)` which retrieves, formats context, and invokes the LLM. The API forwards `mode` (mmr/similarity) to this function.
	- `main.py`: FastAPI app with endpoints: `/health`, `/ingest`, `/query`, `/upload`, `/reset`. Loads the Chroma store at startup.
	- `test_api.py`: lightweight end-to-end test harness that exercises ingest, query, upload, and edge cases.

Why both folders exist:
- `RAG CODE 1` is quick to iterate and explore different embedding/indexing ideas; useful when learning or prototyping.
- `RAG CODE 2` collects the patterns and hardening needed for a small local service (consistent metadata, persistence, server endpoints, and improved retrieval filtering).

## Recommended next steps

- Run the `RAG CODE 2` service and tests to validate end-to-end behavior (see `RAG CODE 2/README.md` for commands).
- If you want retrieval mode switching to be active, I can wire `rag_chain.query_with_sources` to call `retriever.get_retriever(mode)` so `mode` controls MMR vs similarity retrieval directly.



