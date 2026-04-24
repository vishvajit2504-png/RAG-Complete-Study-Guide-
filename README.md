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

---

## RAG CODE 3 — Vectorless RAG (LLM-Navigated Tree Index)

`RAG CODE 3/` is a **vectorless** retrieval system. Instead of embedding documents and doing similarity search, it builds a hierarchical tree index from the document structure, summarizes every node with an LLM, and then uses a second LLM call to *reason* over the tree and pick the most relevant sections.

### How it works

```
PDF
  └─► text_indexer.py       Parses document structure → nested tree of sections
        └─► node_summarizer.py    LLM summarizes every node (bottom-up)
              └─► reasoning_retriever.py  LLM navigates compact tree → selects node IDs
                    └─► context_extractor.py   Fetches raw text for selected nodes
                          └─► vectorless_pipeline.py  Builds answer prompt → calls LLM
```

### Files

| File | Role |
|---|---|
| `config.py` | Pydantic settings (Ollama URL, model names, chunking params) |
| `document_loader.py` | Loads `.pdf`, `.txt`, `.docx` files with source metadata |
| `text_indexer.py` | Builds a nested tree (sections/subsections) from raw text |
| `node_summarizer.py` | Recursively summarizes all tree nodes; builds flat `node_map` for O(1) lookup |
| `reasoning_retriever.py` | Stage 1: LLM reads compact tree → picks node IDs. Stage 2: fetches raw text |
| `context_extractor.py` | Assembles the retrieved sections into a token-budgeted context string |
| `vectorless_pipeline.py` | End-to-end orchestrator + disk caching + generation prompt |
| `main.py` | FastAPI service (`/health`, `/index`, `/query`, `/query/full`) |
| `test_main.py` | End-to-end test harness |

### Key design points

- **No vector embeddings at query time** — retrieval is done by LLM reasoning over section summaries, not cosine similarity.
- **Hierarchical tree index** — preserves document structure (chapters → sections → subsections). The LLM navigates the outline, not a flat list of chunks.
- **Full explainability** — every retrieval produces a structured audit trail: which node IDs were selected, why, and which pages they cover.
- **Disk cache** — tree and node map are saved as JSON after first indexing run. Subsequent queries skip re-indexing.
- **Two-stage LLM calls**: Stage 1 uses `temperature=0` (deterministic navigation), Stage 2 uses `temperature=0.2` (natural answer generation).

### Usage

```bash
# Start the API
uvicorn main:app --reload --port 8001

# One-shot endpoint (indexes + queries in a single call)
curl -X POST http://localhost:8001/query/full \
  -H "Content-Type: application/json" \
  -d '{"pdf_path": "data/docs/paper.pdf", "query": "What is the main contribution?"}'

# Or run the pipeline directly
python vectorless_pipeline.py
```

### When to use RAG CODE 3 vs RAG CODE 2

| | RAG CODE 2 (Vector RAG) | RAG CODE 3 (Vectorless) |
|---|---|---|
| Retrieval mechanism | Cosine similarity on embeddings | LLM reasoning over section summaries |
| Speed | Fast (vector lookup) | Slower (two LLM calls per query) |
| Explainability | Low (score is a float) | High (named sections + page ranges) |
| Best for | Large corpora, many short chunks | Structured documents (reports, papers) |

---

## RAG CODE 4 — Advanced Retrieval Pipeline (Semantic Chunking + Reranking)

`RAG CODE 4/` is a **four-module production retrieval pipeline** built on ChromaDB and a cross-encoder reranker. It improves on RAG CODE 2 by replacing fixed-size chunking with semantic parent-child chunking, expanding queries before retrieval, fusing multi-query results with RRF, and reranking candidates with a cross-encoder.

### How it works

```
papers/
  └─► chunking.py (Module 1)
        Semantic split → Parents (~1500 tok)
        Recursive split → Children (~200 tok)
              └─► Retrieval.py (Module 3 — indexing)
                    Children → ChromaDB vector index
                    Parents  → parent_store.pkl
                          └─► query_transform.py (Module 2)
                                QueryDecomposer  → splits multi-hop queries
                                HyDETransformer  → generates hypothetical answer
                                multiQueryTransformer → rewrites with varied phrasing
                                      └─► Retrieval.py (Module 3 — retrieval)
                                            Per-variant vector search
                                            RRF fusion → top parent chunks
                                                  └─► reranker.py (Module 4)
                                                        Cross-encoder scores (query, parent)
                                                        → Final top-K results
```

### Files

| File | Role |
|---|---|
| `chunking.py` | Module 1 — semantic parent-child chunking; `ParentChunk` and `ChildChunk` dataclasses |
| `query_transform.py` | Module 2 — `HyDETransformer`, `multiQueryTransformer`, `QueryDecomposer`, `CompositeQueryTransformer` |
| `Retrieval.py` | Module 3 — `VectorStoreManager` (ChromaDB + pkl), `reciprocal_rank_fusion`, `ParentChildRetriever` |
| `reranker.py` | Module 4 — `BGEReranker` using `BAAI/bge-reranker-base` cross-encoder |
| `run_end_to_end.py` | CLI runner — wires all four modules; supports `--reindex`, `--query`, `--top-k` |
| `README.md` | Detailed module-level documentation |

### Key design points

- **Two-level chunks** — small children (~200 tokens) for precise retrieval; large parents (~1500 tokens) for rich LLM context. Children point back to their parent via `parent_id`.
- **Semantic boundaries** — `SemanticChunker` uses sentence-embedding similarity to split on topic shifts instead of arbitrary character counts.
- **Query expansion** — one user query becomes N variants (decomposed sub-queries + HyDE hypothetical + multi-query rewrites), maximising the chance of matching relevant chunks.
- **RRF fusion** — combines the N ranked child-retrieval lists into one parent ranking. Parents hit by multiple query variants score higher.
- **Cross-encoder reranking** — `BAAI/bge-reranker-base` reads `(query, parent.content)` jointly (not independently) and produces a more accurate relevance score than bi-encoder similarity.

### Usage

```bash
# First run — indexes papers/ then answers the query
python "RAG CODE 4/run_end_to_end.py" --query "What loss function does BERT use?" --reindex

# Subsequent runs — reuses saved index
python "RAG CODE 4/run_end_to_end.py" --query "How does self-attention work?"

# Control final result count
python "RAG CODE 4/run_end_to_end.py" --query "Compare BERT and GPT" --top-k 3
```

### How RAG CODE 4 improves on earlier phases

| Concern | RAG CODE 1 / 2 | RAG CODE 4 |
|---|---|---|
| Chunking | Fixed character size | Semantic topic boundaries |
| Chunk granularity | Single size | Two sizes (parent context + child retrieval) |
| Query coverage | Single query | Decomposition + HyDE + multi-query |
| Result fusion | First-hit wins | Reciprocal Rank Fusion across variants |
| Ranking quality | Bi-encoder similarity | Cross-encoder reranking |

---

## Repository overview

| Folder | Phase | Core technique |
|---|---|---|
| `RAG CODE 1/` | Exploration | Embeddings, FAISS, fixed chunking |
| `RAG CODE 2/` | Production service | ChromaDB, FastAPI, MMR retrieval |
| `RAG CODE 3/` | Vectorless RAG | LLM-navigated tree index, no embeddings |
| `RAG CODE 4/` | Advanced retrieval | Semantic chunking, RRF, cross-encoder reranking |
