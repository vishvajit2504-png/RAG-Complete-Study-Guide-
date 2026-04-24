# RAG Code 4 — Advanced Retrieval Pipeline

Phase 3 of the RAG Study Guide. Implements a production-grade retrieval pipeline on top of four modules: semantic chunking, query transformation, parent-child retrieval with RRF fusion, and cross-encoder reranking.

---

## Architecture

```
papers/
  └── your PDFs / TXTs / DOCXs
        │
        ▼
┌─────────────────────────────┐
│  Module 1 — chunking.py     │  Semantic parent-child splitting
│                             │  Parents ~1500 tok | Children ~200 tok
└────────────┬────────────────┘
             │  parents, children
             ▼
┌─────────────────────────────┐
│  Module 3 — Retrieval.py    │  Children → ChromaDB (vector index)
│  (indexing half)            │  Parents  → parent_store.pkl
└────────────┬────────────────┘
             │
    ── query time ──
             │
             ▼
┌─────────────────────────────┐
│  Module 2 — query_transform │  Decomposition → HyDE → Multi-Query
│                .py          │  One query expands to N variants
└────────────┬────────────────┘
             │  list of query strings
             ▼
┌─────────────────────────────┐
│  Module 3 — Retrieval.py    │  Vector search per variant
│  (retrieval half)           │  RRF fusion → top parent IDs → ParentChunks
└────────────┬────────────────┘
             │  List[ParentChunk]  (~15 candidates)
             ▼
┌─────────────────────────────┐
│  Module 4 — reranker.py     │  Cross-encoder scores each (query, parent) pair
│                             │  Returns top-K by relevance
└─────────────────────────────┘
             │
             ▼
        Final top-K parent chunks
```

---

## Modules

### Module 1 — `chunking.py`

Converts raw documents into a two-level chunk hierarchy.

| Class | Role |
|---|---|
| `ChunkingConfig` | Tunable parameters (threshold, chunk sizes) |
| `ParentChunk` | Large context chunk (`id`, `content`, `metadata`) |
| `ChildChunk` | Small retrieval chunk (`content`, `parent_id`, `metadata`) |
| `SemanticParentChildChunker` | Orchestrates both split stages |

**Stage 1 — Semantic split (parents):** Uses `SemanticChunker` with sentence-embedding similarity to find topic boundaries. Conservative threshold (95th percentile) keeps chunks large and topically coherent.

**Stage 2 — Recursive split (children):** Each parent is split into ~200-token children with `RecursiveCharacterTextSplitter`. These are what gets embedded and searched.

**Why two sizes?** Bi-encoder retrieval works better on short, focused chunks. LLM answer generation needs wide context. Two sizes let you optimise both independently.

---

### Module 2 — `query_transform.py`

Expands a single user query into a richer set of query strings before retrieval.

| Class | Technique | What it does |
|---|---|---|
| `QueryDecomposer` | Decomposition | Detects multi-hop queries and splits them into atomic sub-queries |
| `HyDETransformer` | HyDE | Generates a hypothetical answer and uses that as a query |
| `multiQueryTransformer` | Multi-Query | Rewrites the query with varied vocabulary and angles |
| `CompositeQueryTransformer` | All three | Chains decomposition → HyDE + multi-query, dedupes results |

All transformers share the same `transform(query) -> List[str]` interface, so they are interchangeable.

---

### Module 3 — `Retrieval.py`

Handles both indexing and retrieval.

| Class / Function | Role |
|---|---|
| `VectorStoreManager` | Owns the ChromaDB vector store and the parent dict |
| `VectorStoreManager.index()` | One-time setup: embeds children, saves parents to `.pkl` |
| `VectorStoreManager.load_parent_store()` | Restores parents from disk on subsequent runs |
| `VectorStoreManager.get_parent(id)` | O(1) dict lookup by parent ID |
| `reciprocal_rank_fusion()` | Fuses N ranked lists into one score per document |
| `ParentChildRetriever` | For each query variant → search children → collect parent IDs → RRF → return `ParentChunk`s |

**Why RRF?** Each query variant returns a separate ranked list of parent IDs. A parent hit by multiple variants gets a higher fused score — a natural way to reward broad relevance.

---

### Module 4 — `reranker.py`

Reranks retrieval candidates with a cross-encoder.

| Class | Model | Role |
|---|---|---|
| `BGEReranker` | `BAAI/bge-reranker-base` | Scores each `(query, parent.content)` pair jointly |

**Why rerank after retrieval?** Bi-encoder retrieval is fast but lossy — query and document are encoded independently. A cross-encoder reads both together and produces a more accurate relevance score. The two-stage pattern (retrieve 15 → rerank to 5) keeps latency low while improving precision.

---

### `run_end_to_end.py` — Full Pipeline Runner

Wires all four modules together. Supports two modes:

- **Index mode** (`--reindex` or first run): loads documents, chunks, indexes into ChromaDB, saves `parent_store.pkl`
- **Query mode**: loads the saved index, transforms query, retrieves, reranks, prints results

---

## Setup

### 1. Install dependencies

```bash
pip install langchain langchain-community langchain-chroma langchain-ollama \
            langchain-experimental sentence-transformers chromadb \
            pypdf unstructured python-docx
```

### 2. Start Ollama

The pipeline uses two Ollama models:

```bash
ollama pull nomic-embed-text   # embeddings
ollama pull llama3.2:1b        # query transformations
```

Ollama must be running at `http://localhost:11434`.

### 3. Add your documents

Drop `.pdf`, `.txt`, or `.docx` files into the `papers/` folder:

```
RAG CODE 4/
  papers/
    attention_is_all_you_need.pdf
    bert_paper.pdf
    rag_paper.pdf
```

---

## Usage

```bash
# First run — indexes papers/ then answers the query
python run_end_to_end.py --query "What loss function does BERT use?" --reindex

# Subsequent runs — reuses saved index (faster)
python run_end_to_end.py --query "How does self-attention work?"

# Control number of final results
python run_end_to_end.py --query "Compare BERT and GPT" --top-k 3

# Interactive mode — prompts for a query
python run_end_to_end.py
```

### CLI flags

| Flag | Default | Description |
|---|---|---|
| `--query` / `-q` | (prompt) | Query string |
| `--reindex` | off | Force re-index even if `parent_store.pkl` exists |
| `--top-k` | 5 | Number of reranked results to return |

---

## File Structure

```
RAG CODE 4/
├── chunking.py           # Module 1 — semantic parent-child chunking
├── query_transform.py    # Module 2 — HyDE, multi-query, decomposition
├── Retrieval.py          # Module 3 — ChromaDB indexing + RRF retrieval
├── reranker.py           # Module 4 — cross-encoder reranking
├── run_end_to_end.py     # Full pipeline runner (CLI entry point)
├── papers/               # Drop your documents here
│   └── ...
└── chroma_db/            # Auto-created on first index run
    ├── ...               # ChromaDB collection files
    └── parent_store.pkl  # Serialised dict[str, ParentChunk]
```

---

## Config Reference

All tunable constants live at the top of `run_end_to_end.py`:

| Constant | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `nomic-embed-text:latest` | Ollama embedding model |
| `CHAT_MODEL` | `llama3.2:1b` | Ollama LLM for query transforms |
| `CHILDREN_PER_QUERY` | 20 | Children fetched per query variant |
| `RETRIEVAL_TOP_K` | 15 | Parents passed to the reranker |
| `RERANK_TOP_K` | 5 | Final results returned |

Chunking parameters live in `ChunkingConfig` inside `chunking.py`:

| Parameter | Default | Description |
|---|---|---|
| `breakpoint_threshold_amount` | 95.0 | Percentile for semantic split aggressiveness |
| `child_chunk_size` | 800 chars | Target child chunk size |
| `max_parent_chars` | 6000 chars | Hard cap on parent chunk size |
