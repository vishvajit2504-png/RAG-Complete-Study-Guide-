# RAG Code 2

Notes for building a production Retrieval-Augmented Generation (RAG) pipeline: document loading and text splitting.

---

## Document Loader (`document_loader.py`)

Provides a production-ready approach to load documents dynamically by file extension.

### Loader mapping

```python
LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": UnstructuredWordDocumentLoader,
}
```

Use the mapping to choose the appropriate loader for each source file. Each loader should return text and metadata used downstream.

---

## Text Splitter (`text_splitter.py`)

Splits documents into smaller, context-preserving chunks for indexing and retrieval.

- Uses `RecursiveCharacterTextSplitter` with hierarchical separators (`\n\n`, `\n`, `.`, ` `).
- Configurable `chunk_size` and `chunk_overlap` for balancing chunk granularity and context.
- Preserves original document metadata on each chunk and adds a `chunk_index` for traceability.
- Includes a utility to report chunk statistics (min, max, average sizes and chunks per source).

Example configuration:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " "],
)
```

This strategy improves semantic chunking and retrieval accuracy while making debugging easier.

---

For implementation details, see `document_loader.py` and `text_splitter.py` in this folder.

---

## Embedder (`embedder.py`)

Provides embedding and vector-store integration for indexing and retrieval.

- Key functions:
    - `get_embeddings()` — returns an `OllamaEmbeddings` instance using settings from `config`.
    - `get_vector_store(embeddings)` — returns a `Chroma` collection configured with the embedding function and persistence directory.
    - `embed_and_store(chunks)` — embeds a list of `Document` chunks and persists them to Chroma; skips re-embedding if vectors already exist.
    - `similarity_search(query, vector_store, top_k)` — retrieves the top-k most similar documents for a query.

- Behavior and notes:
    - Uses settings from `config.py` (e.g. `ollama_base_url`, `ollama_embedding_model`, `chroma_collection_name`, `chroma_persist_dir`, `retrieval_top_k`).
    - When run as a script (`__main__`), it loads documents from the `papers` folder, splits them, embeds them, and runs a small retrieval test.

- Run example (from repo root):

```powershell
python "RAG CODE 2\embedder.py"
```

See `embedder.py` for implementation details and for adjusting settings in `config.py`.

---

## Recent Changes (2026-04-08)

- `retriever.py`: Improved `filter_by_score` to handle both distance-style and similarity-style scores. The filter now auto-detects score semantics and applies the correct threshold comparison to avoid dropping relevant chunks when underlying vector store returns distances.
- `main.py`: The `/query` endpoint now passes the client's `mode` parameter through to the RAG layer so callers can request `mmr` or `similarity` retrieval modes.
- `rag_chain.py`: `query_with_sources` now accepts a `mode` parameter and logs retrieval mode for debugging. Retrieval behavior is still driven by `retriever.py`.

These changes were made to address inconsistent query results where relevant context was incorrectly filtered out.

---

## How to apply changes and run tests

- Restart the FastAPI server so the updated code is loaded:

```powershell
# from repository root
uvicorn "RAG CODE 2.main:app" --reload --host 0.0.0.0 --port 8000
```

- Re-run the end-to-end test script:

```powershell
python "RAG CODE 2\test_api.py"
```

If you still see fallback answers like "I don't have enough information to answer that", check the server logs for retrieval scores and thresholds; the logs now include score info and the chosen retrieval mode to help debug further.

---

## Concise File Map

- `main.py`: FastAPI app exposing `/health`, `/ingest`, `/query`, `/upload`, and `/reset`. Starts the app with a lifespan hook that loads the Chroma vectorstore once.
- `config.py`: Central settings for embedding model, Chroma paths, chunking, retrieval parameters (`retrieval_top_k`, `similarity_threshold`), and API host/port.
- `document_loader.py`: Loaders for PDFs, DOCX, and text files. Returns a list of documents with `metadata` (including `source`).
- `text_splitter.py`: Splits documents into `Document` chunks and attaches `chunk_index` metadata. Controlled by `chunk_size` and `chunk_overlap` in `config.py`.
- `embedder.py`: Creates Ollama embeddings, constructs the Chroma store, and provides `embed_and_store` and `similarity_search` helpers.
- `retriever.py`: Exposes retriever helpers, scored retrieval (`retrieve_with_scores`), and `filter_by_score` which now copes with both similarity and distance score semantics.
- `rag_chain.py`: Builds/invokes the RAG chain and provides `query_with_sources(question, mode)` which retrieves, filters, formats context, and calls the LLM.
- `llm.py`: Wraps the chat/LLM component and prepares the RAG prompt template for answer generation.
- `test_api.py`: Simple end-to-end test runner that hits the API endpoints and exercises ingest/query/upload flows.

## How Retrieval & Filtering Works (short)

- The code retrieves scored chunks using `vectorstore.similarity_search_with_score` (see `retriever.py`). Different vector stores or client libraries may return either similarity scores (higher = better) or distance scores (lower = better).
- `filter_by_score` now inspects the returned scores and uses a heuristic to determine whether scores are distances or similarities, then applies the configured `similarity_threshold` appropriately. This prevents relevant chunks being discarded when the vector store uses distances.

## API usage examples

- Query (POST `/query`) — choose retrieval `mode` either `mmr` or `similarity`.

Python example:

```python
import requests

BASE_URL = "http://localhost:8000"
payload = {"question": "Summarize the key findings.", "mode": "mmr"}
resp = requests.post(f"{BASE_URL}/query", json=payload)
print(resp.json())
```

curl example:

```bash
curl -X POST "http://localhost:8000/query" -H "Content-Type: application/json" \
    -d '{"question":"What is Attention is all you need about?","mode":"mmr"}'
```

Notes:
- The `mode` parameter is forwarded to `query_with_sources` for logging and future behavior. `retriever.get_retriever(mode)` is available and can be wired to actively switch between MMR and pure similarity retrieval (see `retriever.get_retriever`).

## Troubleshooting and tips

- If you get the fallback answer "I don't have enough information to answer that":
    - Restart the server to ensure code changes are loaded.
    - Check server logs for the retrieval scores printed by `retriever.retrieve_with_scores` — they show raw scores and `source`/`chunk_index`. This helps determine whether the store returned distances or similarities.
    - If the scores are near zero for relevant chunks, your vector store is probably returning distances; the `filter_by_score` heuristic handles this, but you can also tune `similarity_threshold` in `config.py`.

- To force MMR vs similarity retrieval inside the RAG layer, update `rag_chain.query_with_sources` to call `get_retriever(mode)` and use the retriever directly (this repo logs the mode and currently uses scored retrieval + filtering).

## Run & Test (quick)

1. Start the API (from repo root):

```powershell
uvicorn "RAG CODE 2.main:app" 
```

2. In a second shell, run the test harness:

```powershell
python "RAG CODE 2\test_api.py"
```

3. If you change embeddings or want to re-index, delete `./chroma_db` and re-run `/ingest` or restart the server and call the `/ingest` route.


