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
