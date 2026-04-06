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
