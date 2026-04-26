# RAG Code 5 — Corrective RAG (CRAG) with LangGraph

An implementation of **Corrective RAG (CRAG)** using LangGraph, Ollama, and ChromaDB. The system grades retrieved documents for relevance, rewrites queries when needed, and falls back to a Tavily web search — all orchestrated as a stateful graph.

---

## How It Works

```
retrieve → grade_documents → [decision]
                                 ├── All relevant      → generate → END
                                 ├── Mixed relevance   → rewrite_query → retrieve (loop)
                                 └── All irrelevant    → web_search → generate → END
```

1. **Retrieve** — fetches top-k chunks from ChromaDB using semantic similarity
2. **Grade Documents** — LLM scores each document as relevant (`yes`) or not (`no`)
3. **Decide** — routes based on grading results (generate / rewrite / web search)
4. **Rewrite Query** — rewrites the question for better vector retrieval, then loops back
5. **Web Search** — Tavily fallback when local docs are insufficient
6. **Generate** — final answer from filtered context using the RAG chain

---

## Project Structure

```
RAG CODE 5/
├── main.py          # Entry point — runs the CRAG graph
├── graph.py         # LangGraph workflow definition and conditional routing
├── nodes.py         # Node functions: retrieve, grade, rewrite, web_search, generate
├── chains.py        # LLM chains: relevance grader, query rewriter, RAG generator
├── retriever.py     # ChromaDB vector store loader (auto-indexes if empty)
├── state.py         # GraphState TypedDict shared across all nodes
└── papers/          # Source PDF documents for indexing
```

---

## Setup

### Prerequisites
- [Ollama](https://ollama.com) running locally on `http://localhost:11434`
- Tavily API key (for web search fallback)

### Pull required models
```bash
ollama pull llama3.2:1b
ollama pull nomic-embed-text
```

### Install dependencies
```bash
pip install langchain langchain-community langchain-ollama langchain-text-splitters
pip install langgraph chromadb tavily-python python-dotenv pypdf
```

### Environment variables
Create a `.env` file in the project root:
```
TAVILY_API_KEY=your_tavily_api_key_here
```

---

## Running

```bash
# From the RAG-Complete-Study-Guide- root directory
python "RAG CODE 5/main.py"
```

On first run, if the ChromaDB collection is empty it will automatically load and index all PDFs from `RAG CODE 4/papers/`.

---

## Configuration

| Variable | Location | Default | Description |
|---|---|---|---|
| `OLLAMA_BASE_URL` | `retriever.py` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_EMBEDDING_MODEL` | `retriever.py` | `nomic-embed-text` | Embedding model |
| `CHROMA_PERSIST_DIR` | `retriever.py` | `RAG CODE 4/chroma_db` | ChromaDB storage path |
| `DOCS_PATH` | `retriever.py` | `RAG CODE 4/papers` | PDF source directory |
| LLM model | `chains.py` | `llama3.2:1b` | Chat model for grading/generation |
| `k` | `retriever.py` | `4` | Number of chunks to retrieve |

---

## Key Concepts

- **CRAG (Corrective RAG)** — extends basic RAG by evaluating retrieval quality and correcting it before generation
- **LangGraph** — manages the stateful, cyclical workflow (the rewrite → retrieve loop)
- **Structured Output** — the relevance grader uses `pydantic` + `with_structured_output` for reliable `yes`/`no` scoring
- **Tavily** — web search API used as a fallback when local knowledge is insufficient
