"""
run_end_to_end.py
-----------------
Phase 3 — Full Pipeline Runner

Wires all four modules together in sequence:

  [Module 1] Load docs → semantic parent-child chunking
       ↓
  [Module 3] Index children → ChromaDB | parents → parent_store.pkl
       ↓
  [Module 2] Query → HyDE + Multi-Query + Decomposition → list of queries
       ↓
  [Module 3] Multi-query retrieval → RRF fusion → top parent chunks
       ↓
  [Module 4] Cross-encoder reranking → final top-K parents

Usage:
  # Index + query (first run or forced re-index):
  python run_end_to_end.py --query "What loss function does BERT use?" --reindex

  # Query only (index already built):
  python run_end_to_end.py --query "How does self-attention work?"

  # Interactive mode (no query flag → prompts you):
  python run_end_to_end.py
"""

import argparse
import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings

# ── local modules ──────────────────────────────────────────────────────────────
from chunking import SemanticParentChildChunker
from query_transform import (
    CompositeQueryTransformer,
    HyDETransformer,
    QueryDecomposer,
    multiQueryTransformer,
)
from Retrieval import ParentChildRetriever, VectorStoreManager
from reranker import BGEReranker

# ── config ─────────────────────────────────────────────────────────────────────
PAPERS_DIR       = Path(__file__).parent / "papers"
PERSIST_DIR      = str(Path(__file__).parent / "chroma_db")
COLLECTION_NAME  = "rag_documents"

OLLAMA_BASE_URL       = "http://localhost:11434"
EMBEDDING_MODEL       = "nomic-embed-text:latest"
CHAT_MODEL            = "llama3.2:1b"

CHILDREN_PER_QUERY    = 20   # how many child chunks to fetch per query variant
RETRIEVAL_TOP_K       = 15   # parent chunks to pass into the reranker
RERANK_TOP_K          = 5    # final parents returned to the user

LOADER_MAP = {
    ".pdf":  PyPDFLoader,
    ".txt":  TextLoader,
    ".docx": UnstructuredWordDocumentLoader,
}


# ══════════════════════════════════════════════════════════════════════════════
# Stage helpers
# ══════════════════════════════════════════════════════════════════════════════

def build_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model=EMBEDDING_MODEL)


def build_llm() -> ChatOllama:
    return ChatOllama(model=CHAT_MODEL, temperature=0.2)


# ── Stage 1: Load documents ────────────────────────────────────────────────────

def load_documents() -> list[Document]:
    if not PAPERS_DIR.exists():
        raise FileNotFoundError(
            f"Papers folder not found: {PAPERS_DIR}\n"
            "Create the folder and drop your PDFs / TXTs / DOCXs inside."
        )

    files = [f for f in PAPERS_DIR.rglob("*")
             if f.is_file() and f.suffix.lower() in LOADER_MAP]

    if not files:
        raise ValueError(f"No supported files found in {PAPERS_DIR}. "
                         "Supported: .pdf, .txt, .docx")

    all_docs: list[Document] = []
    for file_path in files:
        suffix = file_path.suffix.lower()
        try:
            loader = LOADER_MAP[suffix](str(file_path))
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"]    = file_path.name
                doc.metadata["file_type"] = suffix
            print(f"  Loaded {len(docs):>3} page(s)  ← {file_path.name}")
            all_docs.extend(docs)
        except Exception as exc:
            print(f"  [WARN] Failed to load {file_path.name}: {exc}")

    print(f"  Total pages loaded: {len(all_docs)}")
    return all_docs


# ── Stage 2: Chunk ─────────────────────────────────────────────────────────────

def chunk_documents(docs: list[Document], embeddings: OllamaEmbeddings):
    chunker = SemanticParentChildChunker(embeddings=embeddings)
    parents, children = chunker.chunk_documents(docs)
    print(f"  Parents: {len(parents)}   Children: {len(children)}")
    return parents, children


# ── Stage 3: Index ─────────────────────────────────────────────────────────────

def index(parents, children, embeddings: OllamaEmbeddings) -> VectorStoreManager:
    vsm = VectorStoreManager(embeddings=embeddings, collection_name=COLLECTION_NAME)
    vsm.index(parents, children, persist_dir=PERSIST_DIR)
    return vsm


# ── Stage 3b: Load existing index ─────────────────────────────────────────────

def load_index(embeddings: OllamaEmbeddings) -> VectorStoreManager:
    vsm = VectorStoreManager(embeddings=embeddings, collection_name=COLLECTION_NAME)
    vsm.vector_store = Chroma(
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )
    vsm.load_parent_store(PERSIST_DIR)
    return vsm


# ── Stage 4: Transform query ───────────────────────────────────────────────────

def transform_query(query: str, llm: ChatOllama) -> list[str]:
    transformer = CompositeQueryTransformer(
        decomposer=QueryDecomposer(llm),
        hyde=HyDETransformer(llm),
        multi_query=multiQueryTransformer(llm),
    )
    queries = transformer.transform(query)
    print(f"  Expanded to {len(queries)} query variant(s):")
    for i, q in enumerate(queries, 1):
        preview = q[:100] + ("..." if len(q) > 100 else "")
        print(f"    [{i}] {preview}")
    return queries


# ── Stage 5: Retrieve ──────────────────────────────────────────────────────────

def retrieve(queries: list[str], vsm: VectorStoreManager):
    retriever = ParentChildRetriever(
        vsm,
        children_per_query=CHILDREN_PER_QUERY,
        final_top_k=RETRIEVAL_TOP_K,
    )
    parents = retriever.retrieve(queries)
    print(f"  Retrieved {len(parents)} parent chunk(s) after RRF fusion")
    return parents


# ── Stage 6: Rerank ────────────────────────────────────────────────────────────

def rerank(query: str, parents, top_k: int = RERANK_TOP_K):
    reranker = BGEReranker()
    top = reranker.rerank(query, parents, top_k=top_k)
    print(f"  Reranked → kept top {len(top)}")
    return top


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(query: str, force_reindex: bool = False) -> None:
    print("\n" + "═" * 70)
    print(f"  QUERY: {query}")
    print("═" * 70)

    embeddings = build_embeddings()
    llm        = build_llm()

    parent_store_path = Path(PERSIST_DIR) / "parent_store.pkl"
    needs_index = force_reindex or not parent_store_path.exists()

    # ── Index ──────────────────────────────────────────────────────────────────
    if needs_index:
        print("\n[1/2] Indexing documents ...")
        docs            = load_documents()
        parents, children = chunk_documents(docs, embeddings)
        vsm             = index(parents, children, embeddings)
    else:
        print("\n[1/2] Loading existing index ...")

        persist_dir = str(Path(__file__).parent / "chroma_db")

        vsm = load_index(embeddings)

    # ── Query pipeline ─────────────────────────────────────────────────────────
    print("\n[2/5] Transforming query ...")
    queries = transform_query(query, llm)

    print("\n[3/5] Retrieving parent chunks ...")
    candidates = retrieve(queries, vsm)

    if not candidates:
        print("\n  No results found. Try a different query or re-index with --reindex.")
        return

    print("\n[4/5] Reranking ...")
    top_parents = rerank(query, candidates)

    # ── Results ────────────────────────────────────────────────────────────────
    print("\n[5/5] Results")
    print("═" * 70)
    for i, p in enumerate(top_parents, 1):
        score = getattr(p, "rerank_score", None)
        score_str = f"  score={score:.4f}" if score is not None else ""
        print(f"\n[{i}]{score_str}")
        print(f"  ID      : {p.id}")
        print(f"  Source  : {p.metadata.get('source', 'N/A')}")
        print(f"  Content :\n{p.content.strip()}")
        print("-" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RAG Code 4 — end-to-end pipeline runner"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Query string. If omitted, you will be prompted interactively.",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force re-indexing even if a saved index already exists.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=RERANK_TOP_K,
        help=f"Number of final results to return (default: {RERANK_TOP_K}).",
    )
    args = parser.parse_args()

    query = args.query
    if not query:
        try:
            query = input("Enter your query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            sys.exit(0)

    if not query:
        print("No query provided. Exiting.")
        sys.exit(1)

    RERANK_TOP_K = args.top_k
    run_pipeline(query, force_reindex=args.reindex)
