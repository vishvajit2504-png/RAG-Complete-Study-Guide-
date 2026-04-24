"""
retrieval.py
------------
Phase 3 — Module 3: Retrieval Orchestration + RRF

Three main responsibilities:
  1. VectorStoreManager — index children into ChromaDB, store parents in a dict
  2. reciprocal_rank_fusion — combine N ranked lists into one
  3. ParentChildRetriever — for each query, search children → fetch parents → fuse

Input from Module 2: list of query strings
Output to Module 4:  list of parent chunks, ranked by relevance
"""

import pickle
from typing import List
from collections import defaultdict

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)
from pathlib import Path
from typing import List

# Import the chunk dataclasses we defined in Module 1
from chunking import ParentChunk, ChildChunk


# ========================================================================
# 1. VectorStoreManager — handles indexing (one-time setup)
# ========================================================================

class VectorStoreManager:
    """Stores children in Chroma (for vector search) and parents in a dict
    (for lookup after retrieval)."""

    def __init__(self, embeddings: OllamaEmbeddings, collection_name: str = "arxiv_papers"):
        # Save embeddings model — used both for indexing and for querying.
        # Must be the SAME model for both or retrieval quality degrades.
        self.embeddings = embeddings
        self.collection_name = collection_name

        # Chroma vector store, empty for now. Created when we call .index().
        self.vector_store = None

        # Parent store: maps parent_id → ParentChunk.
        # In-memory dict is fine for Phase 3. In production this would be Redis or a DB.
        self.parent_store: dict[str, ParentChunk] = {}

    def index(self, parents: List[ParentChunk], children: List[ChildChunk],
              persist_dir: str | None = None) -> None:
        """One-time indexing step. Call after running SemanticParentChildChunker.

        Always wipes the existing ChromaDB collection first so ChromaDB and
        parent_store.pkl are guaranteed to be in sync after this call.
        """
        import chromadb as _chromadb

        if persist_dir is None:
            persist_dir = str(Path(__file__).parent / "chroma_db")

        # Step 1: Wipe the existing collection so stale children from prior runs
        # don't accumulate. Without this, Chroma.from_documents() appends, which
        # causes parent IDs in ChromaDB to outnumber those in the pkl → KeyError.
        _client = _chromadb.PersistentClient(path=persist_dir)
        if self.collection_name in [c.name for c in _client.list_collections()]:
            _client.delete_collection(self.collection_name)
            print(f"Cleared existing collection '{self.collection_name}'.")

        # Step 2: Store all parents in the dict, keyed by their id.
        self.parent_store = {}
        for parent in parents:
            self.parent_store[parent.id] = parent

        # Step 3: Convert children to LangChain Document format.
        child_docs = [
            Document(
                page_content=child.content,
                metadata={"parent_id": child.parent_id, **child.metadata}
            )
            for child in children
        ]

        # Step 4: Embed all children and store in the fresh collection.
        self.vector_store = Chroma.from_documents(
            documents=child_docs,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=persist_dir,
        )

        # Step 5: Persist the parent_store so it survives restarts.
        self.save_parent_store(persist_dir)

        print(f"Indexed {len(parents)} parents and {len(children)} children.")

    def save_parent_store(self, persist_dir: str) -> None:
        """Write parent_store to disk as a pickle file."""
        path = Path(persist_dir) / "parent_store.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.parent_store, f)
        print(f"Saved {len(self.parent_store)} parents → {path}")

    def load_parent_store(self, persist_dir: str) -> None:
        """Restore parent_store from disk after loading an existing ChromaDB collection.

        Must be called whenever you attach to an existing ChromaDB on disk instead of
        calling index() — otherwise get_parent() will raise KeyError for every lookup.
        """
        path = Path(persist_dir) / "parent_store.pkl"
        if not path.exists():
            raise FileNotFoundError(
                f"No parent_store.pkl at {path}. "
                "Run the indexing step first: vsm.index(parents, children, persist_dir)."
            )
        with open(path, "rb") as f:
            self.parent_store = pickle.load(f)
        print(f"Loaded {len(self.parent_store)} parents ← {path}")

    def search_children(self, query: str, top_k: int = 20) -> List[Document]:
        """Given a query string, return top-k most similar children."""
        if self.vector_store is None:
            raise RuntimeError("Must call index() or attach a vector_store before search_children()")

        return self.vector_store.similarity_search(query, k=top_k)

    def get_parent(self, parent_id: str) -> ParentChunk:
        """Look up a parent by id. Fast — it's just a dict access."""
        persist_dir = str(Path(__file__).parent / "chroma_db")
        path = Path(persist_dir) / "parent_store.pkl"
        if not path.exists():
            raise FileNotFoundError(
                f"No parent_store.pkl at {path}. "
                "Run the indexing step first: vsm.index(parents, children, persist_dir)."
            )
        with open(path, "rb") as f:
            self.parent_store = pickle.load(f)
        return self.parent_store[parent_id]


# ========================================================================
# 2. Reciprocal Rank Fusion — combine multiple ranked lists into one
# ========================================================================

def reciprocal_rank_fusion(
    ranked_lists: List[List[str]],
    k: int = 60,
) -> List[tuple[str, float]]:
    """Fuse multiple ranked lists of document IDs into a single ranking.

    Args:
        ranked_lists: each inner list is a ranked list of doc IDs (rank 1 = best).
                      Example: [["docA", "docB", "docC"], ["docB", "docD", "docA"]]
        k: smoothing constant. 60 is the paper default.

    Returns:
        List of (doc_id, rrf_score) tuples, sorted by score descending.
    """

    # defaultdict(float) auto-initializes missing keys to 0.0.
    # Lets us just write scores[doc_id] += ... without checking if key exists.
    scores: dict[str, float] = defaultdict(float)

    # For each ranked list, add RRF contributions to each doc's total.
    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list, start=1):  # rank starts at 1
            # The core RRF formula: score += 1 / (k + rank)
            # Higher ranks (smaller numbers) contribute more.
            scores[doc_id] += 1.0 / (k + rank)

    # Sort docs by total score, highest first.
    # sorted() returns a list of (key, value) tuples from scores.items().
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused


# ========================================================================
# 3. ParentChildRetriever — the main retrieval orchestrator
# ========================================================================

class ParentChildRetriever:
    """Executes multi-query retrieval with RRF fusion and parent lookup.

    This is what the pipeline calls once per user query.
    """

    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        children_per_query: int = 20,
        final_top_k: int = 15,
    ):
        self.vsm = vector_store_manager
        self.children_per_query = children_per_query  # how many children to fetch per query
        self.final_top_k = final_top_k                # how many parents to return at the end

    def retrieve(self, queries: List[str]) -> List[ParentChunk]:
        """Given a list of queries (from Module 2), return ranked parent chunks.

        Flow:
          For each query → search → get top-K children
          Collect one ranked list per query (using parent_ids, not child content)
          Fuse all lists with RRF → single ranked list of parent_ids
          Look up each parent_id → return ParentChunks
        """

        # Step 1: For each query, do a vector search and collect the parent_ids
        # of the retrieved children, in order.
        #
        # Note: we rank at the PARENT level, not child level.
        # If 3 children from the same parent are retrieved, the parent only
        # appears ONCE in this query's list (at its highest-ranked child's position).
        parent_id_lists: List[List[str]] = []

        for query in queries:
            # Vector search: returns LangChain Documents (children) in ranked order
            child_docs = self.vsm.search_children(query, top_k=self.children_per_query)

            # Extract parent_ids in order, but dedupe within this query's list.
            # Why dedupe within a list? If the same parent is hit by multiple children,
            # we want it to appear ONCE in this query's ranking (at its best child's rank).
            seen = set()
            parent_ids_for_this_query = []
            for doc in child_docs:
                pid = doc.metadata["parent_id"]
                if pid not in seen:
                    seen.add(pid)
                    parent_ids_for_this_query.append(pid)

            parent_id_lists.append(parent_ids_for_this_query)

        # Step 2: Fuse the ranked lists with RRF.
        # Note we're fusing at the PARENT level — RRF rewards parents hit by multiple queries.
        fused = reciprocal_rank_fusion(parent_id_lists)

        # Step 3: Take top-N and look up the actual ParentChunk objects.
        top_ids = [pid for pid, score in fused[:self.final_top_k]]
        return [self.vsm.get_parent(pid) for pid in top_ids]


# ========================================================================
# Demo
# ========================================================================

if __name__ == "__main__":
    import os
    from langchain_core.documents import Document as LCDocument
    from chunking import SemanticParentChildChunker

    # Setup embeddings
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text:latest",
        temperature=0.3,  # deterministic embeddings
    )

    # Sample text — in the real pipeline this would be arXiv papers
    sample_text = """
    The transformer architecture uses self-attention to process sequences. Self-attention
    computes weighted sums of value vectors, where weights come from softmaxed dot products
    of query and key projections. This allows each position to attend to all other positions.

    Multi-head attention runs several attention mechanisms in parallel, then concatenates
    their outputs. Each head can learn different types of relationships. The original paper
    used 8 heads with dimension 64 each.

    Positional encodings are added to input embeddings because self-attention is
    permutation-invariant. The original paper used sinusoidal encodings, but learned
    positional embeddings are also common.

    Training transformers requires large datasets. The original paper used WMT 2014 English-
    German for translation. Modern LLMs use trillion-token corpora from web scrapes.
    """

    # # Step 1: Chunk the document (Module 1)
    # chunker = SemanticParentChildChunker(embeddings=embeddings)
    # LOADER_MAP = {
    # ".pdf":  PyPDFLoader,
    # ".txt":  TextLoader,
    # ".docx": UnstructuredWordDocumentLoader,
    # }

    # path = Path(__file__).parent / "papers"

    # folder = Path(path)

    # all_docs: List[Document] = []
    # files = [f for f in folder.rglob("*") if f.is_file() and f.suffix.lower() in LOADER_MAP]
    # all_docs: List[Document] = []

    # for file_path in files:
    #     suffix = file_path.suffix.lower()

    #     if suffix not in LOADER_MAP:
    #         print(f"Unsupported file type: {file_path.name} — skipping")
    #         continue

    #     try:
    #         loader_class = LOADER_MAP[suffix]
    #         loader = loader_class(str(file_path))
    #         docs = loader.load()

    #         # Inject source metadata on every page/chunk
    #         for doc in docs:
    #             doc.metadata["source"] = file_path.name
    #             doc.metadata["file_type"] = suffix

    #         print(f"Loaded {len(docs)} page(s) from {file_path.name}")
    #         for doc in docs:
    #             print(len(doc.page_content), "chars\n")
    #         all_docs.extend(docs)

    #     except Exception as e:
    #         print(f"Failed to load {file_path.name}: {e}")

    # parents, children = chunker.chunk_documents(all_docs)
    # print(f"Created {len(parents)} parents and {len(children)} children\n")

    # # Step 2: Index everything (Module 3)
    vsm = VectorStoreManager(embeddings=embeddings, collection_name="rag_documents")
    # Load existing persisted ChromaDB collection
    persist_dir = str(Path(__file__).parent / "chroma_db")
    vsm.vector_store = Chroma(
        persist_directory=persist_dir,
        collection_name="rag_documents",
        embedding_function=embeddings,
    )
    # If you have a saved parent_store, restore it here before retrieval.
    #vsm.index(parents, children)

    # Step 3: Simulate queries from Module 2 (just using a few phrasings manually)
    queries = [
        "How does attention work?",                      # original
        "Self-attention computes weighted sums of values", # HyDE-style hypothetical
        "What is multi-head attention?",                  # MultiQuery rewrite
    ]

    # Step 4: Print top 5 documents from the vector DB
    print("\n" + "="*60)
    print("TOP 5 DOCUMENTS IN VECTOR DB (ChromaDB)")
    print("="*60)

    all_data = vsm.vector_store.get()
    documents = all_data.get("documents", [])
    metadatas = all_data.get("metadatas", [])
    ids       = all_data.get("ids", [])

    for i, (doc_id, content, meta) in enumerate(zip(ids[:5], documents[:5], metadatas[:5])):
        print(f"[{i+1}] ID       : {doc_id}")
        print(f"     Parent  : {meta.get('parent_id', 'N/A')}")
        print(f"     Metadata: {meta}")
        print(f"     Content : {content.strip()[:200]}...")
        print()

    print("="*60 + "\n")

    # try:


    #     vsm.load_parent_store(persist_dir)
    # except FileNotFoundError as e:
    #     print(f"Could not load parent store: {e}")
    # else:
    #     print("\n" + "="*60)
    #     print("TOP 5 PARENT CHUNKS")
    #     print("="*60)
    #     for i, parent in enumerate(list(vsm.parent_store.values())[:5]):
    #         preview = parent.content.strip().replace("\n", " ")[:200]
    #         print(f"[{i+1}] ID       : {parent.id}")
    #         print(f"     Metadata : {parent.metadata}")
    #         print(f"     Content  : {preview}...")
    #         print()

    # Step 5: Retrieve!
    retriever = ParentChildRetriever(vsm, children_per_query=5, final_top_k=3)
    top_parents = retriever.retrieve(queries)

    print(f"\nTop {len(top_parents)} parents retrieved:")
    for i, p in enumerate(top_parents):
        preview = p.content.strip().replace("\n", " ")[:150]
        print(f"\n[{i+1}] {preview}...")
