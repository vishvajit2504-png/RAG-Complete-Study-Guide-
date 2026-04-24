"""
reranker.py
-----------
Phase 3 — Module 4: Cross-Encoder Reranking

Takes a list of parent chunks and a query, returns the top-K most relevant chunks
using a cross-encoder (BAAI/bge-reranker-base).

Why this module exists:
  Bi-encoder retrieval (vector search) is fast but lossy — it compares query and
  document through two independent embeddings. A cross-encoder reads (query, doc)
  TOGETHER and produces a much more accurate relevance score.

  The standard two-stage pattern:
     Retrieve top 15-20 candidates with bi-encoder (fast)
     → Rerank to top 5 with cross-encoder (accurate)
"""

from pathlib import Path
import pickle
from typing import List
from sentence_transformers import CrossEncoder

from chunking import ParentChunk


class BGEReranker:
    """Reranker using BAAI/bge-reranker-base.

    This is a cross-encoder: it takes (query, document) pairs and returns
    a relevance score per pair. Higher score = more relevant.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        # Load the model. First call downloads ~275MB to your HuggingFace cache
        # (~/.cache/huggingface/), subsequent calls load from cache.
        #
        # The CrossEncoder class handles: tokenization, batching, GPU if available,
        # sigmoid on the output. We just call .predict() and get scores.
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        parents: List[ParentChunk],
        top_k: int = 5,
    ) -> List[ParentChunk]:
        """Rerank a list of parent chunks by relevance to the query.

        Args:
            query: the ORIGINAL user query (not the HyDE hypothetical).
            parents: candidate parent chunks from Module 3.
            top_k: how many to keep after reranking.

        Returns:
            Top-k parents, re-ordered by cross-encoder score.
            Each parent has a `.rerank_score` attribute attached for inspection.
        """

        # Edge case: no candidates, nothing to rerank.
        if not parents:
            return []

        # Edge case: asking for more than we have — just return what we have.
        # We still score them so the caller has scores available for inspection.
        if len(parents) <= top_k:
            top_k = len(parents)

        # Build (query, document) pairs — the cross-encoder's input format.
        # Note we pair query with parent.content (the full-context chunk),
        # not with any child or transformed query.
        pairs = [(query, parent.content) for parent in parents]

        # Run the cross-encoder. Returns a list of float scores, one per pair.
        # Internally this batches efficiently and uses GPU if available.
        # On CPU: ~100ms for 15 pairs of ~1500 tokens each.
        scores = self.model.predict(pairs)

        # Pair each parent with its score, sort descending, take top-k.
        # Using list of tuples keeps this simple and debuggable.
        scored_parents = list(zip(parents, scores))
        scored_parents.sort(key=lambda x: x[1], reverse=True)

        # Attach the score to each parent so downstream code (and humans debugging)
        # can see WHY each parent is where it is in the ranking.
        top_parents = []
        for parent, score in scored_parents[:top_k]:
            parent.rerank_score = float(score)  # attach as attribute
            top_parents.append(parent)

        return top_parents


# ========================================================================
# Demo
# ========================================================================

if __name__ == "__main__":
    # Manually construct some parent chunks to show how reranking works.
    # In the real pipeline these would come from Module 3.
    fake_parents = [
        ParentChunk(
            id="p1",
            content="BERT is pretrained with masked language modeling (MLM) using "
                    "cross-entropy loss on masked tokens, plus next sentence prediction "
                    "(NSP) with binary cross-entropy loss.",
            metadata={"source": "bert.pdf"},
        ),
        ParentChunk(
            id="p2",
            content="BERT was pretrained on Wikipedia and BookCorpus, containing "
                    "3.3 billion words total. The model has 340M parameters in its "
                    "large configuration.",
            metadata={"source": "bert.pdf"},
        ),
        ParentChunk(
            id="p3",
            content="The BERT architecture stacks 24 transformer layers with 1024 "
                    "hidden dimensions and 16 attention heads.",
            metadata={"source": "bert.pdf"},
        ),
        ParentChunk(
            id="p4",
            content="BERT achieves state-of-the-art on GLUE benchmarks, surpassing "
                    "previous methods by a substantial margin across 11 tasks.",
            metadata={"source": "bert.pdf"},
        ),
        ParentChunk(
            id="p5",
            content="During fine-tuning, BERT is adapted to downstream tasks by adding "
                    "a task-specific head and training end-to-end on labeled data.",
            metadata={"source": "bert.pdf"},
        ),
    ]

    persist_dir = str(Path(__file__).parent / "chroma_db")
    path = Path(persist_dir) / "parent_store.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"No parent_store.pkl at {path}. "
            "Run the indexing step first: vsm.index(parents, children, persist_dir)."
        )
    with open(path, "rb") as f:
        parent_store: dict = pickle.load(f)  # dict[str, ParentChunk]

    # Replace fake_parents with ParentChunk objects looked up from the store.
    fake_parents = [parent_store[pid] for pid in parent_store]

    # Query that only one of the chunks truly answers.
    query = "What loss function does BERT use during pretraining?"

    print(f"Query: {query}\n")
    print("Original order (as they came from retrieval):")
    for i, p in enumerate(fake_parents, 1):
        print(f"  [{i}] id={p.id} | {p.content[:80]}...")

    print("\n--- Running reranker ---\n")
    reranker = BGEReranker()
    top = reranker.rerank(query, fake_parents, top_k=3)

    print("Reranked order (top 3):")
    for i, p in enumerate(top, 1):
        print(f"  [{i}] score={p.rerank_score:.4f}")
        print(p)