"""
chunking.py
-----------
Phase 3 — Advanced Retrieval
Module 1: Semantic Chunking + Parent-Child Splitting

This module solves two chunking problems simultaneously:
  1. WHERE to split (semantic boundaries, not arbitrary character counts)
  2. HOW BIG chunks should be (two sizes: small for retrieval, large for context)

Flow:
    Document
      → SemanticChunker splits on topic shifts           → Parent chunks (~1500 tok)
      → Each parent is further split by RecursiveSplitter → Child chunks (~200 tok)
      → Children embedded & stored in vector DB
      → Parents stored in doc store, keyed by ID
      → Each child stores parent_id in metadata
"""

from __future__ import annotations

import os
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

from anyio import Path
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)
from pathlib import Path
from typing import List

# ------------------------------------------------------------
# Config — kept simple; in production this would live in a YAML/pydantic settings
# ------------------------------------------------------------
@dataclass
class ChunkingConfig:
    """All tunable knobs for the chunking pipeline, in one place."""

    # Semantic chunker
    # 'percentile' = split where sentence-to-sentence distance is above the Nth percentile
    # 95 is conservative (splits only on the most dissimilar boundaries → fewer, larger chunks)
    # Lower to 75 if you want more aggressive splitting
    breakpoint_threshold_type: str = "percentile"
    breakpoint_threshold_amount: float = 95.0

    # Child splitter (for retrieval)
    # ~200 tokens ≈ ~800 chars for English prose (rule of thumb: 1 token ≈ 4 chars)
    child_chunk_size: int = 800
    child_chunk_overlap: int = 80

    # Soft cap for parent chunks. If a semantic chunk exceeds this,
    # we split it again with RecursiveCharacterTextSplitter to keep LLM context manageable.
    # ~1500 tokens ≈ ~6000 chars
    max_parent_chars: int = 6000


# ------------------------------------------------------------
# Data classes — explicit types make the pipeline easier to reason about
# ------------------------------------------------------------
@dataclass
class ParentChunk:
    """A large chunk (~1500 tok) used as LLM context. Keyed by `id`."""

    id: str
    content: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class ChildChunk:
    """A small chunk (~200 tok) that gets embedded and indexed for retrieval.
    The `parent_id` field is the bridge back to the full-context parent."""

    content: str
    parent_id: str
    metadata: Dict = field(default_factory=dict)


# ------------------------------------------------------------
# Main chunker
# ------------------------------------------------------------
class SemanticParentChildChunker:
    """
    Two-stage chunker for Parent-Child retrieval.

    Stage 1 (semantic): split document on topic boundaries using sentence-embedding
                        similarity. This is our 'parent' layer.
    Stage 2 (recursive): split each parent into small children on paragraph/sentence
                         boundaries for fine-grained retrieval.

    Why two stages?
      - Semantic chunking is slow (embeds every sentence). Running it twice would be wasteful.
      - Recursive splitting on already-topically-coherent parents is cheap and sufficient.
    """

    def __init__(self, embeddings: OllamaEmbeddings, config: ChunkingConfig | None = None):
        self.config = config or ChunkingConfig()

        # Semantic chunker uses the SAME embeddings model as our retriever.
        # Keep these consistent — otherwise your "semantic boundaries" are drawn in
        # a different space than your retrieval happens in.
        self.semantic_splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=self.config.breakpoint_threshold_type,
            breakpoint_threshold_amount=self.config.breakpoint_threshold_amount,
        )

        # Fallback splitter for oversized parent chunks.
        # Hierarchy of separators: paragraph > line > sentence > word > char
        self.parent_size_capper = RecursiveCharacterTextSplitter(
            chunk_size=self.config.max_parent_chars,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # Child splitter: fixed small size. Using RecursiveCharacterTextSplitter because
        # at ~200 tokens, semantic boundaries matter less — we just want precise, focused chunks.
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.child_chunk_size,
            chunk_overlap=self.config.child_chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------
    def chunk_documents(
        self, docs: List[Document]
    ) -> Tuple[List[ParentChunk], List[ChildChunk]]:
        """
        Chunk a list of source Documents into (parents, children).

        Args:
            docs: Source documents (e.g., one per arXiv paper). Each should have
                  `page_content` and `metadata` (we preserve source info in both layers).

        Returns:
            parents: List of ParentChunk — store these in a doc store (key → text).
            children: List of ChildChunk — embed these and store in vector DB.
                      Each child's metadata includes `parent_id` so we can look up
                      the parent after retrieval.
        """
        all_parents: List[ParentChunk] = []
        all_children: List[ChildChunk] = []

        # Merge pages from the same PDF before chunking so topic continuity
        # across page boundaries is preserved for the semantic splitter.
        docs = self._merge_pdf_pages(docs)

        for doc in docs:
            
            # Stage 1: semantic split of this document into topic-coherent parents
            parent_texts = self._split_into_parents(doc.page_content)

            for parent_text in parent_texts:
                parent_id = str(uuid.uuid4())

                # Preserve original doc metadata (e.g., arXiv id, title) + our parent id
                parent_meta = {**doc.metadata, "parent_id": parent_id}
                all_parents.append(
                    ParentChunk(id=parent_id, content=parent_text, metadata=parent_meta)
                )

                # Stage 2: split this parent into small children
                child_texts = self.child_splitter.split_text(parent_text)
                for child_text in child_texts:
                    # Children inherit the parent's metadata so we can trace provenance
                    # without a second lookup. Critical for debugging retrieval failures.
                    child_meta = {**parent_meta}  # includes parent_id
                    all_children.append(
                        ChildChunk(
                            content=child_text,
                            parent_id=parent_id,
                            metadata=child_meta,
                        )
                    )

        return all_parents, all_children

    # --------------------------------------------------------
    # Internals
    # --------------------------------------------------------
    def _merge_pdf_pages(self, docs: List[Document]) -> List[Document]:
        """Merge pages from the same PDF into a single Document.

        PyPDFLoader yields one Document per page. Chunking page-by-page cuts
        topics at page boundaries. Merging first lets the semantic chunker
        see full cross-page context. Non-PDF documents pass through unchanged.
        """
        pdf_groups: Dict[str, List[Document]] = defaultdict(list)
        non_pdf: List[Document] = []

        for doc in docs:
            if doc.metadata.get("file_type") == ".pdf":
                pdf_groups[doc.metadata.get("source", "unknown")].append(doc)
            else:
                non_pdf.append(doc)

        merged: List[Document] = list(non_pdf)
        for _, pages in pdf_groups.items():
            pages.sort(key=lambda d: d.metadata.get("page", 0))
            combined_text = "\n\n".join(p.page_content for p in pages)
            base_meta = {**pages[0].metadata, "total_pages": len(pages)}
            base_meta.pop("page", None)  # replaced by total_pages
            merged.append(Document(page_content=combined_text, metadata=base_meta))

        return merged

    def _split_into_parents(self, text: str) -> List[str]:
        """Semantic split, then size-cap any chunks that are too large.

        Why size-cap?
          Semantic chunker can occasionally produce very large chunks when a section
          is topically uniform (e.g., a long proof, a detailed experiment description).
          Such parents would bloat LLM context. We cap them with recursive splitting,
          accepting that some splits may not be perfectly semantic — it's a pragmatic
          safety net.
        """
        # SemanticChunker.split_text returns List[str]
        semantic_chunks = self.semantic_splitter.split_text(text)

        sized_chunks: List[str] = []
        for chunk in semantic_chunks:
            if len(chunk) <= self.config.max_parent_chars:
                sized_chunks.append(chunk)
            else:
                # Oversized → recursive split. The sub-chunks are no longer guaranteed
                # to be topic-coherent, but they stay within LLM context budget.
                sized_chunks.extend(self.parent_size_capper.split_text(chunk))

        return sized_chunks


# ------------------------------------------------------------
# Quick sanity-check demo (runs when file is executed directly)
# ------------------------------------------------------------
if __name__ == "__main__":
    # A small synthetic "paper" with clear topic shifts — lets you eyeball whether
    # the semantic chunker is splitting sensibly before you run it on real arXiv PDFs.
    sample_text = """
    Transformers have revolutionized natural language processing. The self-attention
    mechanism allows each token to attend to every other token in the sequence. This
    architecture, introduced in the 2017 paper "Attention is All You Need", replaced
    recurrent networks as the dominant paradigm. The key innovation is parallelizable
    computation across sequence positions.

    Training large transformers requires massive datasets. The Common Crawl corpus
    provides over a petabyte of web text. Data quality matters more than quantity —
    deduplication, filtering low-quality content, and removing PII are essential
    preprocessing steps. Many teams now invest more engineering effort in data curation
    than in model architecture.

    Evaluation of language models remains challenging. Standard benchmarks like GLUE
    and SuperGLUE have been saturated. Newer benchmarks such as MMLU, BIG-Bench, and
    HELM attempt to probe broader capabilities. However, benchmark contamination —
    where evaluation data leaks into training — is a persistent concern.

    Retrieval-augmented generation combines parametric and non-parametric knowledge.
    A retriever fetches relevant documents from an external corpus. A generator
    conditions its output on both the query and retrieved context. This decouples
    factual recall from reasoning, enabling smaller models to achieve competitive
    performance on knowledge-intensive tasks.
    """

    # Note: this demo assumes Azure OpenAI env vars are set:
    #   AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION
    # and an embeddings deployment name set via AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT.

    ollama_base_url: str = "http://localhost:11434"
    ollama_embedding_model: str = "nomic-embed-text" 

    def get_embeddings() -> OllamaEmbeddings:
        return OllamaEmbeddings(
            base_url = ollama_base_url,
                model = ollama_embedding_model
        )

    embeddings = get_embeddings()

    chunker = SemanticParentChildChunker(embeddings=embeddings)

    LOADER_MAP = {
    ".pdf":  PyPDFLoader,
    ".txt":  TextLoader,
    ".docx": UnstructuredWordDocumentLoader,
    }

    path = Path(__file__).parent / "papers"

    folder = Path(path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {path}")

    all_docs: List[Document] = []
    files = [f for f in folder.rglob("*") if f.is_file() and f.suffix.lower() in LOADER_MAP]
    all_docs: List[Document] = []
    for file_path in files:
        suffix = file_path.suffix.lower()

        if suffix not in LOADER_MAP:
            print(f"Unsupported file type: {file_path.name} — skipping")
            continue

        try:
            loader_class = LOADER_MAP[suffix]
            loader = loader_class(str(file_path))
            docs = loader.load()

            # Inject source metadata on every page/chunk
            for doc in docs:
                doc.metadata["source"] = file_path.name
                doc.metadata["file_type"] = suffix

            print(f"Loaded {len(docs)} page(s) from {file_path.name}")
            for doc in docs:
                print(len(doc.page_content), "chars\n")
            all_docs.extend(docs)

        except Exception as e:
            print(f"Failed to load {file_path.name}: {e}")

    parents, children = chunker.chunk_documents(all_docs)

    print(f"Produced {len(parents)} parent chunks, {len(children)} child chunks\n")
    print("--- PARENTS ---")
    for i, p in enumerate(parents):
        preview = p.content.strip().replace("\n", " ")[:120]
        print(f"[P{i}] {preview}... (len={len(p.content)} chars)")
    print("\n--- CHILDREN ---")
    for i, c in enumerate(children[:6]):  # first 6 only for readability
        preview = c.content.strip().replace("\n", " ")[:100]
        print(f"[C{i}] parent={c.parent_id[:8]}... | {preview}...")