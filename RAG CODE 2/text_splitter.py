# text_splitter.py
import logging
from typing import List

from anyio import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import get_settings

# test_splitter.py
from document_loader import load_documents_from_folder


settings = get_settings()
logger = logging.getLogger(__name__)


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split raw documents into overlapping chunks.
    Metadata from each parent document is preserved on every chunk.
    """
    if not documents:
        logger.warning("No documents provided to split.")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,          # 1000 chars default
        chunk_overlap=settings.chunk_overlap,    # 200 chars default
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""], # priority order
    )

    chunks = splitter.split_documents(documents)

    # Inject chunk index per source file for traceability
    source_counters = {}
    chunk_counter = 0
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        source_counters[source] = source_counters.get(source, 0) + 1
        chunk.metadata["chunk_index"] = chunk_counter
        chunk_counter+=1

    logger.info(f"Split {len(documents)} document(s) into {len(chunks)} chunks")
    return chunks


def get_chunk_stats(chunks: List[Document]) -> dict:
    """
    Utility to inspect chunk size distribution.
    Useful during development to tune chunk_size and overlap.
    """
    if not chunks:
        return {}

    lengths = [len(c.page_content) for c in chunks]
    sources = {}
    for c in chunks:
        src = c.metadata.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1

    return {
        "total_chunks": len(chunks),
        "avg_chunk_size": round(sum(lengths) / len(lengths)),
        "min_chunk_size": min(lengths),
        "max_chunk_size": max(lengths),
        "chunks_per_source": sources,
    }

if __name__ == "__main__":
    path = Path(__file__).parent / "papers"
    docs = load_documents_from_folder(path)
    chunks = split_documents(docs)
    print(chunks[100])  # print first chunk for sanity check
    stats = get_chunk_stats(chunks)
    print(stats)