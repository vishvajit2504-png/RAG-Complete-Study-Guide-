# document_loader.py
import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_core.documents import Document

from config import get_settings

settings = get_settings()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Map file extensions to their loader classes
LOADER_MAP = {
    ".pdf":  PyPDFLoader,
    ".txt":  TextLoader,
    ".docx": UnstructuredWordDocumentLoader,
}


def load_document(file_path: Path) -> List[Document]:
    """Load a single file and return list of Document objects."""
    suffix = file_path.suffix.lower()

    if suffix not in LOADER_MAP:
        logger.warning(f"Unsupported file type: {file_path.name} — skipping")
        return []

    try:
        loader_class = LOADER_MAP[suffix]
        loader = loader_class(str(file_path))
        docs = loader.load()

        # Inject source metadata on every page/chunk
        for doc in docs:
            doc.metadata["source"] = file_path.name
            doc.metadata["file_type"] = suffix

        logger.info(f"Loaded {len(docs)} page(s) from {file_path.name}")
        return docs

    except Exception as e:
        logger.error(f"Failed to load {file_path.name}: {e}")
        return []


def load_documents_from_folder(folder_path: str) -> List[Document]:
    """
    Recursively load all supported documents from a folder.
    Returns a flat list of Document objects with metadata.
    """
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    all_docs: List[Document] = []
    files = [f for f in folder.rglob("*") if f.is_file() and f.suffix.lower() in LOADER_MAP]

    if not files:
        logger.warning(f"No supported files found in {folder_path}")
        return []

    logger.info(f"Found {len(files)} file(s) in {folder_path}")

    for file_path in files:
        docs = load_document(file_path)
        all_docs.extend(docs)

    logger.info(f"Total documents loaded: {len(all_docs)}")
    return all_docs

if __name__ == "__main__":
    path = Path(__file__).parent / "papers"
    load_documents_from_folder(str(path))