from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
CHROMA_PERSIST_DIR = "RAG CODE 4/chroma_db"
DOCS_PATH = "RAG CODE 4/papers"


def build_retriever(k: int = 4):
    embeddings = OllamaEmbeddings(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_EMBEDDING_MODEL,
    )

    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name="rag_documents",
        embedding_function=embeddings,
    )

    # If the collection is empty, load, chunk, and embed from the papers directory
    if vectorstore._collection.count() == 0:
        print(f"[retriever] ChromaDB empty — loading docs from '{DOCS_PATH}'...")

        loader = PyPDFDirectoryLoader(DOCS_PATH)
        docs = loader.load()
        if not docs:
            raise FileNotFoundError(f"No PDFs found in '{DOCS_PATH}'")

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        print(f"[retriever] Indexing {len(chunks)} chunks from {len(docs)} pages...")

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        print("[retriever] Done — vector store ready.")
    else:
        print(f"[retriever] Loaded existing ChromaDB ({vectorstore._collection.count()} chunks).")

    return vectorstore.as_retriever(search_kwargs={"k": k})
