from operator import le
import os
import json
import time
import subprocess
import numpy as np
import faiss
import fitz          # PyMuPDF
import requests
from pathlib import Path

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
OLLAMA_BASE_URL   = "http://localhost:11434"
EMBED_MODEL       = "nomic-embed-text"
CHAT_MODEL        = "llama3.2:1b"        # change to "mistral" if preferred
EMBED_DIM         = 768               # nomic-embed-text output dimension
CHUNK_SIZE        = 500               # characters
CHUNK_OVERLAP     = 50                # characters
TOP_K             = 5                 # chunks to retrieve
PAPERS_DIR        = "RAG CODE 1\\papers"          # folder with your PDFs
INDEX_DIR         = """RAG CODE 1\\rag_index"""      # where we save FAISS index


def check_ollama_running():
    try:

        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        models = [m["name"] for m in r.json().get("models", [])]
        print(f"Ollama running. Available models: {models}")
        
        # Warn if required models are missing
        for required in [EMBED_MODEL, CHAT_MODEL]:
            if not any(required in m for m in models):
                print(f" Model '{required}' not found. Run: ollama pull {required}")
    except requests.exceptions.ConnectionError:
        try:
            subprocess.run(["ollama", "list"], check=True)  # Attempt to start Ollama
            time.sleep(5)  # Wait a moment for Ollama to start
        except Exception:
            raise RuntimeError(
            "Ollama is not running. cmd command ollama list failed to start it. Start Ollama manually with: ollama serve"
            )
        raise RuntimeError(
            "Ollama is  running. Wait a moment and try again. If the problem persists, start Ollama manually with: ollama serve"
        )
    
def get_embedding(text: str) -> list[float]:
    print(f"Embedding text ({len(text)} chars)...")
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={
            "model": EMBED_MODEL,
            "prompt": text
        },
        timeout=30
    )
    response.raise_for_status()
    print(f" → Received embedding of length {len(response.json()['embedding'])}")
    return response.json()["embedding"]
    
def get_embeddings_batch(texts: list[str], 
                          batch_size: int = 8) -> np.ndarray:
    all_embeddings = []
    total = len(texts)
    print(f"Processing {total} texts in batches of {batch_size}")
    count = 0
    for i in range(0, total, batch_size):
        print(f"Processing batch {count}")
        count += 1
        batch = texts[i:i+batch_size]
        batch_embeddings = []
        for text in batch:
            try:
                emb = get_embedding(text)
                batch_embeddings.append(emb)
            except Exception as e:
                print(f"Error embedding text: {e}")
                batch_embeddings.append([0.0] * EMBED_DIM)  # Fallback to zero vector   
        all_embeddings.extend(batch_embeddings)
    embeddings = np.array(all_embeddings, dtype=np.float32)
    
    # L2 normalize so inner product = cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-10)
    return embeddings,all_embeddings


def chat_with_ollama(system_prompt: str, 
                     user_prompt: str) -> str:
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": CHAT_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            "stream": False,
            "options": {
                "temperature": 0.1,     # factual, low creativity
                "num_predict": 512,     # max tokens to generate
            }
        },
        timeout=120     # local models can be slow
    )
    response.raise_for_status()
    print(response)
    return response.json()["message"]["content"]

def load_pdf(pdf_path: str) -> str:
    """Extract text from a single PDF with page markers."""
    doc = fitz.open(pdf_path)
    full_text = ""
    for page_num, page in enumerate(doc):
        text = page.get_text().strip()
        if text:    # skip empty pages
            full_text += f"\n[Page {page_num + 1}]\n{text}\n"
    doc.close()
    return full_text


def load_all_pdfs(directory: str) -> dict[str, str]:
    
    pdf_files = list(Path(PAPERS_DIR).glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {PAPERS_DIR}. Please add some PDF files and try again.")
        return {}
    texts = []
    for pdf_path in pdf_files:
        print(f"Loading {pdf_path}...")
        text = str(load_pdf(pdf_path))
        texts.append({
            "filename": pdf_path.name,
            "text": text,
            "char_count":len(text)
        }
        )
        print(f"    → {len(text):,} characters")
    return texts

def chunk_document(doc : dict,chunk_size = CHUNK_SIZE,overlap = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    text = doc["text"]
    start = 0
    chunk_id = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size,text_length)

        chunk_text = text[start:end].strip()
        if len(chunk_text) >= 50:
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "source": doc["filename"],
                "start_char": start,
                "end_char": end,
                "metadata": {
                    "filename": doc["filename"],
                    "start_char": start,
                    "end_char": end
                }
            })
            chunk_id += 1
        if end == len(text):
            break
        start += chunk_size - overlap
    
    return chunks

def chunk_all_documents(documents: list[dict]) -> list[dict]:
    """Chunk all documents and return a flat list of all chunks."""
    all_chunks = []
    
    for doc in documents:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
        print(f"  {doc['filename']}: {len(chunks)} chunks")
    
    print(f"Total chunks: {len(all_chunks)}")
    return all_chunks

def build_index(chunks: list[dict]) -> faiss.IndexFlatIP:

    text = [c["text"] for c in chunks]

    start_time = time.time()
    embeddings, _ = get_embeddings_batch(text)
    end_time = time.time()
    elapsed = end_time - start_time 

    print(f"Generated embeddings for {len(chunks)} chunks in {elapsed:.2f} seconds")
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)
    print(f"FAISS index: {index.ntotal} vectors, dim={EMBED_DIM}")
    return index

def save_index(index: faiss.Index, chunks: list[dict]):
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, f"{INDEX_DIR}/index.faiss")
    with open(f"{INDEX_DIR}/chunks.json", "w") as f:
        json.dump(chunks, f, indent=2)
    print(f"Saved index to {INDEX_DIR}/")

def load_index() -> tuple[faiss.Index, list[dict]]:
    index = faiss.read_index(f"{INDEX_DIR}/index.faiss")
    with open(f"{INDEX_DIR}/chunks.json") as f:
        chunks = json.load(f)
    print(f"Loaded index: {index.ntotal} vectors, {len(chunks)} chunks")
    return index, chunks

def retrieve_chunks(query: str, index: faiss.Index, chunks: list[dict], top_k: int = TOP_K) -> list[dict]:
    query_emb = get_embedding(query)
    query_emb = np.array(query_emb, dtype=np.float32)
    # Normalize query embedding too (same space as indexed vectors)
    norm = np.linalg.norm(query_emb)
    query_emb = query_emb / (norm + 1e-10)
    query_emb = np.expand_dims(query_emb, axis=0)
    scores, indices = index.search(query_emb, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append({
            **chunks[idx],
            "similarity_score": float(score)
        })
    
    return results

def generate_answer(query: str, retrieved: list[dict]) -> str:
    
    # Build context with source attribution
    context_parts = []
    for i, chunk in enumerate(retrieved):
        context_parts.append(
            f"[Source: {chunk['source']} | "
            f"Score: {chunk['similarity_score']:.3f}]\n"
            f"{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    print(f"Constructed context for answer generation:\n{context[:500]}...")  # Show start of context
    
    system_prompt = """You are a precise research assistant. 
    Answer questions using ONLY the provided context.
    Always mention which source document your answer comes from.
    If the answer is not in the context, say: 
    "I cannot find this information in the provided documents."
    Never use outside knowledge."""

    user_prompt = f"""Context:

    {context}

    ---

    Question: {query}

    Answer (cite your sources):"""

    return chat_with_ollama(system_prompt, user_prompt)

def main():
    print("="*55)
    print("  Local RAG Bot — Ollama + FAISS + Multi-PDF")
    print("="*55 + "\n")
    # Verify Ollama is running
    check_ollama_running()
    # Load or build index
    if os.path.exists(INDEX_DIR) and \
       os.path.exists(f"{INDEX_DIR}/index.faiss"):
        print("\nExisting index found. Loading...")
        index, chunks = load_index()
    else:
        print(f"\nNo index found. Building from PDFs in '{PAPERS_DIR}/'...")
        os.makedirs(PAPERS_DIR, exist_ok=True)
        documents = load_all_pdfs(PAPERS_DIR)
        chunks = chunk_all_documents(documents)
        index = build_index(chunks)
        save_index(index, chunks)

    # Show corpus stats
    sources = set(c["source"] for c in chunks)
    print(f"\n Corpus: {len(sources)} documents, {len(chunks)} chunks")
    for src in sorted(sources):
        count = sum(1 for c in chunks if c["source"] == src)
        print(f"   {src}: {count} chunks")

    # Q&A loop
    print("\n" + "="*55)
    print(f"  Ready! Using {CHAT_MODEL} for answers.")
    print("  Type 'quit' to exit | 'sources' to list docs")
    print("="*55 + "\n")

    while True:
            query = input("Question: ").strip()
            
            if not query:
                continue
            if query.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            if query.lower() == "sources":
                for src in sorted(sources):
                    print(f"{src}")
                continue
            
            print("\n Retrieving...")
            retrieved = retrieve_chunks(query, index, chunks)
            
            print(f"\n Top {len(retrieved)} chunks retrieved:")
            for r in retrieved:
                print(f"   [{r['similarity_score']:.3f}] "
                    f"{r['source']} | "
                    f"chars {r['start_char']}–{r['end_char']}")
            
            print(f"\nGenerating answer with {CHAT_MODEL}...")
            answer = generate_answer(query, retrieved)
            
            print(f"\nAnswer:\n{answer}")
            print("\n" + "-"*55 + "\n")


if __name__ == "__main__":
    main()
