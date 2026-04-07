# llm.py
import logging

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


def get_llm() -> ChatOllama:
    """Initialize local Ollama chat model."""
    llm = ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_chat_model,     # llama3.2
        temperature=0.0,                       # deterministic — important for RAG
        num_ctx=4096,                          # context window size
    )
    logger.info(f"LLM ready — model='{settings.ollama_chat_model}', temp=0.0")
    return llm


def get_rag_prompt() -> ChatPromptTemplate:
    """
    RAG prompt template.
    Strict instructions to prevent hallucination outside provided context.
    """
    template = """You are a helpful assistant that answers questions strictly 
based on the provided context. 

Rules:
- Answer ONLY from the context below. 
- If the answer is not in the context, say: "I don't have enough information to answer that."
- Be concise and precise.
- Always mention which document your answer comes from.

Context:
{context}

Question:
{question}

Answer:"""

    prompt = ChatPromptTemplate.from_template(template)
    logger.info("RAG prompt template loaded.")
    return prompt


if __name__ == "__main__":
    llm = get_llm()
    prompt = get_rag_prompt()

    # Sanity check — call LLM directly without RAG
    response = llm.invoke("Say hello in one sentence.")
    print(response.content)