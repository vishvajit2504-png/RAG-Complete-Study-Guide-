
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

llm = ChatOllama(model="llama3.2:1b", temperature=0)

# ── 1. Relevance Grader ──────────────────────────────────────────────
class GradeDocuments(BaseModel):
    """Binary relevance score for a retrieved document."""
    binary_score: str = Field(
        description="Document is relevant to the question? 'yes' or 'no'"
    )

structured_llm_grader = llm.with_structured_output(GradeDocuments)

grade_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a relevance grader. Assess whether a document 
    contains information useful for answering a question.
    Give a binary 'yes' or 'no' score. Be lenient — if the document 
    contains ANY relevant facts, score 'yes'."""),
    ("human", "Document:\n{document}\n\nQuestion: {question}"),
])

retrieval_grader = grade_prompt | structured_llm_grader


# ── 2. Query Rewriter ────────────────────────────────────────────────
rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a query optimizer for vector search. 
    Rewrite the input question to improve semantic retrieval.
    Make it more specific and self-contained. Output ONLY the rewritten question."""),
    ("human", "Original question: {question}\nRewritten question:"),
])

question_rewriter = rewrite_prompt | llm | StrOutputParser()


# ── 3. RAG Generator ─────────────────────────────────────────────────
generate_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert assistant. Answer the question using 
    ONLY the provided context. If context is insufficient, say so clearly.
    Be concise and factual."""),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

rag_chain = generate_prompt | llm | StrOutputParser()