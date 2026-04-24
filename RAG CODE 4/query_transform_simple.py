"""
query_transforms_simple.py
--------------------------
SIMPLIFIED version of Module 2 — focus on understanding each line.

Three ways to transform a query before retrieval:
  1. HyDE           — write a fake answer, embed that
  2. Multi-Query    — write 3 alternative phrasings
  3. Decomposition  — split a complex query into sub-queries

No error handling, no tracing, no config objects. Just the core idea.
"""

import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama


# ========================================================================
# 1. HyDE — generate a hypothetical answer, then search with IT instead of the query
# ========================================================================

class HyDE:
    def __init__(self, llm):
        # Save the LLM we'll use to generate the fake answer.
        self.llm = llm

        # Build the prompt: tell the LLM to write a plausible answer.
        # "human" is the slot where the actual question goes.
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "Write a short, direct answer to the question in about 100 words. "
                       "Use technical language. Do not hedge."),
            ("human", "{query}"),
        ])

        # A "chain" in LangChain = prompt | model | parser.
        # The `|` operator pipes the output of one stage into the next.
        # StrOutputParser() just converts the model's response object to a plain string.
        self.chain = self.prompt | self.llm | StrOutputParser()

    def transform(self, query: str) -> list[str]:
        # Invoke the chain with the user's query filled into the {query} slot.
        # Returns a string (the hypothetical answer).
        hypothetical_answer = self.chain.invoke({"query": query})

        # Return it as a list (so all 3 transformers return the same type).
        # Note: we return ONLY the hypothetical, NOT the original query.
        # HyDE replaces the query for embedding purposes.
        return [hypothetical_answer]


# ========================================================================
# 2. Multi-Query — generate 3 alternative phrasings of the question
# ========================================================================

class MultiQuery:
    def __init__(self, llm):
        self.llm = llm

        # Prompt asks for 3 alternatives, one per line.
        # The "vary along" line is important — without it, LLMs just swap synonyms.
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "Generate exactly 3 alternative phrasings of the user's question. "
                       "Each must vary in vocabulary, abstraction, or angle. "
                       "Output one per line, no numbering."),
            ("human", "{query}"),
        ])

        self.chain = self.prompt | self.llm | StrOutputParser()

    def transform(self, query: str) -> list[str]:
        # Get raw text with 3 lines in it.
        raw = self.chain.invoke({"query": query})

        # Split on newlines, drop any empty lines.
        rewrites = [line.strip() for line in raw.split("\n") if line.strip()]

        # Return the ORIGINAL query + the rewrites.
        # Unlike HyDE, MultiQuery is ADDITIVE — we keep the original because
        # it's often the best query; rewrites just add more angles.
        return [query] + rewrites


# ========================================================================
# 3. Decomposition — split a multi-hop query into sub-queries
# ========================================================================

class Decomposer:
    def __init__(self, llm):
        self.llm = llm

        # Ask the LLM to split into up to 3 atomic sub-queries.
        # "Atomic" = answerable from a single passage.
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "If the user's query asks about multiple things, split it into "
                       "up to 3 simpler sub-queries, one per line. "
                       "If the query is already simple, output it unchanged. "
                       "Output only the queries, no numbering or explanation."),
            ("human", "{query}"),
        ])

        self.chain = self.prompt | self.llm | StrOutputParser()

    def transform(self, query: str) -> list[str]:
        raw = self.chain.invoke({"query": query})

        # Same parsing as MultiQuery: lines → list.
        sub_queries = [line.strip() for line in raw.split("\n") if line.strip()]

        # If the LLM returned nothing useful, fall back to the original query.
        if not sub_queries:
            return [query]

        return sub_queries


# ========================================================================
# Demo — run all three on a couple of test queries
# ========================================================================

if __name__ == "__main__":
    # Create an LLM client. Needs these env vars:
    #   AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION
    llm = ChatOllama(
        model="llama3.2:1b",
        temperature=0.2,   # low but not zero — we want slight variation in rewrites
    )

    # Create one instance of each transformer.
    hyde = HyDE(llm)
    mq = MultiQuery(llm)
    decomp = Decomposer(llm)

    # Test query 1: simple conceptual question (good for HyDE + MultiQuery)
    q1 = "How does attention work?"
    print(f"\n--- Query: {q1} ---")
    print("\nHyDE output:")
    print(hyde.transform(q1))
    print("\nMultiQuery output:")
    for q in mq.transform(q1):
        print(f"  {q}")
    print("\nDecomposer output:")
    for q in decomp.transform(q1):
        print(f"  {q}")

    # Test query 2: multi-hop question (good for Decomposer)
    q2 = "Compare BERT and GPT architectures."
    print(f"\n--- Query: {q2} ---")
    print("\nDecomposer output:")
    for q in decomp.transform(q2):
        print(f"  {q}")