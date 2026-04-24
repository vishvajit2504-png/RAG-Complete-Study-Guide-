"""
query_transforms.py
-------------------
Phase 3 — Advanced Retrieval
Module 2: Query Transformations
 
Attacks query-document asymmetry with three techniques:
 
  1. HyDE — Generate a hypothetical answer, embed that instead of the query.
     Closes the gap between short queries and dense document chunks.
 
  2. Multi-Query — Rewrite the query N times with diverse vocabulary/phrasing.
     Catches chunks that only a specific phrasing would have matched.
 
  3. Decomposition — For multi-hop questions, split into atomic sub-queries.
     Each sub-query retrieves independently; results union at the next stage.
 
Each transformer returns a list of query strings (or (query, intent) pairs for
decomposition). Downstream (Module 3) will embed each, retrieve, and fuse.
 
Tracing: All public `transform()` methods are decorated with @traced, a no-op
decorator in Module 2. Module 5 replaces it with real LangSmith instrumentation.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from typing import Callable, List, Tuple, Union, Protocol

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)

def traced(name: str | None = None) -> Callable:
    """No-op decorator in Module 2. Placeholder so call sites don't change
    when Module 5 swaps in real tracing. The `name` argument becomes the
    LangSmith run name once activated.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Module 5 will add: start_trace(name or func.__name__, inputs=...)
            result = func(*args, **kwargs)
            # Module 5 will add: end_trace(outputs=result)
            return result
        return wrapper
    return decorator

@dataclass
class QueryTransformConfig:
    hyde_target_words: int = 100
    mulit_query_count : int = 3
    max_subqueries: int = 4
    temperature: float = 0.2

class QueryTransformer(Protocol):
    """Protocol for query transformers. Each takes a raw query string and returns
    a list of transformed queries (or (query, intent) pairs for decomposition).
    """
    def transform(self, query: str) -> List[Union[str, Tuple[str, str]]]:
        ...

class HyDETransformer:
    """HyDE (Hypothetical Document Embeddings) generates a synthetic answer to the
    query and embeds that instead of the original query. This helps bridge the
    gap between short queries and dense document chunks.
    """

    _PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         "You are a domain expert. Given a question, write a plausible and "
         "direct answer in approximately {target_words} words. "
         "Write in the style of a technical document — use precise terminology, "
         "avoid hedging phrases like 'it depends' or 'various factors'. "
         "Commit to specifics even if you're uncertain; the answer will be "
         "used for retrieval, not shown to users."),    
        ("human", "{query}"),
    ])

    def __init__(self, llm: ChatOllama, config: QueryTransformConfig | None = None):
        self.config = config or QueryTransformConfig()
        self.llm = llm
        self._chain = self._PROMPT | llm | StrOutputParser()

    @traced("hyde_transform")
    def transform(self, query: str) -> List[str]:
        try:
            answer = self._chain.invoke({ "query": query, "target_words": self.config.hyde_target_words })
            return [answer.strip()] # Return as a list for consistency with other transformers
        except OutputParserException as e:
            logger.error(f"HyDE output parsing failed: {e}")
            return [query] # Fallback to original query if generation fails
        
class multiQueryTransformer:
    """Multi-Query rewriting generates multiple diverse rewrites of the original
    query. This can help catch relevant document chunks that only match a specific
    phrasing or vocabulary.
    """
    query_count = QueryTransformConfig.mulit_query_count

    _PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         "You generate diverse alternative phrasings of a user's query for "
         "retrieval augmentation. Your goal is to maximize coverage of the "
         "underlying information need.\n\n"
         "Generate exactly {count} alternative queries. Each must vary from the "
         "original along AT LEAST ONE of these axes:\n"
         "  - Vocabulary (technical vs. casual, synonyms, domain jargon)\n"
         "  - Abstraction level (specific mechanisms vs. general concept)\n"
         "  - Angle (cause vs. effect, definition vs. application, what vs. how)\n\n"
         "Rules:\n"
         "  - Preserve the original intent — don't drift to unrelated topics.\n"
         "  - Each query must be self-contained (no 'it', 'this').\n"
         "  - Output ONLY the queries, one per line. No numbering, no preamble."),
        ("human", "Original query: {query}"),
    ])

    def __init__(self, llm: ChatOllama, config: QueryTransformConfig | None = None):
        self.config = config or QueryTransformConfig()
        self.llm = llm
        self._chain = self._PROMPT | llm | StrOutputParser()

    @traced("multi_query_transform")
    def transform(self, query: str) -> List[str]:
        try:
            rewrites_str = self._chain.invoke({ "query": query, "count": self.config.mulit_query_count })
            rewrites = self._parse_rewrites(rewrites_str)
            results = [query] + [r for r in rewrites if r.lower() != query.lower()]
            return results
        except Exception as e:
            logger.warning(f"MultiQuery failed, falling back to original query: {e}")
            return [query]
        
    @staticmethod
    def _parse_rewrites(rewrites_str: str) -> List[str]:
        rewrites = []
        for line in rewrites_str.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Strip common list prefixes: "1.", "1)", "- ", "* ", etc.
            for prefix_pattern in [". ", ") ", "- ", "* "]:
                if prefix_pattern in line[:4]:
                    line = line.split(prefix_pattern, 1)[-1].strip()
                    break
            # Strip surrounding quotes the LLM sometimes adds
            line = line.strip("\"'")
            if line:
                rewrites.append(line)
        return rewrites
    
@dataclass
class SubQuery:
    query: str
    intent: str

class QueryDecomposer:
    """Splits multi-hop queries into atomic sub-queries.
 
    Classifier-gated: first asks the LLM if the query is actually multi-hop.
    If not, returns the original query unchanged (cheap fall-through).
 
    Returns: [query_str_1, query_str_2, ...] — for API consistency with other
    transformers. Internally it also tracks intents; access via `.last_subqueries`
    if you need the structured form (debugging, tracing).
    """

    _CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         "Classify whether a query requires multiple independent retrievals to answer.\n"
         "Multi-hop = the query asks about 2+ distinct entities/topics/time periods "
         "that would be found in different documents.\n"
         "Single-hop = the query is about one thing, even if complex.\n\n"
         "Examples:\n"
         "  'What is self-attention?' → single\n"
         "  'How does self-attention work in detail?' → single (still one topic)\n"
         "  'Compare BERT and GPT architectures' → multi (two distinct models)\n"
         "  'What did the RAG paper propose and how does it compare to fine-tuning?' → multi\n\n"
         "Respond with exactly one word: 'single' or 'multi'."),
        ("human", "{query}"),
    ])

    max_subqueries = QueryTransformConfig.max_subqueries

    _DECOMPOSE_PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         "Decompose a complex multi-hop query into {max_n} or fewer atomic sub-queries. "
         "Each sub-query must:\n"
         "  - Be answerable from a single document or passage.\n"
         "  - Be self-contained (no pronouns referring to other sub-queries).\n"
         "  - Contribute a distinct piece of information to answering the original.\n\n"
         "Output format (strict): one sub-query per line, formatted as:\n"
         "  <sub-query> | <short intent>\n\n"
         "Example:\n"
         "  Original: 'Compare BERT and GPT architectures'\n"
         "  Output:\n"
         "    What is the architecture of BERT? | get BERT architecture\n"
         "    What is the architecture of GPT? | get GPT architecture"),
        ("human", "Original query: {query}"),
    ])

    def __init__(self, llm: ChatOllama, config: QueryTransformConfig | None = None):
        self.config = config or QueryTransformConfig()
        self.llm = llm
        self._classifier_chain = self._CLASSIFIER_PROMPT | llm | StrOutputParser()
        self._decompose_chain = self._DECOMPOSE_PROMPT | llm | StrOutputParser()
        # Expose last decomposition for debugging/tracing access
        self.last_subqueries: List[SubQuery] = []

    @traced("query_decomposition")
    def transform(self, query: str) -> List[str]:
        try:
            if not self._is_multi_hop(query):
                self.last_subqueries = [SubQuery(query=query, intent="single")]
                return [query]
            
            raw = self._decompose_chain.invoke({ "query": query, "max_n": self.config.max_subqueries })
            subqueries = self._parse_decomposition(raw)
            if not subqueries:
                logger.warning("Decomposition produced no valid sub-queries, falling back to original query")
                self.last_subqueries = [SubQuery(query=query, intent="single")]
                return [query]
            self.last_subqueries = subqueries
            return [sq.query for sq in subqueries]
        except Exception as e:
            logger.warning(f"Decomposition failed, falling back to original query: {e}")
            return [query]
        
    def _is_multi_hop(self, query: str) -> bool:
        """Cheap LLM classification. ~50 tokens, ~$0.00002 — well worth avoiding
        a bad decomposition on a simple query."""
        verdict = self._classifier_chain.invoke({"query": query}).strip().lower()
        return verdict == "multi"
    
    @staticmethod
    def _parse_decomposition(raw: str) -> List[SubQuery]:
        results = []
        for line in raw.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # Strip list prefixes (same defensive parsing as MultiQuery)
            for prefix_pattern in [". ", ") ", "- ", "* "]:
                if prefix_pattern in line[:4]:
                    line = line.split(prefix_pattern, 1)[-1].strip()
                    break
            # Skip lines that look like LLM preamble/refusals, not actual queries
            garbage_signals = ("here are", "i can't", "i cannot", "original:", "output:", "example:")
            if any(line.lower().startswith(s) for s in garbage_signals):
                continue
            if "|" in line:
                q, intent = line.split("|", 1)
                results.append(SubQuery(query=q.strip(), intent=intent.strip()))
            else:
                results.append(SubQuery(query=line, intent=""))
        return results
    
class  CompositeQueryTransformer:
    """Utility to chain multiple transformers together. Applies each in sequence,
    flattening the results. For example, you could do HyDE → Multi-Query to
    generate a hypothetical answer and then rewrite that into multiple queries.
    """
    def __init__(
        self,
        decomposer: QueryDecomposer | None = None,
        hyde: HyDETransformer | None = None,
        multi_query: multiQueryTransformer | None = None,
    ):
        self.decomposer = decomposer
        self.hyde = hyde
        self.multi_query = multi_query


    @traced(name="composite_query_transform")
    def transform(self, query: str) -> List[str]:
        # Step 1: decompose if multi-hop (or just return [query] if single-hop)
        base_queries = (
            self.decomposer.transform(query) if self.decomposer else [query]
        )
        
        # Step 2: for each (sub-)query, apply HyDE and MultiQuery
        all_queries: List[str] = []
        for q in base_queries:
            all_queries.append(q)  # always keep the (sub-)query itself
            if self.multi_query:
                all_queries.extend(self.multi_query.transform(q))
            if self.hyde:
                all_queries.extend(self.hyde.transform(q))
 
        # Dedupe while preserving order (order matters for RRF rank fusion in Module 3).
        # We lowercase for dedup comparison but keep original casing in output.
        seen_lower = set()
        deduped = []
        for q in all_queries:
            key = q.lower().strip()
            if key and key not in seen_lower:
                seen_lower.add(key)
                deduped.append(q)
        return deduped

if __name__ == "__main__":
    # Expects Azure OpenAI env vars:
    #   AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION
    #   AZURE_OPENAI_CHAT_DEPLOYMENT (e.g., "gpt-4o-mini")
    llm = ChatOllama(
        model="llama3.2:1b",
        temperature=0.2,
    )

 
    test_queries = [
        "How does attention work?",                                   # single-hop, conceptual
        "Compare the training objectives of BERT and GPT.",           # multi-hop
        "What is the Transformer architecture?",                      # single-hop
    ]
 
    hyde = HyDETransformer(llm)
    mq = multiQueryTransformer(llm)
    decomposer = QueryDecomposer(llm)
    composite = CompositeQueryTransformer(
        decomposer=decomposer, hyde=hyde, multi_query=mq
    )
 
    for q in test_queries:
        print(f"\n{'=' * 70}\nORIGINAL: {q}\n{'=' * 70}")
 
        print("\n[HyDE]")
        for h in hyde.transform(q):
            print(f"  → {h}")
 
        print("\n[Multi-Query]")
        for m in mq.transform(q):
            print(f"  → {m}")
 
        print("\n[Decomposition]")
        for d in decomposer.transform(q):
            print(f"  → {d}")
 
        print("\n[Composite — all transforms combined, deduped]")
        for i, c in enumerate(composite.transform(q)):
            preview = c[:120] + ("..." if len(c) > 120 else "")
            print(f"  {i+1}. {preview}")