# reasoning_retriever.py

import json
import os
import requests
import re
from node_summarizer import build_node_map
from config import get_settings


# ── Ollama LLM call ──────────────────────────────────────────────────────────

def call_ollama(prompt: str, model: str = "",
                temperature: float = 0.0) -> str:
    """
    Same Ollama wrapper as node_summarizer.py.
    temperature=0 is critical here — we want deterministic,
    logical node selection, not creative responses.
    """
    url = f"{get_settings().ollama_base_url}/api/generate"

    payload = {
        "model":  get_settings().ollama_chat_model,  # override with function arg if needed
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature}
    }

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "[ReasoningRetriever] Ollama not running. Start with: ollama serve"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(
            "[ReasoningRetriever] Ollama timed out. Try a smaller model."
        )
    except requests.exceptions.HTTPError as e:
        resp = getattr(e, "response", None)
        status = resp.status_code if resp is not None else "?"
        body = resp.text if resp is not None else str(e)
        raise RuntimeError(
            f"[ReasoningRetriever] Ollama returned HTTP {status}: {body}"
        )


# ── Compact tree builder ─────────────────────────────────────────────────────

def build_compact_tree_view(nodes: list[dict],
                             depth: int = 0) -> str:
    """
    Converts the full nested tree into a compact text representation
    showing only node_id, title, and summary per node.

    This is what the LLM reads in Stage 1 — NOT the raw text.
    Keeps the Stage 1 prompt small regardless of document size.

    Example output:
        [1] Introduction
            Summary: This section introduces the study background...
          [1.1] Background
              Summary: Covers the historical context of...
          [1.2] Objectives
              Summary: Lists the three primary research objectives...
    """
    lines = []
    indent = "  " * depth

    for node in nodes:
        # Node header line
        lines.append(
            f"{indent}[{node['node_id']}] {node['title']} "
            f"(pages {node['start_page']}–{node['end_page']})"
        )

        # Summary line (truncated to keep prompt concise)
        summary = node.get("summary", "").strip()
        if summary:
            short_summary = summary[:200] + "..." if len(summary) > 200 else summary
            lines.append(f"{indent}  Summary: {short_summary}")

        # Recurse into children
        if node.get("children"):
            lines.append(
                build_compact_tree_view(node["children"], depth + 1)
            )

    return "\n".join(lines)


# ── Stage 1 prompt — Navigation ──────────────────────────────────────────────

def build_navigation_prompt(query: str, compact_tree: str,
                             max_nodes: int = 3) -> str:
    """
    Asks the LLM to reason over the compact tree and return
    the node_ids most likely to contain the answer.

    Critically: instructs LLM to return ONLY a JSON list of node_ids.
    This makes parsing the response reliable.
    """
    return f"""You are an expert document navigator.

You have been given a document's table of contents with section summaries.
Your job is to identify which sections are most relevant to answer the user's query.

USER QUERY:
{query}

DOCUMENT STRUCTURE:
{compact_tree}

INSTRUCTIONS:
- Read each section's title and summary carefully.
- Identify the {max_nodes} most relevant sections that would contain the answer.
- Think step by step about which sections logically contain the answer.
- Return ONLY a JSON array of node_ids. Nothing else. No explanation.

Example output format:
["1.2", "2.1", "3"]

Your answer:"""


# ── Stage 1 — Parse node_ids from LLM response ───────────────────────────────

def parse_node_ids(llm_response: str) -> list[str]:
    """
    Parses the LLM's response to extract node_ids.
    LLM is instructed to return a JSON array, but we add
    regex fallback in case it adds extra text anyway.

    Returns list of node_id strings e.g. ["1.2", "2.1", "3"]
    """
    # Primary: try direct JSON parse
    try:
        cleaned = llm_response.strip()
        # Handle case where LLM wraps in ```json ... ```
        if "```" in cleaned:
            cleaned = re.sub(r"```(?:json)?", "", cleaned).replace("```", "").strip()
        node_ids = json.loads(cleaned)
        if isinstance(node_ids, list):
            return [str(nid) for nid in node_ids]
    except json.JSONDecodeError:
        pass

    # Fallback: regex extract anything that looks like a node_id
    # Matches patterns like "1", "1.2", "1.2.3"
    pattern = r'\b(\d+(?:\.\d+)*)\b'
    matches = re.findall(pattern, llm_response)

    # Filter out numbers that are clearly not node_ids (e.g. years like 2024)
    node_ids = [m for m in matches if len(m) <= 7]

    return node_ids if node_ids else []


# ── Stage 2 — Fetch raw text from selected nodes ─────────────────────────────

def fetch_node_contexts(node_ids: list[str],
                         node_map: dict) -> list[dict]:
    """
    Fetches the raw_text and metadata for each selected node_id
    from the flat node_map (O(1) lookup per node).

    Returns list of context dicts:
    [{node_id, title, start_page, end_page, raw_text}, ...]

    Gracefully skips node_ids not found in map (LLM hallucination guard).
    """
    contexts = []

    for node_id in node_ids:
        node = node_map.get(node_id)

        if node is None:
            print(f"[ReasoningRetriever] Warning: node_id '{node_id}' "
                  f"not found in map — skipping")
            continue

        contexts.append({
            "node_id":    node["node_id"],
            "title":      node["title"],
            "start_page": node["start_page"],
            "end_page":   node["end_page"],
            "raw_text":   node["raw_text"]
        })

    return contexts


# ── Retrieval audit trail ────────────────────────────────────────────────────

def build_retrieval_trace(query: str, selected_ids: list[str],
                           contexts: list[dict]) -> dict:
    """
    Builds a structured audit trail of the retrieval decision.
    This is one of Vectorless RAG's biggest advantages over vector RAG —
    full explainability of WHY each section was selected.

    In production, this trace can be logged, displayed to users,
    or used for evaluation/debugging.
    """
    return {
        "query":         query,
        "selected_ids":  selected_ids,
        "retrieved_sections": [
            {
                "node_id":    ctx["node_id"],
                "title":      ctx["title"],
                "page_range": f"{ctx['start_page']}–{ctx['end_page']}"
            }
            for ctx in contexts
        ]
    }


# ── Main orchestrator ────────────────────────────────────────────────────────

def retrieve(
    query:          str,
    tree:           list[dict],
    node_map:       dict,
    model:          str = get_settings().ollama_chat_model,
    max_nodes:      int = 3
) -> dict:
    """
    Full two-stage retrieval pipeline:
      Stage 1: LLM navigates compact tree → selects node_ids
      Stage 2: Fetch raw_text from selected node_ids

    Args:
        query:     User's question
        tree:      Full nested tree (from tree_summarized.json)
        node_map:  Flat {node_id: node} map (from node_map.json)
        model:     Ollama model name
        max_nodes: Max sections to retrieve (default 3)

    Returns:
        {
          "query":      str,
          "contexts":   [{node_id, title, start_page, end_page, raw_text}],
          "trace":      {retrieval audit trail},
          "node_ids":   [selected node_ids]
        }
    """
    print(f"\n[ReasoningRetriever] Query: {query}")

    # ── Stage 1: Build compact view and navigate ──
    print("[ReasoningRetriever] Stage 1: Building compact tree view...")
    compact_tree = build_compact_tree_view(tree)

    print("[ReasoningRetriever] Stage 1: LLM navigating tree...")
    nav_prompt   = build_navigation_prompt(query, compact_tree, max_nodes)
    llm_response = call_ollama(nav_prompt, model=model, temperature=0.0)

    print(f"[ReasoningRetriever] LLM response: {llm_response}")

    selected_ids = parse_node_ids(llm_response)
    print(f"[ReasoningRetriever] Selected node_ids: {selected_ids}")

    if not selected_ids:
        print("[ReasoningRetriever] Warning: No node_ids parsed. "
              "Falling back to root nodes.")
        selected_ids = [node["node_id"] for node in tree[:max_nodes]]

    # ── Stage 2: Fetch raw text ──
    print("[ReasoningRetriever] Stage 2: Fetching raw text from nodes...")
    contexts = fetch_node_contexts(selected_ids, node_map)

    # ── Build audit trace ──
    trace = build_retrieval_trace(query, selected_ids, contexts)

    print(f"[ReasoningRetriever] Retrieved {len(contexts)} sections")
    for ctx in contexts:
        print(f"  → [{ctx['node_id']}] {ctx['title']} "
              f"(pages {ctx['start_page']}–{ctx['end_page']})")

    return {
        "query":    query,
        "contexts": contexts,
        "trace":    trace,
        "node_ids": selected_ids
    }


# ── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    # Load tree and node map

   
    with open("data/tree_summarized.json", "r", encoding="utf-8", errors="replace") as f:
        tree = json.load(f)

    with open("data/node_map.json", "r", encoding="utf-8", errors="replace") as f:
        node_map = json.load(f)

    # Test query
    result = retrieve(
        query    = "What is attention is all you need about?",
        tree     = tree,
        node_map = node_map,
        model    = get_settings().ollama_chat_model,
        max_nodes = 3
    )

    print("\n── Retrieval Trace ──")
    print(json.dumps(result["trace"], indent=2))

    print("\n── Retrieved Context Preview ──")
    for ctx in result["contexts"]:
        print(f"\n[{ctx['node_id']}] {ctx['title']}")
        print(ctx["raw_text"])