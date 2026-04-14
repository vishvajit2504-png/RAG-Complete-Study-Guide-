# context_extractor.py

import json


# ── Token estimation ─────────────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """
    Rough token estimate: ~4 characters per token (OpenAI/Ollama rule of thumb).
    Used to enforce context budget without a real tokenizer dependency.
    Good enough for budget checks — not for exact billing.
    """
    return len(text) // 4


# ── Single section formatter ──────────────────────────────────────────────────

def format_section(ctx: dict, index: int,
                   max_chars: int = 4000) -> str:
    """
    Formats a single retrieved node into a clean labeled section string.

    Includes:
      - Section index (for easy reference in the answer)
      - Node ID (for audit trail)
      - Title
      - Page range
      - Raw text (truncated to max_chars if needed)

    max_chars per section prevents one huge section from
    consuming the entire context window.
    """
    raw_text = ctx.get("raw_text", "").strip()

    # Truncate if section is too long
    if len(raw_text) > max_chars:
        raw_text = raw_text[:max_chars] + "\n[... section truncated ...]"

    section = (
        f"--- SECTION {index} ---\n"
        f"Node ID    : {ctx['node_id']}\n"
        f"Title      : {ctx['title']}\n"
        f"Pages      : {ctx['start_page']}–{ctx['end_page']}\n"
        f"Content    :\n{raw_text}\n"
    )

    return section


# ── Context budget enforcer ───────────────────────────────────────────────────

def enforce_token_budget(sections: list[str],
                          max_tokens: int = 6000) -> list[str]:
    """
    Ensures the combined context doesn't exceed max_tokens.
    Drops sections from the end (least relevant — reasoning_retriever
    returns them in relevance order) until budget is satisfied.

    Args:
        sections:   List of formatted section strings
        max_tokens: Hard limit for combined context

    Returns:
        Trimmed list of sections within budget
    """
    kept    = []
    running = 0

    for section in sections:
        tokens = estimate_tokens(section)
        if running + tokens > max_tokens:
            print(f"[ContextExtractor] Budget reached at section "
                  f"{len(kept) + 1} — dropping remaining "
                  f"{len(sections) - len(kept)} section(s)")
            break
        kept.append(section)
        running += tokens

    return kept


# ── Context assembler ─────────────────────────────────────────────────────────

def assemble_context(contexts: list[dict],
                      max_chars_per_section: int = 4000,
                      max_total_tokens: int = 6000) -> str:
    """
    Assembles all retrieved node contexts into one formatted string.

    Pipeline:
      1. Format each section with metadata headers
      2. Enforce token budget (drop least relevant if over limit)
      3. Join with clear separators
      4. Add header and footer for the generator

    Args:
        contexts:              List of context dicts from reasoning_retriever
        max_chars_per_section: Max chars per individual section
        max_total_tokens:      Max total tokens across all sections

    Returns:
        Single formatted context string ready for prompt injection
    """
    if not contexts:
        return "No relevant sections were retrieved from the document."

    # Step 1 — Format each section
    formatted_sections = [
        format_section(ctx, index=i + 1, max_chars=max_chars_per_section)
        for i, ctx in enumerate(contexts)
    ]

    # Step 2 — Enforce token budget
    formatted_sections = enforce_token_budget(
        formatted_sections, max_tokens=max_total_tokens
    )

    # Step 3 — Join with separators
    body = "\n\n".join(formatted_sections)

    # Step 4 — Wrap with header/footer
    header = (
        f"RETRIEVED CONTEXT ({len(formatted_sections)} section(s)):\n"
        f"{'=' * 60}\n"
    )
    footer = f"\n{'=' * 60}"

    return header + body + footer


# ── Metadata summary builder ──────────────────────────────────────────────────

def build_context_metadata(contexts: list[dict]) -> dict:
    """
    Builds a lightweight metadata summary of what was extracted.
    Useful for logging, API responses, and frontend citation display.

    Returns:
        {
          "total_sections": int,
          "total_pages_covered": int,
          "sections": [{node_id, title, page_range}]
          "estimated_tokens": int
        }
    """
    sections_meta = [
        {
            "node_id":    ctx["node_id"],
            "title":      ctx["title"],
            "page_range": f"{ctx['start_page']}–{ctx['end_page']}"
        }
        for ctx in contexts
    ]

    # Count unique pages covered across all retrieved sections
    all_pages = set()
    for ctx in contexts:
        all_pages.update(range(ctx["start_page"], ctx["end_page"] + 1))

    combined_text = " ".join(ctx.get("raw_text", "") for ctx in contexts)

    return {
        "total_sections":      len(contexts),
        "total_pages_covered": len(all_pages),
        "sections":            sections_meta,
        "estimated_tokens":    estimate_tokens(combined_text)
    }


# ── Main extractor entry point ────────────────────────────────────────────────

def extract_context(retrieval_result: dict,
                     max_chars_per_section: int = 4000,
                     max_total_tokens: int = 6000) -> dict:
    """
    Main entry point called by vectorless_pipeline.py.

    Takes the full retrieval_result dict from reasoning_retriever.retrieve()
    and produces a ready-to-use context package.

    Args:
        retrieval_result:      Output of reasoning_retriever.retrieve()
        max_chars_per_section: Per-section character limit
        max_total_tokens:      Total token budget

    Returns:
        {
          "query":            str,
          "context_string":   str,   ← inject this into generator prompt
          "metadata":         dict,  ← for logging / API response
          "trace":            dict   ← full retrieval audit trail
        }
    """
    contexts = retrieval_result.get("contexts", [])
    query    = retrieval_result.get("query", "")
    trace    = retrieval_result.get("trace", {})

    print(f"[ContextExtractor] Assembling context from "
          f"{len(contexts)} section(s)...")

    context_string = assemble_context(
        contexts,
        max_chars_per_section=max_chars_per_section,
        max_total_tokens=max_total_tokens
    )

    metadata = build_context_metadata(contexts)

    print(f"[ContextExtractor] Done — "
          f"~{metadata['estimated_tokens']} tokens across "
          f"{metadata['total_sections']} section(s) "
          f"covering {metadata['total_pages_covered']} page(s)")

    return {
        "query":          query,
        "context_string": context_string,
        "metadata":       metadata,
        "trace":          trace
    }


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Simulate a retrieval result (as if from reasoning_retriever.retrieve())
    mock_retrieval_result = {
        "query": "What caused the recall campaign?",
        "node_ids": ["2.1", "3.2"],
        "contexts": [
            {
                "node_id":    "2.1",
                "title":      "Defect Analysis",
                "start_page": 8,
                "end_page":   11,
                "raw_text":   "The primary defect was identified in the braking system. "
                              "Engineers found that the brake pads degraded faster than "
                              "expected under high temperature conditions..." * 10
            },
            {
                "node_id":    "3.2",
                "title":      "Investigation Findings",
                "start_page": 18,
                "end_page":   22,
                "raw_text":   "The investigation concluded that supplier batch B-447 "
                              "used a non-compliant compound in the brake pad material. "
                              "This affected approximately 12,000 units..." * 10
            }
        ],
        "trace": {
            "query":        "What caused the recall campaign?",
            "selected_ids": ["2.1", "3.2"],
            "retrieved_sections": [
                {"node_id": "2.1", "title": "Defect Analysis",         "page_range": "8–11"},
                {"node_id": "3.2", "title": "Investigation Findings",   "page_range": "18–22"}
            ]
        }
    }

    result = extract_context(mock_retrieval_result)

    print("\n── Context String ──")
    print(result["context_string"])

    print("\n── Metadata ──")
    print(json.dumps(result["metadata"], indent=2))