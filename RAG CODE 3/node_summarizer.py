# node_summarizer.py

import json
import requests
import os
import sys
from pathlib import Path

from text_indexer import load_tree, save_tree
from config import get_settings


# ── Ollama LLM call ──────────────────────────────────────────────────────────

def call_ollama(prompt: str, model: str = "llama3", temperature: float = 0.0) -> str:
    """
    Calls your local Ollama instance to generate a summary.
    temperature=0 keeps summaries deterministic and factual.
    """
    url = get_settings().ollama_base_url + "/api/generate"

    payload = {
        "model":  get_settings().ollama_chat_model,  # override with function arg if needed
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "").strip()

    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "[NodeSummarizer] Ollama not running. Start with: ollama serve"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(
            "[NodeSummarizer] Ollama timed out. Try a smaller model."
        )


# ── Summary prompt ───────────────────────────────────────────────────────────

def build_summary_prompt(node_title: str, raw_text: str,
                          max_chars: int = 3000) -> str:
    """
    Builds the summarization prompt for a single node.
    Truncates raw_text to max_chars to avoid context overflow for large sections.
    """
    truncated_text = raw_text[:max_chars] if len(raw_text) > max_chars else raw_text

    return f"""You are a precise document analyst.

Summarize the following document section in 2-3 sentences.
Focus on: what topics are covered, key facts, and what questions this section can answer.
Do NOT add any preamble. Just the summary.

Section Title: {node_title}

Section Content:
{truncated_text}

Summary:"""


# ── Single node summarizer ───────────────────────────────────────────────────

def summarize_node(node: dict, model: str) -> str:
    """
    Generates a summary for a single node.
    Skips summarization if raw_text is too short (not worth an LLM call).
    """
    raw_text = node.get("raw_text", "").strip()

    # Skip very short sections — summary would be longer than content
    if len(raw_text) < 100:
        return raw_text

    prompt  = build_summary_prompt(node["title"], raw_text)
    summary = call_ollama(prompt, model=model)

    return summary


# ── Recursive tree walker ────────────────────────────────────────────────────

def summarize_tree(nodes: list[dict], model: str,
                   depth: int = 0, total: list = None) -> None:
    """
    Recursively walks the tree bottom-up and summarizes every node.

    Bottom-up order: children are summarized before their parent.
    This means by the time we summarize a parent node, its children
    already have summaries — useful for future parent-summary enrichment.

    Args:
        nodes:  List of tree nodes (roots at top level)
        model:  Ollama model name
        depth:  Current tree depth (for logging indentation)
        total:  Mutable counter list [done, total] for progress tracking
    """
    if total is None:
        # Count all nodes in tree first for progress display
        total = [0, count_nodes(nodes)]

    for node in nodes:
        # ── Recurse into children first (bottom-up) ──
        if node.get("children"):
            summarize_tree(node["children"], model, depth + 1, total)

        # ── Summarize this node ──
        total[0] += 1
        indent = "  " * depth
        print(f"{indent}[{total[0]}/{total[1]}] Summarizing: "
              f"[{node['node_id']}] {node['title'][:60]}")

        node["summary"] = summarize_node(node, model)

        print(f"{indent}  → Done ({len(node['summary'])} chars)")


# ── Node counter helper ──────────────────────────────────────────────────────

def count_nodes(nodes: list[dict]) -> int:
    """Recursively counts total nodes in the tree."""
    count = 0
    for node in nodes:
        count += 1
        if node.get("children"):
            count += count_nodes(node["children"])
    return count


# ── Flat node map builder ────────────────────────────────────────────────────

def build_node_map(nodes: list[dict], node_map: dict = None) -> dict:
    """
    Builds a flat {node_id: node} lookup map from the nested tree.
    Used by reasoning_retriever.py to quickly fetch a node by ID
    without traversing the full tree.

    Example:
        {
          "1":     {...},
          "1.1":   {...},
          "1.1.1": {...},
          "2":     {...}
        }
    """
    if node_map is None:
        node_map = {}

    for node in nodes:
        node_map[node["node_id"]] = node
        if node.get("children"):
            build_node_map(node["children"], node_map)

    return node_map


# ── Save node map separately ─────────────────────────────────────────────────

def save_node_map(node_map: dict, output_path: str) -> None:
    """
    Saves the flat node map to disk.
    reasoning_retriever.py loads this for O(1) node lookup by ID.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(node_map, f, indent=2, ensure_ascii=False)
    print(f"[NodeSummarizer] Node map saved → {output_path}")


# ── Main orchestrator ────────────────────────────────────────────────────────

def summarize_all_nodes(
    tree_path:    str,
    output_tree_path: str,
    node_map_path: str,
    model: str = "llama3"
) -> dict:
    """
    Full pipeline:
      Load tree → summarize all nodes → save updated tree → save node map

    Args:
        tree_path:         Path to tree_index.json (from tree_indexer.py)
        output_tree_path:  Where to save the summarized tree
        node_map_path:     Where to save the flat node map
        model:             Ollama model to use

    Returns:
        Flat node_map dict
    """
    print(f"[NodeSummarizer] Loading tree from: {tree_path}")
    tree = load_tree(tree_path)

    total_nodes = count_nodes(tree)
    print(f"[NodeSummarizer] Found {total_nodes} nodes to summarize\n")

    summarize_tree(tree, model=model)

    print(f"\n[NodeSummarizer] All nodes summarized")

    # Save updated tree (with summaries filled in)
    save_tree(tree, output_tree_path)

    # Build and save flat node map for fast lookup
    node_map = build_node_map(tree)
    save_node_map(node_map, node_map_path)

    return node_map


# ── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    node_map = summarize_all_nodes(
        tree_path         = "data/tree_index.json",
        output_tree_path  = "data/tree_summarized.json",
        node_map_path     = "data/node_map.json",
        model             = "llama3"
    )

    # Filter out nodes with empty raw_text before processing
    def filter_empty_nodes(nodes: list[dict]) -> list[dict]:
        """Recursively removes nodes with empty raw_text."""
        filtered = []
        for node in nodes:
            if node.get("raw_text", "").strip():
                if node.get("children"):
                    node["children"] = filter_empty_nodes(node["children"])
                filtered.append(node)
        return filtered
    
    with open("data/tree_summarized.json", "r", encoding="utf-8", errors="replace") as f:
        tree = json.load(f)

    tree = filter_empty_nodes(tree)
    total_nodes = count_nodes(tree)
    print(f"[NodeSummarizer] Found {total_nodes} non-empty nodes to summarize\n")

    print(f"\n[Test] Node map has {len(node_map)} entries")
    # Print first node's summary as sanity check
    first_key = list(node_map.keys())[0]
    print(f"\nSample summary for node [{first_key}]:")
    print(node_map[first_key]["summary"])