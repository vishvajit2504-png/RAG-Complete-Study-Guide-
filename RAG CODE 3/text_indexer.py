# tree_indexer.py

import fitz  # PyMuPDF
import json
import re
import os
from pathlib import Path


# ── Node structure ──────────────────────────────────────────────────────────

def create_node(node_id: str, title: str, level: int,
                start_page: int, end_page: int, raw_text: str) -> dict:
    """
    Creates a single tree node representing one document section.
    summary is left empty — node_summarizer.py will fill it.
    """
    return {
        "node_id":    node_id,
        "title":      title,
        "level":      level,
        "start_page": start_page,
        "end_page":   end_page,
        "raw_text":   raw_text,
        "summary":    "",
        "children":   []
    }


# ── PDF text extraction ──────────────────────────────────────────────────────

def extract_pages(pdf_path: str) -> list[dict]:
    """
    Extracts text from each page of the PDF using PyMuPDF.
    Returns a list of dicts: [{page_num, text, blocks}, ...]
    blocks contain font size info — used for heading detection.
    """
    doc = fitz.open(pdf_path)
    pages = []

    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]  # rich format with font info
        plain_text = page.get_text("text")        # plain text for storage

        pages.append({
            "page_num": page_num + 1,             # 1-indexed
            "text":     plain_text,
            "blocks":   blocks
        })

    doc.close()
    return pages


# ── Heading detection ────────────────────────────────────────────────────────

def detect_headings(pages: list[dict]) -> list[dict]:
    """
    Scans all pages and detects headings based on:
    - Font size (larger = higher heading level)
    - Bold flag
    - Short line length (headings are rarely long sentences)

    Returns a flat list of detected headings:
    [{page_num, text, font_size, level}, ...]
    """
    all_spans = []

    for page in pages:
        for block in page["blocks"]:
            if block.get("type") != 0:   # type 0 = text block
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span["text"].strip()
                    if not text or len(text) < 3:
                        continue

                    all_spans.append({
                        "page_num":  page["page_num"],
                        "text":      text,
                        "font_size": round(span["size"], 1),
                        "is_bold":   bool(span["flags"] & 2**4),  # bold flag
                        "length":    len(text)
                    })

    if not all_spans:
        return []

    # Determine font size thresholds dynamically
    font_sizes = sorted(set(s["font_size"] for s in all_spans), reverse=True)
    
    # Top 3 distinct font sizes → H1, H2, H3
    heading_sizes = font_sizes[:3] if len(font_sizes) >= 3 else font_sizes

    headings = []
    for span in all_spans:
        is_heading_size = span["font_size"] in heading_sizes
        is_short        = span["length"] < 120          # headings are concise
        is_bold         = span["is_bold"]

        if (is_heading_size or is_bold) and is_short:
            level = 1  # default
            if span["font_size"] == heading_sizes[0]:
                level = 1
            elif len(heading_sizes) > 1 and span["font_size"] == heading_sizes[1]:
                level = 2
            elif len(heading_sizes) > 2 and span["font_size"] == heading_sizes[2]:
                level = 3

            headings.append({
                "page_num":  span["page_num"],
                "text":      span["text"],
                "font_size": span["font_size"],
                "level":     level
            })

    return headings


# ── Raw text grouping ────────────────────────────────────────────────────────

def group_text_under_headings(pages: list[dict],
                               headings: list[dict]) -> list[dict]:
    """
    For each heading, collects all page text from its start page
    until the next heading begins. This becomes the node's raw_text.

    Returns flat sections list:
    [{title, level, start_page, end_page, raw_text}, ...]
    """
    if not headings:
        # No headings detected — treat entire doc as single root node
        full_text = "\n".join(p["text"] for p in pages)
        return [{
            "title":      "Document",
            "level":      1,
            "start_page": 1,
            "end_page":   pages[-1]["page_num"],
            "raw_text":   full_text
        }]

    page_text_map = {p["page_num"]: p["text"] for p in pages}
    sections = []

    for i, heading in enumerate(headings):
        start_page = heading["page_num"]
        end_page   = (headings[i + 1]["page_num"] - 1
                      if i + 1 < len(headings)
                      else pages[-1]["page_num"])

        # Collect raw text for this section's page range
        raw_text = "\n".join(
            page_text_map.get(p, "")
            for p in range(start_page, end_page + 1)
        )

        sections.append({
            "title":      heading["text"],
            "level":      heading["level"],
            "start_page": start_page,
            "end_page":   end_page,
            "raw_text":   raw_text.strip()
        })

    return sections


# ── Tree construction ────────────────────────────────────────────────────────

def build_tree(sections: list[dict]) -> list[dict]:
    """
    Converts the flat sections list into a nested tree.
    H1 nodes are roots; H2 become children of H1; H3 become children of H2.

    Uses a stack to track the current parent at each level.
    Returns list of root nodes (the full tree).
    """
    roots = []
    # stack[level] = current node at that level
    stack = {0: None, 1: None, 2: None, 3: None}
    counters = {1: 0, 2: 0, 3: 0}

    for section in sections:
        level = section["level"]
        counters[level] += 1

        # Reset child counters when parent changes
        for child_level in range(level + 1, 4):
            counters[child_level] = 0

        # Build node_id like "1", "1.2", "1.2.3"
        if level == 1:
            node_id = str(counters[1])
        elif level == 2:
            node_id = f"{counters[1]}.{counters[2]}"
        else:
            node_id = f"{counters[1]}.{counters[2]}.{counters[3]}"

        node = create_node(
            node_id    = node_id,
            title      = section["title"],
            level      = level,
            start_page = section["start_page"],
            end_page   = section["end_page"],
            raw_text   = section["raw_text"]
        )

        stack[level] = node

        if level == 1:
            roots.append(node)
        else:
            parent = stack[level - 1]
            if parent:
                parent["children"].append(node)
            else:
                # Orphan node — promote to root
                roots.append(node)

    return roots


# ── Save / Load ──────────────────────────────────────────────────────────────

def save_tree(tree: list[dict], output_path: str) -> None:
    """Saves the tree index to disk as JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tree, f, indent=2, ensure_ascii=False)
    print(f"[TreeIndexer] Tree saved → {output_path}")


def load_tree(tree_path: str) -> list[dict]:
    """Loads a previously saved tree index from disk."""
    with open(tree_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Main orchestrator ────────────────────────────────────────────────────────

def build_tree_index(pdf_path: str, output_path: str) -> list[dict]:
    """
    Full pipeline:
      PDF → extract pages → detect headings → group text → build tree → save

    Args:
        pdf_path:    Path to the input PDF
        output_path: Where to save tree_index.json

    Returns:
        The tree as a list of root nodes (dicts)
    """
    print(pdf_path)
    print(f"[TreeIndexer] Loading PDF: {pdf_path}")
    pages = extract_pages(pdf_path)
    print(f"[TreeIndexer] Extracted {len(pages)} pages")

    headings = detect_headings(pages)
    print(f"[TreeIndexer] Detected {len(headings)} headings")

    sections = group_text_under_headings(pages, headings)
    print(f"[TreeIndexer] Grouped into {len(sections)} sections")

    tree = build_tree(sections)
    print(f"[TreeIndexer] Built tree with {len(tree)} root nodes")

    save_tree(tree, output_path)
    return tree


# ── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    docs_path = os.path.join(os.path.dirname(__file__), "papers")

    if not os.path.isdir(docs_path):
        raise FileNotFoundError(f"Papers directory not found: {docs_path}")

    pdf_files = sorted([f for f in os.listdir(docs_path)
                        if f.lower().endswith('.pdf')])

    if not pdf_files:
        print(f"[TreeIndexer] No PDF files found in {docs_path}")
        raise SystemExit(0)

    for pdf_name in pdf_files:
        pdf_path = os.path.join(docs_path, pdf_name)
        stem = os.path.splitext(pdf_name)[0]
        output_path = os.path.join('data', f"{stem}_tree_index.json")

        print(f"\n[TreeIndexer] Processing: {pdf_name}")
        try:
            tree = build_tree_index(pdf_path=pdf_path, output_path=output_path)
        except Exception as e:
            print(f"[TreeIndexer] Error processing {pdf_name}: {e}")
            continue

        # Print a short verification of root nodes
        def print_tree(nodes, indent=0):
            for node in nodes:
                print(" " * indent + f"[{node['node_id']}] {node['title']} "
                      f"(pages {node['start_page']}–{node['end_page']})")
                print_tree(node["children"], indent + 4)

        print_tree(tree)

    # Build a single combined tree by namespacing node_ids per document
    def _normalize_prefix(s: str) -> str:
        # Replace non-alphanumeric chars with underscore to form a safe prefix
        return re.sub(r"[^0-9a-zA-Z]+", "_", s)

    def _prefix_node(node: dict, prefix: str) -> dict:
        # Deep-copy node structure while prefixing node_id and children
        new_node = {
            "node_id": f"{prefix}.{node['node_id']}",
            "title": node.get("title", ""),
            "level": node.get("level", 1),
            "start_page": node.get("start_page"),
            "end_page": node.get("end_page"),
            "raw_text": node.get("raw_text", ""),
            "summary": node.get("summary", ""),
            "children": []
        }

        for child in node.get("children", []):
            new_node["children"].append(_prefix_node(child, prefix))

        return new_node

    combined = []
    for pdf_name in pdf_files:
        stem = os.path.splitext(pdf_name)[0]
        safe = _normalize_prefix(stem)
        # load the per-file tree we just saved (if present) to ensure consistency
        per_path = os.path.join('data', f"{stem}_tree_index.json")
        try:
            with open(per_path, 'r', encoding='utf-8') as f:
                per_tree = json.load(f)
        except Exception:
            # fallback to the in-memory `tree` variable if file missing
            print(f"[TreeIndexer] Warning: could not load {per_path}; skipping for combined")
            continue

        for root in per_tree:
            prefixed = _prefix_node(root, safe)
            # add source metadata at top-level nodes
            prefixed["source_file"] = pdf_name
            combined.append(prefixed)

    if combined:
        combined_path = os.path.join('data', 'tree_index.json')
        save_tree(combined, combined_path)
        print(f"[TreeIndexer] Combined tree saved → {combined_path}")