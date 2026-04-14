# test_api_live.py
import requests
import json
import os
import time
from config import get_settings


BASE_URL  = "http://localhost:8001"
path = os.path.dirname(os.path.abspath(__file__))
PDF_PATH  = os.path.join(path, "papers/1810.04805v2.pdf")
CACHE_DIR = "data/cache"
MODEL     = get_settings().ollama_chat_model

# ── adjust these to match your PDF content ───────────────────────────────────
QUERY_1 = "What is the main topic of this document?"
QUERY_2 = "What are the key findings?"
QUERY_3 = "Summarize the introduction."


def print_section(title: str):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")


# ══════════════════════════════════════════════════════════
# 1. Health Check
# ══════════════════════════════════════════════════════════

def test_health():
    print_section("1. Health Check")
    r = requests.get(f"{BASE_URL}/health", timeout=10)
    assert r.status_code == 200, f"Health check failed: {r.text}"
    data = r.json()
    print(f"Status  : {data['status']}")
    print(f"Version : {data['version']}")
    print("PASSED ✅")


# ══════════════════════════════════════════════════════════
# 2. Index — fresh document
# ══════════════════════════════════════════════════════════

def test_index_fresh():
    print_section("2. Index — Fresh Document")
    r = requests.post(f"{BASE_URL}/index", json={
        "pdf_path":      PDF_PATH,
        "model":         MODEL,
        "cache_dir":     CACHE_DIR,
        "force_reindex": True          # always fresh for this test
    }, timeout=1000)

    assert r.status_code == 200, f"Indexing failed: {r.text}"
    data = r.json()
    print(f"Status       : {data['status']}")
    print(f"Total nodes  : {data['total_nodes']}")
    print(f"Cache paths  : {json.dumps(data['cache_paths'], indent=14)}")
    assert data["total_nodes"] > 0, "No nodes built — check your PDF"

    # Verify cache files actually exist on disk
    for name, path in data["cache_paths"].items():
        assert os.path.exists(path), f"Cache file missing: {name} → {path}"
        print(f"  ✓ {name} exists on disk")

    print("PASSED ✅")
    return data


# ══════════════════════════════════════════════════════════
# 3. Index — cache hit (second call)
# ══════════════════════════════════════════════════════════

def test_index_cached():
    print_section("3. Index — Cache Hit (Second Call)")
    start = time.time()
    r = requests.post(f"{BASE_URL}/index", json={
        "pdf_path":  PDF_PATH,
        "cache_dir": CACHE_DIR
    }, timeout=30)
    elapsed = time.time() - start

    assert r.status_code == 200, f"Cache call failed: {r.text}"
    data = r.json()
    print(f"Status   : {data['status']}")
    print(f"Elapsed  : {elapsed:.2f}s")
    assert data["status"] == "cached", "Expected cached on second call"
    assert elapsed < 5, f"Cache hit took {elapsed:.1f}s — too slow"
    print("PASSED ✅")


# ══════════════════════════════════════════════════════════
# 4. Index — missing PDF (404)
# ══════════════════════════════════════════════════════════

def test_index_missing_pdf():
    print_section("4. Index — Missing PDF (404)")
    r = requests.post(f"{BASE_URL}/index", json={
        "pdf_path":  "data/docs/ghost_file.pdf",
        "cache_dir": CACHE_DIR
    }, timeout=10)
    print(f"Status  : {r.status_code}")
    print(f"Detail  : {r.json().get('detail')}")
    assert r.status_code == 404, f"Expected 404, got {r.status_code}"
    print("PASSED ✅")


# ══════════════════════════════════════════════════════════
# 5. Query — basic happy path
# ══════════════════════════════════════════════════════════

def test_query_basic():
    print_section(f"5. Query — '{QUERY_1[:45]}'")
    r = requests.post(f"{BASE_URL}/query", json={
        "pdf_path":  PDF_PATH,
        "query":     QUERY_1,
        "cache_dir": CACHE_DIR
    }, timeout=300)

    assert r.status_code == 200, f"Query failed: {r.text}"
    data = r.json()
    print(f"Answer        : {data['answer'][:300]}")
    print(f"Sections used : {data['metadata']['total_sections']}")
    print(f"Pages covered : {data['metadata']['total_pages_covered']}")
    print(f"Est. tokens   : {data['metadata']['estimated_tokens']}")
    assert len(data["answer"].strip()) > 0, "Empty answer returned"
    assert data["metadata"]["total_sections"] > 0
    print("PASSED ✅")
    return data


# ══════════════════════════════════════════════════════════
# 6. Query — retrieval trace
# ══════════════════════════════════════════════════════════

def test_query_trace():
    print_section(f"6. Query — Retrieval Trace Validation")
    r = requests.post(f"{BASE_URL}/query", json={
        "pdf_path":  PDF_PATH,
        "query":     QUERY_2,
        "cache_dir": CACHE_DIR
    }, timeout=300)

    assert r.status_code == 200, f"Query failed: {r.text}"
    trace = r.json()["trace"]

    print(f"Selected IDs  : {trace['selected_ids']}")
    print("Retrieved sections:")
    for s in trace["retrieved_sections"]:
        print(f"  [{s['node_id']}] {s['title']} — pages {s['page_range']}")

    assert "selected_ids"       in trace
    assert "retrieved_sections" in trace
    assert len(trace["retrieved_sections"]) > 0

    for section in trace["retrieved_sections"]:
        assert "node_id"    in section
        assert "title"      in section
        assert "page_range" in section

    print("PASSED ✅")


# ══════════════════════════════════════════════════════════
# 7. Query — max_nodes respected
# ══════════════════════════════════════════════════════════

def test_query_max_nodes():
    print_section("7. Query — max_nodes=2 Respected")
    r = requests.post(f"{BASE_URL}/query", json={
        "pdf_path":  PDF_PATH,
        "query":     QUERY_1,
        "cache_dir": CACHE_DIR,
        "max_nodes": 2
    }, timeout=300)

    assert r.status_code == 200, f"Query failed: {r.text}"
    sections = r.json()["trace"]["retrieved_sections"]
    print(f"Sections returned : {len(sections)} (max allowed: 2)")
    assert len(sections) <= 2, f"Got {len(sections)} sections — expected ≤ 2"
    print("PASSED ✅")


# ══════════════════════════════════════════════════════════
# 8. Query — multiple queries, all succeed
# ══════════════════════════════════════════════════════════

def test_query_multiple():
    print_section("8. Query — Multiple Queries")
    queries = [QUERY_1, QUERY_2, QUERY_3]
    answers = []

    for i, q in enumerate(queries, 1):
        r = requests.post(f"{BASE_URL}/query", json={
            "pdf_path":  PDF_PATH,
            "query":     q,
            "cache_dir": CACHE_DIR
        }, timeout=300)
        assert r.status_code == 200, f"Query {i} failed: {r.text}"
        answer = r.json()["answer"]
        answers.append(answer)
        print(f"  Q{i}: {q[:45]}")
        print(f"  A{i}: {answer[:120]}...\n")

    # Different queries should produce different answers
    assert answers[0] != answers[1], "Q1 and Q2 returned identical answers"
    print("PASSED ✅")


# ══════════════════════════════════════════════════════════
# 9. Query — unindexed document (404)
# ══════════════════════════════════════════════════════════

def test_query_unindexed():
    print_section("9. Query — Unindexed Document (404)")
    r = requests.post(f"{BASE_URL}/query", json={
        "pdf_path":  "data/docs/never_indexed.pdf",
        "query":     QUERY_1,
        "cache_dir": CACHE_DIR
    }, timeout=10)
    print(f"Status  : {r.status_code}")
    print(f"Detail  : {r.json().get('detail')}")
    assert r.status_code == 404, f"Expected 404, got {r.status_code}"
    print("PASSED ✅")


# ══════════════════════════════════════════════════════════
# 10. Full pipeline — one shot
# ══════════════════════════════════════════════════════════

def test_full_pipeline():
    print_section("10. Full Pipeline — /query/full")
    r = requests.post(f"{BASE_URL}/query/full", json={
        "pdf_path":  PDF_PATH,
        "query":     QUERY_1,
        "cache_dir": CACHE_DIR
    }, timeout=300)

    assert r.status_code == 200, f"Full pipeline failed: {r.text}"
    data = r.json()
    print(f"Answer        : {data['answer'][:300]}")
    print(f"Sections used : {data['metadata']['total_sections']}")
    print(f"Pages covered : {data['metadata']['total_pages_covered']}")

    for field in ["query", "answer", "metadata", "trace"]:
        assert field in data, f"Missing field: {field}"

    assert len(data["answer"].strip()) > 0
    print("PASSED ✅")


# ══════════════════════════════════════════════════════════
# 11. Full pipeline — missing PDF (404)
# ══════════════════════════════════════════════════════════

def test_full_pipeline_missing_pdf():
    print_section("11. Full Pipeline — Missing PDF (404)")
    r = requests.post(f"{BASE_URL}/query/full", json={
        "pdf_path":  "data/docs/ghost.pdf",
        "query":     QUERY_1,
        "cache_dir": CACHE_DIR
    }, timeout=10)
    print(f"Status  : {r.status_code}")
    print(f"Detail  : {r.json().get('detail')}")
    assert r.status_code == 404, f"Expected 404, got {r.status_code}"
    print("PASSED ✅")


# ══════════════════════════════════════════════════════════
# 12. Full pipeline — cache speedup
# ══════════════════════════════════════════════════════════

def test_full_pipeline_cache_speedup():
    print_section("12. Full Pipeline — Cache Speedup")
    payload = {
        "pdf_path":  PDF_PATH,
        "query":     QUERY_1,
        "cache_dir": CACHE_DIR
    }

    start1 = time.time()
    requests.post(f"{BASE_URL}/query/full", json=payload, timeout=300)
    time1 = time.time() - start1

    start2 = time.time()
    r2     = requests.post(f"{BASE_URL}/query/full", json=payload, timeout=300)
    time2  = time.time() - start2

    print(f"First call  : {time1:.1f}s  (index + query)")
    print(f"Second call : {time2:.1f}s  (cache + query)")
    print(f"Speedup     : {time1/max(time2,1):.1f}x")

    assert r2.status_code == 200
    assert time2 < time1, "Second call should be faster than first"
    print("PASSED ✅")


# ══════════════════════════════════════════════════════════
# Run all
# ══════════════════════════════════════════════════════════

def run_all():
    print("\n🚀 Vectorless RAG — Phase 3 Live API Tests")
    print(f"   API      : {BASE_URL}")
    print(f"   PDF      : {PDF_PATH}")
    print(f"   Model    : {MODEL}")
    print("   Make sure: ollama serve + python main.py are running\n")

    # Pre-flight
    if not os.path.exists(PDF_PATH):
        print(f"❌ PDF not found: {PDF_PATH}")
        print("   Add a PDF at that path and re-run.")
        return

    try:
        requests.get(f"{BASE_URL}/health", timeout=5)
    except requests.ConnectionError:
        print(f"❌ Cannot reach API at {BASE_URL}")
        print("   Start it with: python main.py")
        return

    passed = 0
    failed = 0

    tests = [
        test_health,
        test_index_fresh,
        test_index_cached,
        test_index_missing_pdf,
        test_query_basic,
        test_query_trace,
        test_query_max_nodes,
        test_query_multiple,
        test_query_unindexed,
        test_full_pipeline,
        test_full_pipeline_missing_pdf,
        test_full_pipeline_cache_speedup,
    ]

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"\n❌ FAILED: {e}")
            failed += 1
        except requests.ConnectionError:
            print(f"\n❌ Connection lost during {test_fn.__name__}")
            failed += 1

    print(f"\n{'='*55}")
    print(f"  Results : {passed} passed  |  {failed} failed")
    print(f"  {'ALL TESTS PASSED ✅' if failed == 0 else 'SOME TESTS FAILED ❌'}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    run_all()