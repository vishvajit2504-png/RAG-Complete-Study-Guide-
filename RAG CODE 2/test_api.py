# test_api.py
import requests
import json
import os

BASE_URL = "http://localhost:8000"

def print_section(title: str):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")


def test_health():
    print_section("1. Health Check")
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200, f"Health check failed: {response.text}"
    data = response.json()
    print(f"Status  : {data['status']}")
    print(f"Model   : {data['model']}")
    print("PASSED ✅")


def test_ingest(folder_path):
    print_section("2. Ingest Documents")
    response = requests.post(
        f"{BASE_URL}/ingest",
        params={"folder_path": folder_path}
    )
    assert response.status_code == 200, f"Ingestion failed: {response.text}"
    data = response.json()
    print(f"Message          : {data['message']}")
    print(f"Documents loaded : {data['documents_loaded']}")
    print(f"Chunks created   : {data['chunks_created']}")
    assert data["chunks_created"] > 0, "No chunks created — check sample_docs folder"
    print("PASSED ✅")


def test_query(question: str, mode: str = "similarity"):
    print_section(f"3. Query — '{question[:50]}'")
    payload = {"question": question, "mode": mode}
    response = requests.post(
        f"{BASE_URL}/query",
        json=payload
    )
    assert response.status_code == 200, f"Query failed: {response.text}"
    data = response.json()
    print(f"Answer      : {data['answer']}")
    print(f"Sources     : {data['sources']}")
    print(f"Chunks used : {data['chunks_used']}")
    print("PASSED ✅")
    return data


def test_empty_question():
    print_section("4. Edge Case — Empty Question")
    payload = {"question": "  "}
    response = requests.post(f"{BASE_URL}/query", json=payload)
    assert response.status_code == 400, f"Expected 400, got {response.status_code}"
    print(f"Correctly rejected with 400: {response.json()['detail']}")
    print("PASSED ✅")


def test_out_of_domain_query():
    print_section("5. Edge Case — Out of Domain Query")
    payload = {"question": "What is the recipe for biryani?"}
    response = requests.post(f"{BASE_URL}/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    print(f"Answer      : {data['answer']}")
    print(f"Chunks used : {data['chunks_used']}")
    # Should return fallback answer with 0 chunks
    print("PASSED ✅")


def test_upload(file_path: str):
    print_section("6. File Upload")
    if not os.path.exists(file_path):
        print(f"Skipping — file not found: {file_path}")
        return

    with open(file_path, "rb") as f:
        filename = os.path.basename(file_path)
        ext      = os.path.splitext(filename)[1].lower()
        mime_map = {
            ".pdf":  "application/pdf",
            ".txt":  "text/plain",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        }
        mime = mime_map.get(ext, "application/octet-stream")
        response = requests.post(
            f"{BASE_URL}/upload",
            files={"file": (filename, f, mime)}
        )

    assert response.status_code == 200, f"Upload failed: {response.text}"
    data = response.json()
    print(f"Message       : {data['message']}")
    print(f"Chunks created: {data['chunks_created']}")
    print("PASSED ✅")


def test_reset():
    print_section("7. Reset Vector Store")
    response = requests.delete(f"{BASE_URL}/reset")
    assert response.status_code == 200, f"Reset failed: {response.text}"
    print(f"Message: {response.json()['message']}")
    print("PASSED ✅")


def run_all():
    print("\n🚀 Running End-to-End RAG Pipeline Tests")
    print("Make sure: ollama serve + uvicorn main:app are running\n")

    try:
        test_health()
        docs_path = os.path.join(os.path.dirname(__file__), "papers")
        test_ingest(docs_path)
        test_query("what is attention mechanisms", mode="mmr")
        test_query("Summarize the key findings.", mode="mmr")
        test_empty_question()
        test_out_of_domain_query()
        test_upload("./sample_docs/test.pdf")   # swap with a real file
        # test_reset()

        print(f"\n{'='*50}")
        print("  ALL TESTS PASSED ✅")
        print(f"{'='*50}\n")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
    except requests.ConnectionError:
        print("\n❌ Cannot connect to API — is uvicorn running on port 8000?")


if __name__ == "__main__":
    run_all()