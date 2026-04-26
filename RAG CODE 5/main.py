from dotenv import load_dotenv
from graph import build_graph

load_dotenv()

def run_crag(question: str):
    graph = build_graph()
    
    result = graph.invoke({"question": question})
    
    print("\n" + "="*60)
    print("FINAL ANSWER:")
    print("="*60)
    print(result["generation"])
    print(f"\nSources used: {len(result['documents'])} documents")
    for doc in result['documents']:
        src = doc.metadata.get('source', 'vector store')
        doc_type = doc.metadata.get('type', 'local')
        print(f"  [{doc_type}] {src}")
    
    return result

if __name__ == "__main__":
    question = input("Ask a question: ")
    run_crag(question)