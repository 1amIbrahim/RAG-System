from pathlib import Path
import json
from retrieval.embeddings import EmbeddingModel
from retrieval.vector import VectorIndex
from retrieval.bm25 import BM25Index
from retrieval.hybrid import HybridRetriever

def test_retrieval():
    # Load chunks
    chunks_path = Path("data/processed/sample_chunks.json")
    with open(chunks_path, 'r') as f:
        chunks = json.load(f)
    
    print(f"Loaded {len(chunks)} chunks\n")
    
    # Build indices
    print("=== Building Embedding Model ===")
    embed_model = EmbeddingModel()
    
    print("\n=== Building Vector Index ===")
    vector_index = VectorIndex(embed_model)
    vector_index.build_index(chunks)
    
    print("\n=== Building BM25 Index ===")
    bm25_index = BM25Index()
    bm25_index.build_index(chunks)
    
    # Test query
    query = "What is artificial intelligence?"
    
    print(f"\n=== Query: '{query}' ===\n")
    
    # Vector search
    print("--- Vector Search ---")
    vector_results = vector_index.search(query, top_k=3)
    for i, (chunk, score) in enumerate(vector_results, 1):
        print(f"{i}. Score: {score:.3f}")
        print(f"   Source: {chunk['source']}, Page: {chunk['page']}")
        print(f"   Text: {chunk['text'][:100]}...\n")
    
    # BM25 search
    print("--- BM25 Search ---")
    bm25_results = bm25_index.search(query, top_k=3)
    for i, (chunk, score) in enumerate(bm25_results, 1):
        print(f"{i}. Score: {score:.3f}")
        print(f"   Source: {chunk['source']}, Page: {chunk['page']}")
        print(f"   Text: {chunk['text'][:100]}...\n")
    
    # Hybrid search
    print("--- Hybrid Search ---")
    hybrid = HybridRetriever(vector_index, bm25_index)
    hybrid_results = hybrid.search(query, top_k=3)
    for i, (chunk, score) in enumerate(hybrid_results, 1):
        print(f"{i}. Score: {score:.3f}")
        print(f"   Source: {chunk['source']}, Page: {chunk['page']}")
        print(f"   Text: {chunk['text'][:100]}...\n")
    
    print("✅ Retrieval system works!")

if __name__ == "__main__":
    test_retrieval()