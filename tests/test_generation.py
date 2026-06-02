import json
from pathlib import Path

from retrieval.embeddings import EmbeddingModel
from retrieval.vector import VectorIndex
from retrieval.bm25 import BM25Index
from retrieval.hybrid import HybridRetriever
from reranking.cross_encoder import CrossEncoderReranker
from generation.llm import answer


def test_generation():
    chunks_path = Path("data/processed/sample_chunks.json")
    with open(chunks_path, "r") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks\n")

    # Build retrieval indices
    embed_model = EmbeddingModel()
    vector_index = VectorIndex(embed_model)
    vector_index.build_index(chunks)

    bm25_index = BM25Index()
    bm25_index.build_index(chunks)

    hybrid = HybridRetriever(vector_index, bm25_index)

    # Retrieve
    query = "What is artificial intelligence?"
    print(f"Query: {query}\n")

    retrieved = hybrid.search(query, top_k=5)
    retrieved_chunks = [chunk for chunk, _ in retrieved]

    # Rerank
    reranker = CrossEncoderReranker()
    reranked = reranker.rerank(query, retrieved_chunks, top_k=3)
    top_chunks = [chunk for chunk, _ in reranked]

    print(f"Top {len(top_chunks)} chunks after reranking:")
    for i, chunk in enumerate(top_chunks, 1):
        print(f"  [{i}] {chunk['source']} p.{chunk['page']} — {chunk['text'][:80]}...")

    # Generate
    print("\nGenerating answer...")
    result = answer(query, top_chunks)

    print(f"\n=== Answer ===\n{result['answer']}")
    print(f"\nSources: {result['sources']}")

    assert isinstance(result["answer"], str) and len(result["answer"]) > 0, \
        "Answer must be a non-empty string"
    assert len(result["sources"]) > 0, "Sources list must not be empty"

    print("\n✅ Generation test passed!")


if __name__ == "__main__":
    test_generation()
