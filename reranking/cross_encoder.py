from sentence_transformers import CrossEncoder
from typing import List, Dict, Tuple


class CrossEncoderReranker:
    """Reranks retrieval results using a cross-encoder model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print(f"Loading cross-encoder: {model_name}")
        self.model = CrossEncoder(model_name)
        print("[OK] Cross-encoder loaded")

    def rerank(
        self, query: str, chunks: List[Dict], top_k: int = 5
    ) -> List[Tuple[Dict, float]]:
        """
        Rerank chunks by relevance to query.

        Args:
            query: The search query.
            chunks: List of chunk dicts (must have 'text' field).
            top_k: Number of top results to return.

        Returns:
            List of (chunk, score) tuples sorted by descending score.
        """
        if not chunks:
            return []

        pairs = [(query, chunk["text"]) for chunk in chunks]
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(chunks, list(scores)),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(chunk, float(score)) for chunk, score in ranked[:top_k]]
