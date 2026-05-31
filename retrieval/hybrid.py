from typing import List, Dict, Tuple
from .vector import VectorIndex
from .bm25 import BM25Index

class HybridRetriever:
    """Combines vector and BM25 search"""
    
    def __init__(
        self,
        vector_index: VectorIndex,
        bm25_index: BM25Index,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3
    ):
        self.vector_index = vector_index
        self.bm25_index = bm25_index
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Hybrid search with score fusion
        
        Args:
            query: Query string
            top_k: Number of final results
            
        Returns:
            List of (chunk, combined_score) tuples
        """
        # Get results from both indices
        vector_results = self.vector_index.search(query, top_k=top_k*2)
        bm25_results = self.bm25_index.search(query, top_k=top_k*2)
        
        # Normalize and combine scores
        combined_scores = {}
        
        # Add vector scores
        if vector_results:
            max_vector_score = max(score for _, score in vector_results)
            for chunk, score in vector_results:
                chunk_id = chunk["chunk_id"]
                normalized = score / max_vector_score if max_vector_score > 0 else 0
                combined_scores[chunk_id] = {
                    "chunk": chunk,
                    "score": normalized * self.vector_weight
                }
        
        # Add BM25 scores
        if bm25_results:
            max_bm25_score = max(score for _, score in bm25_results)
            for chunk, score in bm25_results:
                chunk_id = chunk["chunk_id"]
                normalized = score / max_bm25_score if max_bm25_score > 0 else 0
                
                if chunk_id in combined_scores:
                    combined_scores[chunk_id]["score"] += normalized * self.bm25_weight
                else:
                    combined_scores[chunk_id] = {
                        "chunk": chunk,
                        "score": normalized * self.bm25_weight
                    }
        
        # Sort by combined score
        ranked = sorted(
            combined_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )[:top_k]
        
        return [(item["chunk"], item["score"]) for item in ranked]