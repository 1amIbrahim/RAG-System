from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple
import re

class BM25Index:
    """BM25 keyword-based search"""
    
    def __init__(self):
        self.index = None
        self.chunks = []
    
    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization (you can improve this)"""
        text = text.lower()
        tokens = re.findall(r'\w+', text)
        return tokens
    
    def build_index(self, chunks: List[Dict]):
        """
        Build BM25 index from chunks
        
        Args:
            chunks: List of chunk dicts with 'text' field
        """
        print(f"Building BM25 index for {len(chunks)} chunks...")
        
        self.chunks = chunks
        tokenized_chunks = [self.tokenize(chunk["text"]) for chunk in chunks]
        
        self.index = BM25Okapi(tokenized_chunks)
        print(f"✅ BM25 index built")
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Search using BM25
        
        Args:
            query: Query string
            top_k: Number of results
            
        Returns:
            List of (chunk, score) tuples
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        tokenized_query = self.tokenize(query)
        scores = self.index.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = [(self.chunks[idx], float(scores[idx])) for idx in top_indices]
        return results