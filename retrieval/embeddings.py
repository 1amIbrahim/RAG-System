from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class EmbeddingModel:
    """Wraps sentence-transformers for embedding generation"""
    
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        """
        Args:
            model_name: HuggingFace model ID
                       bge-small-en-v1.5 = fast, good quality
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"[OK] Model loaded. Embedding dimension: {self.dimension}")
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts
        
        Args:
            texts: List of strings to embed
            
        Returns:
            numpy array of shape (len(texts), dimension)
        """
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # Important for cosine similarity
        )
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query"""
        return self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )