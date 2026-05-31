import faiss
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple
from .embeddings import EmbeddingModel

class VectorIndex:
    """FAISS-based vector similarity search"""
    
    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
        self.index = None
        self.chunks = []
        self.dimension = embedding_model.dimension
    
    def build_index(self, chunks: List[Dict]):
        """
        Build FAISS index from chunks
        
        Args:
            chunks: List of chunk dicts with 'text' field
        """
        print(f"Building FAISS index for {len(chunks)} chunks...")
        
        self.chunks = chunks
        texts = [chunk["text"] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embedding_model.embed_texts(texts)
        
        # Ensure embeddings are normalized
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        
        # Create FAISS index - use L2 for normalized vectors
        self.index = faiss.IndexFlatL2(self.dimension)  # ← CHANGED
        self.index.add(embeddings.astype('float32'))
        
        print(f"✅ FAISS index built with {self.index.ntotal} vectors")
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Search for most similar chunks
        
        Args:
            query: Query string
            top_k: Number of results
            
        Returns:
            List of (chunk, score) tuples
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Embed query
        query_embedding = self.embedding_model.embed_query(query)
        
        # Normalize query embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Search (returns L2 distances)
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Convert L2 distance to similarity score
        # L2 distance for normalized vectors: d = 2 - 2*cos(θ)
        # So: similarity = 1 - d/2
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks) and idx != -1:  # Valid index
                similarity = 1 - (distance / 2)  # ← CONVERT TO SIMILARITY
                similarity = max(0.0, min(1.0, similarity))  # Clamp to [0,1]
                results.append((self.chunks[idx], float(similarity)))
        
        return results
    
    def save(self, index_path: Path, chunks_path: Path):
        """Save index and chunks to disk"""
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Index saved to {index_path}")
    
    def load(self, index_path: Path, chunks_path: Path):
        """Load index and chunks from disk"""
        self.index = faiss.read_index(str(index_path))
        
        with open(chunks_path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        
        print(f"✅ Index loaded with {len(self.chunks)} chunks")