from typing import List, Dict, Any
import numpy as np
import faiss

class RAGEngine:
    """FAISS vector database RAG engine for chunking and semantic text search."""

    def __init__(self, dimension: int = 128):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.chunks: List[Dict[str, Any]] = []

    def _simple_embedding(self, text: str) -> np.ndarray:
        """Deterministic pseudo-embedding for local chunk indexing without external cloud calls."""
        vec = np.zeros(self.dimension, dtype=np.float32)
        for i, char in enumerate(text.encode("utf-8")):
            vec[i % self.dimension] += ord(chr(char))
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def index_document(self, doc_id: str, text: str, chunk_size: int = 500) -> int:
        """Chunks text and adds vectors to FAISS index."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk_str = " ".join(words[i:i + chunk_size])
            if chunk_str:
                chunks.append(chunk_str)

        embeddings = []
        for idx, chunk_str in enumerate(chunks):
            vec = self._simple_embedding(chunk_str)
            embeddings.append(vec)
            self.chunks.append({
                "doc_id": doc_id,
                "chunk_index": idx,
                "text": chunk_str
            })

        if embeddings:
            mat = np.array(embeddings).astype(np.float32)
            self.index.add(mat)

        return len(chunks)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Searches indexed document chunks by query similarity."""
        if self.index.ntotal == 0:
            return []

        query_vec = np.array([self._simple_embedding(query)]).astype(np.float32)
        distances, indices = self.index.search(query_vec, min(top_k, self.index.ntotal))

        results = []
        for idx in indices[0]:
            if idx < len(self.chunks) and idx >= 0:
                results.append(self.chunks[idx])
        return results

rag_engine = RAGEngine()
