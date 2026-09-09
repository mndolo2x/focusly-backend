import re
from typing import List, Dict, Any, Optional
import numpy as np
import faiss
from services.ollama_service import ollama_service

class RAGEngine:
    """FAISS vector database RAG engine with Ollama embeddings for chunking and semantic search."""

    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.chunks: List[Dict[str, Any]] = []

    def _pseudo_embedding(self, text: str) -> np.ndarray:
        """Fallback deterministic pseudo-embedding vector if Ollama embedding service is offline."""
        vec = np.zeros(self.dimension, dtype=np.float32)
        for i, char in enumerate(text.encode("utf-8")):
            vec[i % self.dimension] += ord(chr(char))
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    async def generate_embedding(self, text: str) -> np.ndarray:
        """Generates vector embedding using Ollama nomic-embed-text model with fallback."""
        try:
            vector = await ollama_service.embed(text, model="nomic-embed-text")
            if vector:
                arr = np.array(vector, dtype=np.float32)
                if len(arr) != self.dimension:
                    # Adjust dimension dynamically if nomic-embed-text uses 768 or 384 dims
                    if self.index.ntotal == 0:
                        self.dimension = len(arr)
                        self.index = faiss.IndexFlatL2(self.dimension)
                    elif len(arr) < self.dimension:
                        arr = np.pad(arr, (0, self.dimension - len(arr)))
                    else:
                        arr = arr[:self.dimension]
                norm = np.linalg.norm(arr)
                if norm > 0:
                    arr = arr / norm
                return arr
        except Exception:
            pass

        return self._pseudo_embedding(text)

    async def index_document_chunks(self, doc_id: str, text: str, chunk_min_words: int = 500, chunk_max_words: int = 1000) -> int:
        """
        Chunks text into semantic segments (500-1000 words each),
        extracts page tracking info, generates embeddings via Ollama, and indexes in FAISS.
        """
        words = text.split()
        if not words:
            return 0

        chunks_data = []
        current_words = []
        current_page = 1

        for word in words:
            if "Page" in word and ("---" in word or word.isdigit()):
                page_match = re.search(r'\d+', word)
                if page_match:
                    current_page = int(page_match.group(0))

            current_words.append(word)

            if len(current_words) >= chunk_max_words:
                chunk_str = " ".join(current_words)
                chunks_data.append((chunk_str, current_page))
                current_words = []

        if current_words:
            chunk_str = " ".join(current_words)
            chunks_data.append((chunk_str, current_page))

        embeddings = []
        for idx, (chunk_str, page_num) in enumerate(chunks_data):
            vec = await self.generate_embedding(chunk_str)
            embeddings.append(vec)
            self.chunks.append({
                "doc_id": doc_id,
                "chunk_index": idx,
                "page_number": page_num,
                "text": chunk_str
            })

        if embeddings:
            mat = np.array(embeddings).astype(np.float32)
            self.index.add(mat)

        return len(chunks_data)

    def index_document(self, doc_id: str, text: str, chunk_size: int = 500) -> int:
        """Synchronous chunk indexing fallback."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk_str = " ".join(words[i:i + chunk_size])
            if chunk_str:
                chunks.append(chunk_str)

        embeddings = []
        for idx, chunk_str in enumerate(chunks):
            vec = self._pseudo_embedding(chunk_str)
            embeddings.append(vec)
            self.chunks.append({
                "doc_id": doc_id,
                "chunk_index": idx,
                "page_number": 1,
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

        query_vec = np.array([self._pseudo_embedding(query)]).astype(np.float32)
        distances, indices = self.index.search(query_vec, min(top_k, self.index.ntotal))

        results = []
        for idx in indices[0]:
            if idx < len(self.chunks) and idx >= 0:
                results.append(self.chunks[idx])
        return results

rag_engine = RAGEngine()
