import pickle
import faiss
import numpy as np

from .embeddings import EmbeddingModel
from .hybrid_ranker import HybridRanker
from .generator import Generator

class MedicalRAG:
    def __init__(
        self,
        index_path: str = "faiss_index.bin",
        texts_path: str = "texts.pkl",
        top_k: int = 5,
        final_k: int = 3
    ):
        """
        index_path : path to FAISS index
        texts_path : path to stored text chunks
        top_k      : number of initial retrieved chunks
        final_k    : number of chunks after reranking
        """

        # Load FAISS index
        self.index = faiss.read_index(index_path)

        # Load corresponding text chunks
        with open(texts_path, "rb") as f:
            self.texts = pickle.load(f)

        # Components
        self.embedder = EmbeddingModel()
        self.ranker = HybridRanker()
        self.generator = Generator()

        self.top_k = top_k
        self.final_k = final_k

    # -------------------------------------------------------
    # 🔍 RETRIEVAL
    # -------------------------------------------------------
    def retrieve(self, query: str):
        """
        Step 1: Encode query
        Step 2: Retrieve top_k chunks via FAISS
        Step 3: Hybrid rerank
        """

        # Encode query
        query_vec = self.embedder.encode([query])

        # FAISS search
        distances, indices = self.index.search(query_vec, self.top_k)

        retrieved_chunks = [
            self.texts[i] for i in indices[0]
        ]

        # Convert distance to similarity
        similarities = -distances[0]

        # Hybrid reranking
        ranked_indices, scores = self.ranker.rerank(
            query,
            similarities,
            retrieved_chunks
        )

        # Select final top chunks
        final_chunks = [
            retrieved_chunks[i] for i in ranked_indices[:self.final_k]
        ]

        confidence = float(np.mean(scores[:self.final_k]))

        return final_chunks, confidence

    # -------------------------------------------------------
    # 💬 ASK QUESTION
    # -------------------------------------------------------
    def ask(self, query: str, memory_context: str = ""):
        """
        Full RAG Pipeline:
        Retrieval → Context Building → Generation
        """

        # Retrieve context
        context_chunks, confidence = self.retrieve(query)

        # Build full context
        full_context = memory_context + "\n\n".join(context_chunks)

        # Generate answer
        answer = self.generator.generate(query, full_context)

        return {
            "question": query,
            "answer": answer,
            "confidence": confidence,
            "context_used": context_chunks
        }