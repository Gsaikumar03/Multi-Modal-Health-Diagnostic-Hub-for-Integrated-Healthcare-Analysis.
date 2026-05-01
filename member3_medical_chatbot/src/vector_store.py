import faiss
import numpy as np
import pickle


class FAISSVectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.text_chunks = []

    def add_embeddings(self, embeddings: np.ndarray, texts: list):
        self.index.add(embeddings.astype("float32"))
        self.text_chunks.extend(texts)

    def search(self, query_embedding: np.ndarray, top_k: int = 3):
        distances, indices = self.index.search(
            query_embedding.astype("float32"), top_k
        )
        results = [self.text_chunks[i] for i in indices[0]]
        return results

    def save(self, index_path="faiss_index.bin", meta_path="texts.pkl"):
        faiss.write_index(self.index, index_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.text_chunks, f)
