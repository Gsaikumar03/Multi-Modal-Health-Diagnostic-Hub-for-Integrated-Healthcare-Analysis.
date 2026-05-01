import faiss
import pickle
import numpy as np


class Retriever:
    def __init__(self, index_path="faiss_index.bin", meta_path="texts.pkl"):
        self.index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            self.text_chunks = pickle.load(f)

    def search(self, query_embedding, top_k=3):
        distances, indices = self.index.search(
            query_embedding.astype("float32"), top_k
        )
        return [self.text_chunks[i] for i in indices[0]]