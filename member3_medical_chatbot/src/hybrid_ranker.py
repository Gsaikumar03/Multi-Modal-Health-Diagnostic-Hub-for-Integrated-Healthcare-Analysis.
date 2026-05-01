import numpy as np


class HybridRanker:
    def __init__(self):
        pass

    def keyword_score(self, query, chunks):
        scores = []
        query_words = set(query.lower().split())

        for chunk in chunks:
            chunk_words = set(chunk.lower().split())
            overlap = query_words.intersection(chunk_words)
            scores.append(len(overlap))

        return np.array(scores)

    def rerank(self, query, faiss_scores, chunks):
        keyword_scores = self.keyword_score(query, chunks)

        # Normalize
        faiss_scores = (faiss_scores - np.min(faiss_scores)) / (
            np.max(faiss_scores) - np.min(faiss_scores) + 1e-8
        )
        keyword_scores = keyword_scores / (np.max(keyword_scores) + 1e-8)

        # Hybrid score
        final_scores = 0.7 * faiss_scores + 0.3 * keyword_scores

        ranked_indices = np.argsort(-final_scores)

        return ranked_indices, final_scores