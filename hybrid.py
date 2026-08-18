# hybrid retrieval - dense + BM25, fused by reciprocal rank

import numpy as np

from chunker import chunk_docs
from embedder import embed_texts
from sparse import SparseIndex
from retriever import SCORE_THRESHOLD


RRF_K = 60


def rrf_ranks(scores, k=RRF_K):
    """Convert a score array into RRF contributions, keyed by index."""
    order = np.argsort(scores)[::-1]
    out = {}
    for rank, idx in enumerate(order, start=1):
        out[int(idx)] = 1.0 / (k + rank)
    return out


class HybridRetriever:
    def __init__(self, docs, target_size=300, overlap_sentences=1):
        self.chunks = chunk_docs(docs, target_size, overlap_sentences)
        texts = [c["text"] for c in self.chunks]
        self.vecs = embed_texts(texts)
        self.sparse = SparseIndex(self.chunks)

    def search(self, question, k=5, dedupe=True):
        q = embed_texts([question])[0]
        dense_scores = self.vecs @ q
        sparse_scores = self.sparse.scores(question)

        dense_rrf = rrf_ranks(dense_scores)
        sparse_rrf = rrf_ranks(sparse_scores)

        fused = {}
        for i in range(len(self.chunks)):
            fused[i] = dense_rrf.get(i, 0.0) + sparse_rrf.get(i, 0.0)

        order = sorted(fused, key=fused.get, reverse=True)

        results = []
        seen = set()
        for i in order:
            doc_id = self.chunks[i]["doc_id"]
            if dedupe and doc_id in seen:
                continue
            seen.add(doc_id)
            results.append({
                "text": self.chunks[i]["text"],
                "doc_id": doc_id,
                "score": float(fused[i]),
                "dense": float(dense_scores[i]),
                "sparse": float(sparse_scores[i]),
            })
            if len(results) == k:
                break
        return results
    
    def search_with_check(self, question, k=5, threshold=SCORE_THRESHOLD):
        hits = self.search(question, k)
        best_dense = max((h["dense"] for h in hits), default=0.0)
        good = [h for h in hits if h["dense"] >= threshold]
        return {
            "hits": good if good else hits[:1],
            "all_hits": hits,
            "sufficient": best_dense >= threshold,
            "best_score": best_dense,
        }