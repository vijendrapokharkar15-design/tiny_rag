# finds the chunks most similar to a question

import numpy as np

from chunker import chunk_docs
from embedder import embed_texts

# derived empirically in calibrate_pdf.py against the NPPF corpus
# in-scope min 0.644, out-of-scope max 0.207 - midpoint 0.43
# NOTE: calibrated on 8 clearly-answerable questions, so vaguer real
# questions may score below this and get refused
SCORE_THRESHOLD = 0.43


class Retriever:
    def __init__(self, docs, target_size=300, overlap_sentences=1):
        self.chunks = chunk_docs(docs, target_size, overlap_sentences)
        texts = [c["text"] for c in self.chunks]
        self.vecs = embed_texts(texts)

    def search(self, question, k=3, dedupe=True):
        q = embed_texts([question])[0]
        scores = self.vecs @ q
        order = np.argsort(scores)[::-1]

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
                "score": float(scores[i]),
            })
            if len(results) == k:
                break
        return results
    

    def search_with_check(self, question, k=3, threshold=SCORE_THRESHOLD):
        hits = self.search(question, k)
        good = [h for h in hits if h["score"] >= threshold]
        return {
            "hits": good,
            "all_hits": hits,
            "sufficient": len(good) > 0,
            "best_score": hits[0]["score"] if hits else 0.0,
        }


if __name__ == "__main__":
    from corpus import docs, questions, out_of_scope

    r = Retriever(docs)

    for q in questions:
        print(f"\nQ: {q}")
        for hit in r.search(q, k=2):
            print(f"  doc={hit['doc_id']} score={hit['score']:.3f}")
            print(f"    {hit['text'][:70]}...")

    print("\n--- with threshold check ---")
    for q in questions + out_of_scope:
        res = r.search_with_check(q, k=2)
        flag = "OK " if res["sufficient"] else "MISS"
        print(f"{flag} {res['best_score']:.3f}  {q}")