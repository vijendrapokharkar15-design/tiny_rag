# BM25 keyword retrieval - the sparse half of hybrid search

import re
from rank_bm25 import BM25Okapi

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class SparseIndex:
    def __init__(self, chunks):
        self.chunks = chunks
        corpus = [tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(corpus)

    def scores(self, question):
        return self.bm25.get_scores(tokenize(question))
    


if __name__ == "__main__":
    from pathlib import Path

    import numpy as np

    from loader import load_pdf
    from chunker import chunk_docs

    PDF = Path(__file__).parent / "data" / "National_Planning_Policy_Framework.pdf"

    docs = load_pdf(PDF)
    chunks = chunk_docs(docs, target_size=300)
    print(f"{len(chunks)} chunks indexed")

    idx = SparseIndex(chunks)

    q = "What is the presumption in favour of sustainable development?"
    s = idx.scores(q)
    top = np.argsort(s)[::-1][:5]

    print(f"\nQ: {q}\n")
    for i in top:
        print(f"  {s[i]:.2f}  doc={chunks[i]['doc_id']}")
        print(f"    {chunks[i]['text'][:100]}")