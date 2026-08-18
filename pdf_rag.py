# RAG over the pdf corpus 

import time
from pathlib import Path
from loader import load_pdf
from retriever import Retriever

PDF = Path(__file__).parent / "data" / "National_Planning_Policy_Framework.pdf"

if __name__ == "__main__":
    print("loading PDF ...")
    docs = load_pdf(PDF)
    print(f"{len(docs)} documents")

    print("chunking and embedding (this will take a minute) ...")
    t0 = time.time()
    r = Retriever(docs, target_size=300)
    elapsed = time.time() - t0

    print(f"{len(r.chunks)} chunks embedded in {elapsed:.1f}s")
    print(f"index shape: {r.vecs.shape}")
    print(f"index size: {r.vecs.nbytes / 1024 / 1024:.1f} MB")

    queries = [
        "What are the requirements for a transport assessment?",
        "How should councils assess housing need?",
        "What is the presumption in favour of sustainable development?",
        "What protections apply to the Green Belt?",
    ]

    for q in queries:
        print(f"\n{'=' * 60}\nQ: {q}")
        for h in r.search(q, k=3):
            print(f"  {h['score']:.3f}  doc={h['doc_id']}")
            print(f"    {h['text'][:120]}")