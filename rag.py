# tiny RAG - single entry point, hybrid retrieval over the PDF corpus

from pathlib import Path

from loader import load_pdf
from hybrid import HybridRetriever
from generator import answer

PDF = Path(__file__).parent / "data" / "National_Planning_Policy_Framework.pdf"

_retriever = None


def get_retriever():
    global _retriever
    if _retriever is None:
        print("loading and indexing the PDF (about 30 seconds) ...")
        docs = load_pdf(PDF)
        _retriever = HybridRetriever(docs, target_size=300)
        print(f"{len(_retriever.chunks)} chunks indexed\n")
    return _retriever


def ask(question, k=3, show_sources=False):
    r = get_retriever()
    res = answer(r, question, k=k)

    out = res["answer"]
    if show_sources and res["hits"]:
        lines = ["", "Sources:"]
        for i, h in enumerate(res["hits"], start=1):
            lines.append(f"  [{i}] doc {h['doc_id']} (score {h['score']:.3f})")
        out += "\n".join(lines)
    return out


if __name__ == "__main__":
    print("tiny RAG - ask a question, or 'quit' to exit\n")
    get_retriever()  # warm up so the first question isn't slow

    while True:
        q = input("\n> ").strip()
        if q.lower() in {"quit", "exit", "q"}:
            break
        if not q:
            continue
        print()
        print(ask(q, show_sources=True))