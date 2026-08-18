# measures retrieval quality against the labelled test set

from pathlib import Path
from loader import load_pdf
from retriever import Retriever
from testset import TESTSET

PDF = Path(__file__).parent / "data" / "National_Planning_Policy_Framework.pdf"


def recall_at_k(retrieved_docs, gold_docs):
    return 1.0 if set(retrieved_docs) & set(gold_docs) else 0.0


def reciprocal_rank(retrieved_docs, gold_docs):
    gold = set(gold_docs)
    for rank, d in enumerate(retrieved_docs, start=1):
        if d in gold:
            return 1.0 / rank
    return 0.0


def evaluate(retriever, testset, k=5):
    rows = []
    for item in testset:
        hits = retriever.search(item["q"], k=k)
        got = [h["doc_id"] for h in hits]
        rows.append({
            "q": item["q"],
            "gold": item["docs"],
            "got": got,
            "recall": recall_at_k(got, item["docs"]),
            "rr": reciprocal_rank(got, item["docs"]),
        })
    return rows


if __name__ == "__main__":
    docs = load_pdf(PDF)
    print(f"{len(docs)} documents - building index ...")
    r = Retriever(docs, target_size=300)
    print(f"{len(r.chunks)} chunks\n")

    K = 5
    rows = evaluate(r, TESTSET, k=K)

    for row in rows:
        mark = "HIT " if row["recall"] else "MISS"
        rank = f"rank {int(1/row['rr'])}" if row["rr"] else "not found"
        print(f"{mark} {rank:>10}  {row['q'][:60]}")
        if not row["recall"]:
            print(f"       gold={row['gold']}  got={row['got']}")

    n = len(rows)
    recall = sum(r_["recall"] for r_ in rows) / n
    mrr = sum(r_["rr"] for r_ in rows) / n

    print(f"\n{'=' * 55}")
    print(f"questions:  {n}")
    print(f"recall@{K}:   {recall:.3f}")
    print(f"MRR:        {mrr:.3f}")
