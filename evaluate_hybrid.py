# same metrics, hybrid retriever - compare against the dense baseline

from pathlib import Path

from loader import load_pdf
from hybrid import HybridRetriever
from testset import TESTSET
from evaluate import evaluate

PDF = Path(__file__).parent / "data" / "National_Planning_Policy_Framework.pdf"

if __name__ == "__main__":
    docs = load_pdf(PDF)
    print(f"{len(docs)} documents - building hybrid index ...")
    r = HybridRetriever(docs, target_size=300)
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
    print(f"recall@{K}:   {recall:.3f}   (dense baseline: 0.750)")
    print(f"MRR:        {mrr:.3f}   (dense baseline: 0.521)")