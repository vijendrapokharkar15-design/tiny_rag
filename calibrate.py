# compares score distributions for answerable vs unanswerable questions

import numpy as np

from corpus import docs, questions, out_of_scope
from retriever import Retriever


def top_scores(r, qs):
    out = []
    for q in qs:
        hits = r.search(q, k=1)
        out.append((q, hits[0]["score"], hits[0]["doc_id"]))
    return out


if __name__ == "__main__":
    r = Retriever(docs)

    in_scope = top_scores(r, questions)
    off_scope = top_scores(r, out_of_scope)

    print("IN SCOPE (should score high)")
    for q, s, d in in_scope:
        print(f"  {s:.3f}  doc={d}  {q}")

    print("\nOUT OF SCOPE (should score low)")
    for q, s, d in off_scope:
        print(f"  {s:.3f}  doc={d}  {q}")

    in_vals = np.array([s for _, s, _ in in_scope])
    off_vals = np.array([s for _, s, _ in off_scope])

    print(f"\nin-scope  min={in_vals.min():.3f}  mean={in_vals.mean():.3f}")
    print(f"off-scope max={off_vals.max():.3f}  mean={off_vals.mean():.3f}")
    print(f"\ngap between in-scope min and off-scope max: "
          f"{in_vals.min() - off_vals.max():+.3f}")

    if in_vals.min() > off_vals.max():
        mid = (in_vals.min() + off_vals.max()) / 2
        print(f"clean separation -> threshold around {mid:.2f}")
    else:
        print("OVERLAP - no single threshold separates these cleanly")