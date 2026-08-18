# recalibrate the dense score threshold against the NPPF corpus

from pathlib import Path

import numpy as np

from loader import load_pdf
from hybrid import HybridRetriever
from testset import TESTSET

PDF = Path(__file__).parent / "data" / "National_Planning_Policy_Framework.pdf"

OUT_OF_SCOPE = [
    "What is the capital of France?",
    "How do I train a neural network?",
    "What is the best way to make sourdough bread?",
    "Who won the World Cup in 2018?",
    "How do I fix a leaking tap?",
    "What are the symptoms of flu?",
]

if __name__ == "__main__":
    docs = load_pdf(PDF)
    r = HybridRetriever(docs, target_size=300)

    in_vals = []
    print("IN SCOPE (best dense score)")
    for item in TESTSET:
        hits = r.search(item["q"], k=5)
        best = max(h["dense"] for h in hits)
        in_vals.append(best)
        print(f"  {best:.3f}  {item['q'][:55]}")

    off_vals = []
    print("\nOUT OF SCOPE")
    for q in OUT_OF_SCOPE:
        hits = r.search(q, k=5)
        best = max(h["dense"] for h in hits)
        off_vals.append(best)
        print(f"  {best:.3f}  {q}")

    in_vals = np.array(in_vals)
    off_vals = np.array(off_vals)

    print(f"\nin-scope  min={in_vals.min():.3f}  mean={in_vals.mean():.3f}")
    print(f"off-scope max={off_vals.max():.3f}  mean={off_vals.mean():.3f}")

    if in_vals.min() > off_vals.max():
        mid = (in_vals.min() + off_vals.max()) / 2
        print(f"\nclean separation -> threshold around {mid:.2f}")
    else:
        print("\nOVERLAP - no clean threshold")