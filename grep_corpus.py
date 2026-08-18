# keyword search over the loaded documents, for building the test set

import sys
from pathlib import Path

from loader import load_pdf

PDF = Path(__file__).parent / "data" / "National_Planning_Policy_Framework.pdf"

if __name__ == "__main__":
    phrase = " ".join(sys.argv[1:]).lower()
    if not phrase:
        print("usage: python grep_corpus.py some phrase")
        sys.exit(1)

    docs = load_pdf(PDF)
    for i, d in enumerate(docs):
        if phrase in d.lower():
            print(f"\n--- doc {i} ---")
            print(d[:400])