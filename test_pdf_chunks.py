# how does the chunker behaves on real PDF text

from pathlib import Path

from loader import load_pdf
from chunker import chunk_docs

PDF = Path(__file__).parent / "data" / "National_Planning_Policy_Framework.pdf"

if __name__ == "__main__":
    docs = load_pdf(PDF)
    chunks = chunk_docs(docs, target_size=300)

    lens = [len(c["text"]) for c in chunks]
    print(f"{len(docs)} docs -> {len(chunks)} chunks")
    print(f"min={min(lens)} max={max(lens)} mean={sum(lens)//len(lens)}")

    over = [n for n in lens if n > 300]
    print(f"chunks over target: {len(over)} ({100*len(over)//len(lens)}%)")

    print("\n--- the longest chunk ---")
    worst = lens.index(max(lens))
    print(chunks[worst]["text"][:600])

