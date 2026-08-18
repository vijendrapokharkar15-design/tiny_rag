# loads a pdf and flattens it into plain prose documents

import re
from pathlib import Path

from pypdf import PdfReader

def extract_pages(pdf_path, skip_pages=0):
    reader=PdfReader(pdf_path)
    pages = []
    for page in reader.pages[skip_pages:]:
        text=page.extract_text() or ""
        pages.append(text)
    return pages


def clean_text(text):
    # normalise line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\.{4,}", " ", text)

    # a line that is only whitespace marks a real break - standardise it
    text = re.sub(r"\n[ \t]*\n", "\n\n", text)

    # single newlines are visual line wraps - turn them into spaces
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # collapse runs of spaces
    text = re.sub(r"[ \t]+", " ", text)

    # tidy space before punctuation left behind by the joins
    text = re.sub(r" ([.,;:])", r"\1", text)

    return text.strip()


def drop_page_numbers(text):
    # a paragraph that is just digits is almost certainly a page number
    paras = text.split("\n\n")
    keep = [p for p in paras if not re.fullmatch(r"\d{1,4}", p.strip())]
    return "\n\n".join(keep)

CLAUSE_RE = re.compile(r"^(?:[ivxlc]+|[a-z])\.\s", re.IGNORECASE)
PARA_RE = re.compile(r"^\d+\.\s")


def is_clause(text):
    return bool(CLAUSE_RE.match(text.strip()))


def load_pdf(pdf_path, min_chars=100, skip_pages=4):
    pages=extract_pages(pdf_path, skip_pages)
    docs = []
    for text in pages:
        text=clean_text(text)
        text=drop_page_numbers(text)
        for para in text.split("\n\n"):
            para = para.strip()
            if not para:
                continue
            if is_clause(para) and docs:
                docs[-1] = docs[-1] + " " + para
            else:
                docs.append(para)
    return [d for d in docs if len(d) >= min_chars]



if __name__ == "__main__":
    path = Path("data/National_Planning_Policy_Framework.pdf")
    docs = load_pdf(path)

    print(f"{len(docs)} documents extracted")
    lens = [len(d) for d in docs]
    print(f"min={min(lens)} max={max(lens)} mean={sum(lens)//len(lens)}")

    print("\n--- first 3 ---")
    for d in docs[:3]:
        print(f"\n[{len(d)} chars] {d[:300]}")

    print("\n--- 3 from the middle ---")
    mid = len(docs) // 2
    for d in docs[mid:mid + 3]:
        print(f"\n[{len(d)} chars] {d[:300]}")