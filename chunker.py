# splits documents into chunks for embedding
# naive sentence splitting - fine for clean prose, breaks on "Dr. Smith" etc

import re



def split_sentences(text):
    # split on ". " and "; " - policy text uses semicolons as clause boundaries
    parts = re.split(r"(?<=[.;])\s+", text)
    out = []
    for p in parts:
        p = p.strip()
        if p:
            out.append(p)
    return out


def chunk_doc(text, target_size=300, overlap_sentences=1):
    sents = split_sentences(text)
    chunks = []
    current = []
    size = 0

    for s in sents:
        if size + len(s) > target_size and current:
            chunks.append(" ".join(current))
            # only carry overlap if the flushed chunk had more than
            # overlap_sentences in it - otherwise we'd duplicate the whole chunk
            if overlap_sentences and len(current) > overlap_sentences:
                current = current[-overlap_sentences:]
            else:
                current = []
            size = sum(len(x) for x in current)
        current.append(s)
        size += len(s)

    if current:
        chunks.append(" ".join(current))

    return chunks


def chunk_docs(docs, target_size=300, overlap_sentences=1):
    all_chunks = []
    for i, d in enumerate(docs):
        for c in chunk_doc(d, target_size, overlap_sentences):
            all_chunks.append({"text": c, "doc_id": i})
    return all_chunks


if __name__ == "__main__":
    from corpus import docs

    chunks = chunk_docs(docs, target_size=300)
    print(f"{len(docs)} docs -> {len(chunks)} chunks\n")
    for i, c in enumerate(chunks):
        print(f"[{i}] doc={c['doc_id']} len={len(c['text'])}")
        print(f"    {c['text'][:80]}...")