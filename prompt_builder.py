# assembles the LLM prompt from retrieved chunks

REFUSAL = "I don't have information about that in my documents."

SYSTEM_PROMPT = """You are a careful assistant answering questions about a set of documents.

Rules:
- Answer using ONLY the context provided below.
- Do not use outside knowledge, even if you are confident.
- Cite the context number(s) you used, like [1] or [2].
- If the context does not contain the answer, say you don't know.
- Keep the answer to two or three sentences."""


def format_context(hits):
    parts = []
    for i, h in enumerate(hits, start=1):
        parts.append(f"[{i}] (source doc {h['doc_id']})\n{h['text']}")
    return "\n\n".join(parts)


def build_prompt(question, hits):
    context = format_context(hits)
    return f"""Context:

{context}

Question: {question}

Answer:"""


if __name__ == "__main__":
    from corpus import docs, questions
    from retriever import Retriever

    r = Retriever(docs)
    res = r.search_with_check(questions[0], k=2)

    print(SYSTEM_PROMPT)
    print("\n" + "=" * 50 + "\n")
    print(build_prompt(questions[0], res["hits"]))

