# calls a local LLM via Ollama with the retrieved context

import ollama

from prompt_builder import SYSTEM_PROMPT, REFUSAL, build_prompt

MODEL = "llama3.2"


def generate(question, hits, model=MODEL):
    prompt = build_prompt(question, hits)
    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0.0},
    )
    return resp["message"]["content"].strip()


def answer(retriever, question, k=3):
    res = retriever.search_with_check(question, k=k)
    if not res["sufficient"]:
        return {
            "answer": REFUSAL,
            "hits": [],
            "best_score": res["best_score"],
            "refused": True,
        }
    text = generate(question, res["hits"])
    return {
        "answer": text,
        "hits": res["hits"],
        "best_score": res["best_score"],
        "refused": False,
    }

if __name__ == "__main__":
    from corpus import docs, questions, out_of_scope
    from retriever import Retriever

    r = Retriever(docs)

    for q in questions + out_of_scope[:2]:
        res = answer(r, q, k=2)
        tag = "REFUSED" if res["refused"] else "ANSWERED"
        print(f"\n{'=' * 60}")
        print(f"Q: {q}")
        print(f"[{tag}] best_score={res['best_score']:.3f}")
        print(f"A: {res['answer']}")

