# tiny_rag

A retrieval-augmented generation system over the UK **National Planning Policy Framework** (130 pages), built from scratch in numpy — no LangChain, no vector database — so that every stage of the pipeline is visible and measurable.

The point of the project is not that it works. It is that its retrieval quality is **measured**, its failures are **diagnosed**, and one improvement is **proven** against a baseline.

---

## Result

Retrieval quality on a hand-labelled test set of 8 questions:

| Retrieval method | recall@5 | MRR |
|---|---|---|
| Dense only (MiniLM cosine) | 0.750 | 0.521 |
| Hybrid (dense + BM25, RRF fusion) | **1.000** | **0.781** |

Both previously-failing questions were fixed, and 5 of 8 questions now return the correct document at rank 1.

---

## Pipeline

```
PDF → extract → clean → clause-join → chunk → embed ─┐
                                                      ├─ RRF fusion → top-k → prompt → LLM → answer
                                    tokenize → BM25 ──┘
```

| File | Responsibility |
|---|---|
| `loader.py` | PDF extraction, whitespace normalisation, clause joining |
| `chunker.py` | Sentence/clause splitting with overlap |
| `embedder.py` | `all-MiniLM-L6-v2`, normalised to unit length |
| `sparse.py` | BM25 keyword index |
| `retriever.py` | Dense retrieval, deduplication, score threshold |
| `hybrid.py` | Dense + BM25 combined via reciprocal rank fusion |
| `prompt_builder.py` | Context formatting with citation markers |
| `generator.py` | Local LLM call via Ollama (`llama3.2`) |
| `rag.py` | `ask()` entry point and interactive loop |
| `testset.py` | Hand-labelled question → document pairs |
| `evaluate.py` | recall@k and MRR |

Retrieval is a single matrix multiply: `vecs @ query`, `(1539, 384) @ (384,) → (1539,)`. Because embeddings are unit-normalised, the dot product *is* cosine similarity. At this scale numpy is faster than a vector database — 1,539 vectors is 2.3 MB, and search is sub-millisecond.

---

## Evaluation method

Eight questions were labelled by hand. Two decisions matter:

- **Labels were found by keyword search, not by the embedding model.** A test set produced by the system under test only measures the system against itself.
- **Documents are labelled, not chunks.** Chunk indices change whenever `target_size` or the splitter changes; document indices are stable, so the test set survives re-chunking.

`recall@5` asks whether *any* correct document appeared in the top 5 — the right question for RAG, since the LLM needs only one good passage. `MRR` captures what recall discards: whether the answer was first or scraped in at rank 5.

---

## What was diagnosed

**Dense retrieval misses exact terminology.** The query *"What is the presumption in favour of sustainable development?"* returned five documents about sustainable development generally, and never the one paragraph containing the phrase. That paragraph exists exactly once in 614 documents.

BM25 returned it at **rank 1 with a score of 20.02**, against 13.04 for rank 2. The distinguishing word — *presumption* — is rare across the corpus, so IDF weighting gives it enormous influence. A 384-dimensional embedding averages that same word away.

The two methods fail in complementary ways: dense handles paraphrase and misses rare terms; sparse handles rare terms and has no notion of meaning. Reciprocal rank fusion combines them by **rank rather than score**, because a BM25 score of 20.02 and a cosine similarity of 0.674 are not comparable quantities.

**List-shaped documents embed poorly.** A paragraph consisting of bulleted building types (police stations, basement dwellings, caravans) was invisible to dense retrieval — no sentence structure to encode, just a pile of nouns. BM25 recovered it, since those nouns are exactly the rare terms keyword search handles well.

---

## Refusal

The system refuses out-of-scope questions **before** calling the LLM, using a dense-score threshold calibrated empirically:

| | best dense score |
|---|---|
| In-scope questions | 0.644 – 0.838 |
| Out-of-scope questions | 0.073 – 0.207 |

Midpoint: **0.43**. Clean separation with a gap of 0.437.

Ranking is RRF-driven, but the threshold reads the dense score, because RRF values are rank-derived and carry no information about match quality — something always ranks first, however irrelevant.

---

## Limitations

Stated plainly, because they are the honest part of the result.

**The test set is 8 questions.** One question is worth 12.5 recall points. A perfect 1.000 means "no failures among the eight cases chosen", not "solved". Extending the set to 30 would very likely drop recall below 1.0 — and that would be the metric working, not breaking.

**The threshold is calibrated on easy questions.** All eight in-scope questions have clear, locatable answers. Vaguer or worse-phrased real questions will score below 0.644 and be refused.

**Tables extract as unusable text.** The flood zone compatibility matrix flattens to `Zone 1 ✓ ✓ ✓ ✓ ✓ Zone 2 ✓ Exception test required ✓...` — row and column relationships lost. `pypdf` extracts text, not structure. Two such tables are in the index and one has already surfaced as a retrieval result.

**Clause flattening loses scope.** Sub-clauses are joined to their parent paragraph, but the `a.` / `b.` / `i.` hierarchy is not preserved, so a chunk can present a clause without the stem that governs it.

**Fusion can demote a correct result.** *"What is a heritage asset?"* ranked 1st under dense retrieval and 2nd under hybrid. Net effect is strongly positive; it is not free.

**Correct retrieval does not guarantee a correct answer.** Asked about the presumption in favour of sustainable development — with the correct paragraph retrieved at rank 1 and present in the context — `llama3.2` answered with the Brundtland definition of sustainable development instead, drawn from a different retrieved chunk and from its own training data. The instruction *"answer using ONLY the context"* is a mitigation, not a constraint, and small models gravitate toward the definition they already know.

Fixing retrieval moved the failure downstream. Faithfulness evaluation is the missing layer.

---

## Running it

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install sentence-transformers pypdf rank-bm25 ollama

ollama pull llama3.2

python evaluate.py          # dense baseline
python evaluate_hybrid.py   # hybrid, with baseline printed for comparison
python rag.py               # interactive
```

Place the source PDF in `data/`.

---

## What would come next

1. **Extend the test set to 30+ questions** — the current numbers are directionally right but statistically thin.
2. **Faithfulness evaluation** — a second LLM pass checking whether each claim in an answer is supported by the retrieved context. This is the layer that would have caught the Brundtland failure automatically.
3. **A larger generator.** The retrieval half is sound; the generation half is limited by a 3B model.
4. **Structure-aware chunking** — preserving the clause hierarchy would make chunks citable as `DM3(1)(d)`, matching how the document is actually referenced.
