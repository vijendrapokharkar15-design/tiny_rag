# turns text into vectors using a pretrained sentence embedding model

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

_model = None

def get_model():
    global _model
    if _model is None:
        print(f"loading {MODEL_NAME} ...")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts):
    model = get_model()
    vecs = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.array(vecs, dtype=np.float32)

if __name__ == "__main__":
    from corpus import docs
    from chunker import chunk_docs

    chunks = chunk_docs(docs, target_size=300)
    texts = [c["text"] for c in chunks]

    vecs = embed_texts(texts)

    print(f"{len(texts)} chunks -> matrix {vecs.shape}")
    print(f"dtype: {vecs.dtype}")
    print(f"first vector, first 8 values:\n{vecs[0][:8]}")
    print(f"norm of each vector: {np.linalg.norm(vecs, axis=1)}")