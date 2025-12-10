"""
Embedding utilities using free/open sentence-transformers models.

- Primary model: 'all-MiniLM-L6-v2' (lightweight, fast, good semantic embeddings)
- Optionally support 'BAAI/bge-small-en' if you add it to the environment and have it cached.
- Exposes: get_embedding(text) -> np.ndarray, get_embeddings(list_of_texts)

Notes:
- This code caches the model instance on first import.
- Requires: sentence-transformers (pip install sentence-transformers)
"""

from typing import List
import numpy as np

# Lazy import to keep module import cheap
_model = None
_model_name_tried = None

def _load_model(preferred: str = None):
    global _model, _model_name_tried
    if _model is not None:
        return _model

    # prefer user-provided model name if passed
    candidates = []
    if preferred:
        candidates.append(preferred)
    # default fallback
    candidates += [
        "all-MiniLM-L6-v2",    # light, good general-purpose
        # "BAAI/bge-small-en", # optional: uncomment if available locally / cached
    ]

    for model_name in candidates:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(model_name)
            _model_name_tried = model_name
            return _model
        except Exception as e:
            # keep trying next model
            _model = None
            _model_name_tried = model_name
            continue

    raise RuntimeError(
        "No sentence-transformers model could be loaded. Install 'sentence-transformers' "
        "and ensure one of the models (e.g. 'all-MiniLM-L6-v2') can be downloaded."
    )


def get_embedding(text: str, model_name: str = None) -> np.ndarray:
    """
    Return a 1-D numpy array embedding for the provided text.

    Args:
        text: str
        model_name: optional override for the sentence-transformers model name

    Returns:
        1-D np.ndarray (float32)
    """
    if text is None:
        text = ''
    model = _load_model(preferred=model_name)
    emb = model.encode([text], convert_to_numpy=True, show_progress_bar=False)[0]
    # normalize to unit length to make cosine operations easier
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb.astype('float32')


def get_embeddings(texts: List[str], model_name: str = None) -> np.ndarray:
    """
    Batch encode a list of texts into an (N, D) numpy array.
    """
    model = _load_model(preferred=model_name)
    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=True, batch_size=64)
    # normalize rows
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs = embs / norms
    return embs.astype('float32')
