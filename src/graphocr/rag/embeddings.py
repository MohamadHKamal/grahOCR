"""Embedding engine — open-source multilingual embeddings for Arabic/English.

Uses a local embedding model (no API calls) for data sovereignty compliance.
The embeddings power the vector store for policy retrieval.
"""

from __future__ import annotations

import numpy as np

from graphocr.core.logging import get_logger

logger = get_logger(__name__)

# Lazy-loaded model to avoid import-time GPU allocation
_model = None


def _get_model():
    """Lazy-load the embedding model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        # multilingual-e5-large: strong Arabic + English, 1024-dim
        # Runs locally — no data leaves the jurisdiction
        _model = SentenceTransformer("intfloat/multilingual-e5-large")
        logger.info("embedding_model_loaded", model="multilingual-e5-large", dim=1024)
    return _model


def embed_texts(texts: list[str], prefix: str = "passage: ") -> list[list[float]]:
    """Embed a batch of texts.

    Args:
        texts: Texts to embed.
        prefix: E5 models require a prefix. Use "query: " for queries
                and "passage: " for documents.

    Returns:
        List of embedding vectors.
    """
    model = _get_model()
    prefixed = [f"{prefix}{t}" for t in texts]
    embeddings = model.encode(prefixed, normalize_embeddings=True)
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single query text."""
    return embed_texts([query], prefix="query: ")[0]


def embed_document(text: str) -> list[float]:
    """Embed a single document/passage text."""
    return embed_texts([text], prefix="passage: ")[0]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_np = np.array(a)
    b_np = np.array(b)
    return float(np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np) + 1e-10))
