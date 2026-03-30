"""Vector store for policy chunks — backed by ChromaDB (local, open-source).

ChromaDB runs in-process (no external server needed) and stores everything
locally for data sovereignty compliance. It supports metadata filtering
which we use for temporal-aware retrieval.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from graphocr.core.logging import get_logger
from graphocr.models.policy import PolicyChunk, PolicyType
from graphocr.rag.embeddings import embed_document, embed_query

logger = get_logger(__name__)


class PolicyVectorStore:
    """ChromaDB-backed vector store for insurance policy chunks.

    Each chunk carries temporal and hierarchical metadata so the
    retriever can filter by date, policy type, jurisdiction, etc.
    This prevents the core Failure Type B (context-blind RAG).
    """

    def __init__(self, persist_dir: str = "./data/vectorstore"):
        self._persist_dir = persist_dir
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name="policy_chunks",
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "vector_store_initialized",
            persist_dir=persist_dir,
            chunks=self._collection.count(),
        )

    def add_chunks(self, chunks: list[PolicyChunk]) -> int:
        """Index policy chunks into the vector store.

        Each chunk's temporal metadata (effective_date, expiry_date) is
        stored as filterable metadata — this is what makes retrieval
        context-aware instead of context-blind.
        """
        if not chunks:
            return 0

        ids = []
        documents = []
        metadatas = []
        embeddings = []

        for chunk in chunks:
            # Combine text + Arabic text for bilingual embedding
            combined_text = chunk.text
            if chunk.text_ar:
                combined_text = f"{chunk.text}\n{chunk.text_ar}"

            embedding = embed_document(combined_text)

            ids.append(chunk.chunk_id)
            documents.append(combined_text)
            metadatas.append({
                "policy_id": chunk.policy_id,
                "policy_number": chunk.policy_number,
                "policy_type": chunk.policy_type.value,
                "policy_version": chunk.policy_version,
                "effective_date": chunk.effective_date.isoformat(),
                "expiry_date": chunk.expiry_date.isoformat() if chunk.expiry_date else "",
                "section_title": chunk.section_title,
                "section_type": chunk.section_type,
                "parent_policy_id": chunk.parent_policy_id or "",
                "jurisdiction": chunk.jurisdiction,
            })
            embeddings.append(embedding)

        self._collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        logger.info("chunks_indexed", count=len(chunks))
        return len(chunks)

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
    ) -> list[dict]:
        """Raw vector similarity search with optional metadata filters.

        Args:
            query: Search query text.
            n_results: Max results.
            where: ChromaDB metadata filter (e.g., {"jurisdiction": "SA"}).
            where_document: Document content filter.

        Returns:
            List of dicts with 'chunk_id', 'text', 'metadata', 'distance'.
        """
        query_embedding = embed_query(query)

        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
        }
        if where:
            kwargs["where"] = where
        if where_document:
            kwargs["where_document"] = where_document

        results = self._collection.query(**kwargs)

        hits: list[dict] = []
        if results and results["ids"]:
            for i, chunk_id in enumerate(results["ids"][0]):
                hits.append({
                    "chunk_id": chunk_id,
                    "text": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0,
                })

        return hits

    def delete_policy(self, policy_id: str) -> None:
        """Remove all chunks for a policy (e.g., when superseded)."""
        self._collection.delete(where={"policy_id": policy_id})
        logger.info("policy_chunks_deleted", policy_id=policy_id)

    @property
    def count(self) -> int:
        return self._collection.count()
