"""Temporal-aware policy retriever — solves Failure Type B (context-blind RAG).

The standard RAG approach fails because it retrieves the most semantically
similar policy chunk regardless of temporal context. A claim from 2018
referencing a "Rider 2018-R3" gets matched to the 2025 Standard Plan
because the embeddings are similar.

This retriever implements a 3-stage retrieval strategy:

  Stage 1: EXTRACT — Pull the policy reference and date from the claim tokens
  Stage 2: FILTER — Narrow to policies valid on the claim's date of service
  Stage 3: RANK — Semantic similarity within the temporally-filtered set

If no explicit policy reference is found, falls back to temporal+semantic
hybrid. If no date is found, falls back to pure semantic with a warning.
"""

from __future__ import annotations

import re
from datetime import date

from graphocr.core.logging import get_logger
from graphocr.models.policy import PolicyChunk, PolicyType, RetrievalContext
from graphocr.models.token import SpatialToken
from graphocr.rag.vector_store import PolicyVectorStore

logger = get_logger(__name__)


class TemporalPolicyRetriever:
    """Context-aware policy retriever that prevents Failure Type B.

    Unlike naive semantic retrieval, this retriever:
    1. Extracts the policy reference and date from the claim
    2. Filters policies by temporal validity
    3. Ranks by semantic similarity within the valid set
    4. Warns when falling back to less-precise methods
    """

    def __init__(self, vector_store: PolicyVectorStore):
        self._store = vector_store

    def retrieve(
        self,
        tokens: list[SpatialToken],
        claim_date: date | None = None,
        jurisdiction: str = "",
        n_results: int = 5,
    ) -> RetrievalContext:
        """Retrieve the correct policy context for a claim.

        Args:
            tokens: SpatialTokens from OCR extraction.
            claim_date: Date of service (if already extracted).
            jurisdiction: Jurisdiction code for filtering.
            n_results: Max chunks to retrieve.

        Returns:
            RetrievalContext with policy chunks and retrieval metadata.
        """
        # Stage 1: Extract policy reference and date from tokens
        full_text = " ".join(t.text for t in tokens)
        policy_ref = self._extract_policy_reference(full_text)
        service_date = claim_date or self._extract_date(full_text)

        warnings: list[str] = []

        # Stage 2 + 3: Retrieve with temporal filtering
        if policy_ref and service_date:
            # Best case: exact policy reference + temporal filter
            context = self._retrieve_by_reference_and_date(
                policy_ref, service_date, jurisdiction, n_results
            )
            context.retrieval_method = "temporal_filtered"

        elif service_date:
            # No explicit reference — temporal + semantic hybrid
            context = self._retrieve_by_date_and_semantic(
                full_text, service_date, jurisdiction, n_results
            )
            context.retrieval_method = "temporal_semantic_hybrid"
            warnings.append(
                "No explicit policy reference found in claim. "
                "Retrieved by temporal filtering + semantic similarity."
            )

        elif policy_ref:
            # Reference but no date — filter by reference, warn about temporal
            context = self._retrieve_by_reference(
                policy_ref, jurisdiction, n_results
            )
            context.retrieval_method = "reference_only"
            warnings.append(
                "No date of service found for temporal filtering. "
                "Retrieved by policy reference only — verify policy version is correct."
            )

        else:
            # Worst case: pure semantic (context-blind) — this is what we're trying to avoid
            context = self._retrieve_semantic_only(
                full_text, jurisdiction, n_results
            )
            context.retrieval_method = "semantic_only"
            warnings.append(
                "WARNING: Falling back to pure semantic retrieval (no temporal or "
                "reference filtering). This is a Failure Type B risk — verify the "
                "retrieved policy version matches the claim."
            )

        context.policy_reference_from_claim = policy_ref or ""
        context.query_date = service_date
        context.warnings = warnings

        logger.info(
            "policy_retrieved",
            method=context.retrieval_method,
            policy_ref=policy_ref,
            date=str(service_date),
            chunks=len(context.policy_chunks),
            warnings=len(warnings),
        )

        return context

    def _retrieve_by_reference_and_date(
        self,
        policy_ref: str,
        service_date: date,
        jurisdiction: str,
        n_results: int,
    ) -> RetrievalContext:
        """Best retrieval: exact policy reference + temporal filter."""
        # Build metadata filter
        where_filter: dict = {}
        conditions = []

        # Filter by policy number (partial match via document search)
        # and temporal validity
        date_str = service_date.isoformat()

        # First try exact policy number match
        hits = self._store.search(
            query=policy_ref,
            n_results=n_results * 2,  # Fetch extra, then filter
            where={"policy_number": policy_ref} if not " " in policy_ref else None,
        )

        # If no exact match, try semantic search with broader filter
        if not hits:
            hits = self._store.search(
                query=policy_ref,
                n_results=n_results * 2,
            )

        # Temporal filter: keep only chunks valid on the service date
        filtered = self._temporal_filter(hits, service_date)

        # Take top n
        chunks = self._hits_to_chunks(filtered[:n_results])

        return RetrievalContext(
            claim_id="",
            policy_chunks=chunks,
            matched_policy_number=chunks[0].policy_number if chunks else "",
            matched_policy_version=chunks[0].policy_version if chunks else "",
            confidence=0.9 if chunks else 0.0,
        )

    def _retrieve_by_date_and_semantic(
        self,
        query_text: str,
        service_date: date,
        jurisdiction: str,
        n_results: int,
    ) -> RetrievalContext:
        """Temporal + semantic hybrid retrieval."""
        # Build a richer query from claim text
        query = self._build_semantic_query(query_text)

        where_filter = {}
        if jurisdiction:
            where_filter["jurisdiction"] = jurisdiction

        hits = self._store.search(
            query=query,
            n_results=n_results * 3,
            where=where_filter or None,
        )

        # Temporal filter
        filtered = self._temporal_filter(hits, service_date)
        chunks = self._hits_to_chunks(filtered[:n_results])

        return RetrievalContext(
            claim_id="",
            policy_chunks=chunks,
            matched_policy_number=chunks[0].policy_number if chunks else "",
            matched_policy_version=chunks[0].policy_version if chunks else "",
            confidence=0.7 if chunks else 0.0,
        )

    def _retrieve_by_reference(
        self,
        policy_ref: str,
        jurisdiction: str,
        n_results: int,
    ) -> RetrievalContext:
        """Reference-only retrieval (no temporal filter)."""
        hits = self._store.search(
            query=policy_ref,
            n_results=n_results,
        )

        chunks = self._hits_to_chunks(hits[:n_results])

        return RetrievalContext(
            claim_id="",
            policy_chunks=chunks,
            matched_policy_number=chunks[0].policy_number if chunks else "",
            matched_policy_version=chunks[0].policy_version if chunks else "",
            confidence=0.5 if chunks else 0.0,
        )

    def _retrieve_semantic_only(
        self,
        query_text: str,
        jurisdiction: str,
        n_results: int,
    ) -> RetrievalContext:
        """Pure semantic retrieval — Failure Type B risk."""
        query = self._build_semantic_query(query_text)

        where_filter = {}
        if jurisdiction:
            where_filter["jurisdiction"] = jurisdiction

        hits = self._store.search(
            query=query,
            n_results=n_results,
            where=where_filter or None,
        )

        chunks = self._hits_to_chunks(hits[:n_results])

        return RetrievalContext(
            claim_id="",
            policy_chunks=chunks,
            matched_policy_number=chunks[0].policy_number if chunks else "",
            matched_policy_version=chunks[0].policy_version if chunks else "",
            confidence=0.3 if chunks else 0.0,
        )

    def _temporal_filter(
        self, hits: list[dict], service_date: date
    ) -> list[dict]:
        """Filter search hits to only include policies valid on the given date.

        This is the key operation that prevents Failure Type B. A naive
        retriever skips this step entirely.
        """
        filtered = []
        for hit in hits:
            meta = hit.get("metadata", {})
            eff_str = meta.get("effective_date", "")
            exp_str = meta.get("expiry_date", "")

            try:
                effective = date.fromisoformat(eff_str)
            except (ValueError, TypeError):
                # If no effective date, include with lower priority
                filtered.append(hit)
                continue

            # Check if policy was active on the service date
            if service_date < effective:
                continue  # Policy not yet effective

            if exp_str:
                try:
                    expiry = date.fromisoformat(exp_str)
                    if service_date > expiry:
                        continue  # Policy expired
                except (ValueError, TypeError):
                    pass

            filtered.append(hit)

        return filtered

    @staticmethod
    def _extract_policy_reference(text: str) -> str | None:
        """Extract policy reference number from claim text.

        Looks for common patterns:
        - "Policy No: XXX-YYYY-ZZ"
        - "Plan: Standard 2018"
        - "Rider 2018-R3"
        - Arabic: "رقم البوليصة: ..."
        """
        patterns = [
            r"(?:Policy|Plan|Rider|Contract)\s*(?:No\.?:?|Number:?|#|:)\s*([\w\-/]+)",
            r"(?:بوليصة|خطة|ملحق|عقد)\s*(?:رقم:?|#|:)\s*([\w\-/]+)",
            r"(Rider\s+\d{4}[\-\w]*)",
            r"(Plan\s+\d{4}[\-\w]*)",
            r"(?:Policy\s+)([\w]{2,3}-\d{4,}-\d+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None

    @staticmethod
    def _extract_date(text: str) -> date | None:
        """Extract a date of service from claim text."""
        patterns = [
            r"(\d{4}[-/]\d{1,2}[-/]\d{1,2})",     # YYYY-MM-DD
            r"(\d{1,2}[-/]\d{1,2}[-/]\d{4})",       # DD/MM/YYYY or MM/DD/YYYY
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return date.fromisoformat(match.group(1).replace("/", "-"))
                except ValueError:
                    pass

        return None

    @staticmethod
    def _build_semantic_query(full_text: str) -> str:
        """Build a focused semantic query from claim text.

        Instead of embedding the entire claim, extract the key terms
        that should match policy language (diagnoses, procedures, coverage).
        """
        # Extract medical/insurance keywords
        keywords = []

        # ICD codes
        icd = re.findall(r"\b[A-Z]\d{2}(?:\.\d{1,4})?\b", full_text)
        keywords.extend(icd)

        # CPT codes
        cpt = re.findall(r"\b\d{5}\b", full_text)
        keywords.extend(cpt[:3])

        # Common insurance terms (English + Arabic)
        terms = re.findall(
            r"(?:coverage|benefit|exclusion|deductible|copay|preauth|authorization|"
            r"تغطية|فوائد|استثناء|خصم|تفويض)",
            full_text,
            re.IGNORECASE,
        )
        keywords.extend(terms)

        if keywords:
            return " ".join(keywords[:10])

        # Fallback: first 200 chars
        return full_text[:200]

    @staticmethod
    def _hits_to_chunks(hits: list[dict]) -> list[PolicyChunk]:
        """Convert vector store hits to PolicyChunk objects."""
        chunks = []
        for hit in hits:
            meta = hit.get("metadata", {})
            try:
                effective = date.fromisoformat(meta.get("effective_date", "2020-01-01"))
            except ValueError:
                effective = date(2020, 1, 1)

            expiry = None
            if meta.get("expiry_date"):
                try:
                    expiry = date.fromisoformat(meta["expiry_date"])
                except ValueError:
                    pass

            chunks.append(PolicyChunk(
                chunk_id=hit.get("chunk_id", ""),
                policy_id=meta.get("policy_id", ""),
                policy_number=meta.get("policy_number", ""),
                policy_type=PolicyType(meta.get("policy_type", "standard")),
                policy_version=meta.get("policy_version", ""),
                effective_date=effective,
                expiry_date=expiry,
                text=hit.get("text", ""),
                section_title=meta.get("section_title", ""),
                section_type=meta.get("section_type", ""),
                parent_policy_id=meta.get("parent_policy_id") or None,
                jurisdiction=meta.get("jurisdiction", ""),
            ))

        return chunks
