"""Policy models — what the RAG retrieves and the LLM adjudicates against.

The core problem with context-blind RAG (Failure Type B) is that a naive
retriever fetches the most semantically similar policy chunk — e.g. a
"2025 Standard Plan" — when the claim explicitly references an obsolete
"2018 Rider." These models encode the temporal, hierarchical, and
jurisdictional structure of insurance policies so the retriever can be
context-aware.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field
from uuid_extensions import uuid7


class PolicyStatus(str, Enum):
    """Lifecycle state of an insurance policy.

    Attributes:
        ACTIVE: Policy is currently in force and can cover claims.
        EXPIRED: Policy has passed its expiry date; no new claims accepted.
        SUPERSEDED: A newer version of this policy exists (e.g. 2018-v1
            replaced by 2025-v1). The old version is kept for historical
            claims but should not be used for new ones.
        PENDING: Policy has been created but is not yet effective
            (e.g. waiting for regulatory approval or a future start date).
    """

    ACTIVE = "active"
    EXPIRED = "expired"
    SUPERSEDED = "superseded"
    PENDING = "pending"


class PolicyType(str, Enum):
    """Kind of insurance policy document.

    Attributes:
        STANDARD: A standalone base plan (e.g. "2025 Standard Health Plan").
        RIDER: An optional add-on purchased on top of a standard plan
            (e.g. dental rider, maternity rider). Always linked to a
            parent policy via ``parent_policy_id``.
        AMENDMENT: A formal modification to an existing policy that changes
            specific terms (e.g. raising the annual limit). Also linked
            to a parent policy.
        GROUP: An employer- or organisation-sponsored plan that covers
            multiple members under a single ``group_id``.
    """

    STANDARD = "standard"
    RIDER = "rider"
    AMENDMENT = "amendment"
    GROUP = "group"


class CoverageRule(BaseModel):
    """A single coverage rule within a policy.

    Each rule maps a medical scenario (identified by procedure/diagnosis
    codes) to the financial terms the insurer will apply. During claim
    adjudication the LLM looks up the matching rule(s) and uses these
    fields to decide approval, partial approval, or denial.

    Example:
        A rule might say: "Knee MRI (CPT 73721) is covered up to
        SAR 2 000 with a 20 % copay, requires pre-authorisation,
        and has a 30-day waiting period after policy activation."

    Attributes:
        rule_id: Auto-generated UUID v7 identifier for this rule.
        description: Human-readable English description of the rule.
        description_ar: Arabic translation of the description (optional).
        procedure_codes: CPT / HCPCS codes this rule covers
            (e.g. ``["73721", "73722"]``).
        diagnosis_codes: ICD-10 diagnosis codes that must be present on
            the claim for this rule to apply (e.g. ``["M17.1"]``).
        max_amount: Maximum reimbursable amount per occurrence. ``None``
            means no per-occurrence cap.
        copay_percentage: Percentage of the approved amount the patient
            must pay (0.0 = fully covered, 0.2 = 20 % copay).
        requires_preauth: Whether the procedure needs pre-authorisation
            from the insurer before it is performed.
        waiting_period_days: Number of days after policy activation
            before this coverage kicks in (0 = immediate).
        annual_limit: Maximum total amount payable under this rule per
            policy year. ``None`` means no annual cap.
        exclusions: Specific scenarios under which this rule does NOT
            apply (e.g. ``["cosmetic procedures", "pre-existing conditions"]``).
    """

    rule_id: str = Field(default_factory=lambda: str(uuid7()))
    description: str
    description_ar: str = ""
    procedure_codes: list[str] = Field(default_factory=list)
    diagnosis_codes: list[str] = Field(default_factory=list)
    max_amount: Decimal | None = None
    copay_percentage: float = 0.0
    requires_preauth: bool = False
    waiting_period_days: int = 0
    annual_limit: Decimal | None = None
    exclusions: list[str] = Field(default_factory=list)


class PolicyDocument(BaseModel):
    """A versioned insurance policy document.

    This is the central model in the policy layer. Every policy has
    **temporal bounds** (effective/expiry dates) and a **version string**
    so the system can resolve *which* edition of a policy applies to a
    given claim.

    Why versioning matters (Type B failure):
        A claim dated 2018-06-15 that references "Plan A" must be
        adjudicated against the 2018 edition of Plan A, not the current
        2025 edition.  The ``effective_date``, ``expiry_date``, ``version``,
        and ``supersedes`` fields work together to let the retriever pick
        the correct version.

    Hierarchy:
        STANDARD policies stand alone.  RIDER and AMENDMENT policies link
        back to their parent via ``parent_policy_id``.  GROUP policies
        additionally carry a ``group_id`` shared by all members.

    Attributes:
        policy_id: Auto-generated UUID v7 primary key.
        policy_number: Human-readable reference the insurer prints on
            cards / letters (e.g. ``"POL-2025-00412"``).
        policy_type: Kind of document (standard, rider, amendment, group).
        status: Current lifecycle state (active, expired, superseded, pending).

        effective_date: First day this policy version is valid.
        expiry_date: Last day this policy version is valid. ``None`` means
            the policy has no fixed end date (open-ended).
        version: Free-text version tag (e.g. ``"2018-v2"``, ``"2025-v1"``).
        supersedes: ``policy_id`` of the older version this one replaces,
            creating a version chain for auditing.

        insurer_name: Name of the insurance company.
        plan_name: Marketing / official name of the plan in English.
        plan_name_ar: Arabic translation of the plan name.
        group_id: Identifier linking all members of an employer/group plan.

        jurisdiction: ISO 3166-1 alpha-2 country code where this policy
            is governed (e.g. ``"SA"`` for Saudi Arabia).
        language: Primary language of the policy document (default ``"ar"``).

        coverage_rules: List of ``CoverageRule`` objects describing what
            procedures/diagnoses are covered and under what terms.
        general_exclusions: Blanket exclusions that apply across all rules
            (e.g. ``["war injuries", "self-inflicted harm"]``).
        max_annual_benefit: Overall cap on total benefits per policy year.
        deductible: Amount the patient pays out-of-pocket before coverage
            begins (default ``0``).

        full_text: Complete English text of the policy — used as input for
            the embedding model when indexing into the vector store.
        full_text_ar: Complete Arabic text of the policy.

        parent_policy_id: For riders/amendments, the ``policy_id`` of the
            base policy this document extends or modifies.
    """

    policy_id: str = Field(default_factory=lambda: str(uuid7()))
    policy_number: str
    policy_type: PolicyType = PolicyType.STANDARD
    status: PolicyStatus = PolicyStatus.ACTIVE

    effective_date: date
    expiry_date: date | None = None
    version: str = "1.0"
    supersedes: str | None = None

    insurer_name: str = ""
    plan_name: str = ""
    plan_name_ar: str = ""
    group_id: str | None = None

    jurisdiction: str = ""
    language: str = "ar"

    coverage_rules: list[CoverageRule] = Field(default_factory=list)
    general_exclusions: list[str] = Field(default_factory=list)
    max_annual_benefit: Decimal | None = None
    deductible: Decimal = Decimal("0")

    full_text: str = ""
    full_text_ar: str = ""

    parent_policy_id: str | None = None

    def is_valid_on(self, check_date: date) -> bool:
        """Check if this policy was valid on a specific date.

        Args:
            check_date: The date to test against (typically the claim's
                service date or submission date).

        Returns:
            ``True`` if ``effective_date <= check_date <= expiry_date``
            (or the policy has no expiry date), ``False`` otherwise.
        """
        if check_date < self.effective_date:
            return False
        if self.expiry_date and check_date > self.expiry_date:
            return False
        return True


class PolicyChunk(BaseModel):
    """A chunk of a policy document prepared for vector-store indexing.

    When a ``PolicyDocument`` is ingested, it is split into smaller
    chunks (paragraphs, sections, or tables). Each chunk is stored in
    the vector database alongside its embedding.

    Unlike naive chunking — where chunks lose their origin metadata —
    every ``PolicyChunk`` carries the **temporal** and **hierarchical**
    context of the parent policy. This lets the retriever apply
    date-range and jurisdiction filters *before* semantic similarity,
    preventing Type B failures (wrong-version retrieval).

    Typical flow:
        1. ``PolicyDocument`` is split into chunks.
        2. Each chunk is embedded (e.g. via ``text-embedding-3-small``).
        3. Chunks are upserted into the vector store with metadata filters.
        4. At query time the retriever says: "give me chunks where
           ``effective_date <= claim_date <= expiry_date`` AND
           ``policy_number == X``", then ranks by cosine similarity.

    Attributes:
        chunk_id: Auto-generated UUID v7 identifier for this chunk.
        policy_id: ``policy_id`` of the parent ``PolicyDocument``.
        policy_number: Copied from the parent for fast filtering.
        policy_type: Copied from the parent for fast filtering.
        policy_version: Version tag of the parent policy.

        effective_date: Inherited from the parent policy — used as a
            metadata filter during retrieval.
        expiry_date: Inherited from the parent policy.

        text: The English text content of this chunk.
        text_ar: The Arabic text content of this chunk.
        section_title: Title of the section this chunk was extracted from
            (e.g. ``"Outpatient Benefits"``).
        section_type: Semantic category of the section — one of
            ``"coverage"``, ``"exclusion"``, ``"benefit_limit"``, or
            ``"definition"``.

        parent_policy_id: If the parent policy is a rider/amendment, this
            points to the base policy for hierarchical traversal.
        jurisdiction: Country code, copied for filtering.

        embedding: The dense vector representation of ``text``, populated
            by the embedding model. ``None`` until the chunk is embedded.
    """

    chunk_id: str = Field(default_factory=lambda: str(uuid7()))
    policy_id: str
    policy_number: str
    policy_type: PolicyType
    policy_version: str

    effective_date: date
    expiry_date: date | None = None

    text: str
    text_ar: str = ""
    section_title: str = ""
    section_type: str = ""

    parent_policy_id: str | None = None
    jurisdiction: str = ""

    embedding: list[float] | None = None


class RetrievalContext(BaseModel):
    """The context package delivered to the LLM for claim adjudication.

    After the retriever selects the most relevant ``PolicyChunk`` objects,
    they are bundled into a ``RetrievalContext`` together with metadata
    explaining **why** those chunks were chosen. This metadata serves two
    purposes:

    1. **LLM grounding** — the adjudication prompt includes
       ``retrieval_method``, ``confidence``, and ``warnings`` so the model
       can hedge or escalate when the retrieval is uncertain.
    2. **Auditability** — human reviewers can inspect the context to
       verify that the correct policy version was used.

    Attributes:
        claim_id: The claim this context was built for.
        policy_chunks: Ordered list of retrieved ``PolicyChunk`` objects
            (most relevant first).
        retrieval_method: Strategy used to fetch chunks:
            - ``"temporal_filtered"`` — date-range filter applied first,
              then semantic ranking.
            - ``"semantic_only"`` — pure cosine-similarity search (no
              date filter; higher risk of Type B failure).
            - ``"hybrid"`` — combination of temporal, semantic, and
              graph-based retrieval.
        query_date: The date used for temporal filtering (usually the
            claim's service date). ``None`` if no date filter was applied.
        policy_reference_from_claim: The raw policy reference string
            extracted from the claim document (e.g. ``"2018 Rider B"``).
        matched_policy_number: The ``policy_number`` the retriever
            resolved the reference to.
        matched_policy_version: The ``version`` of the matched policy.
        confidence: A 0.0 – 1.0 score indicating how confident the
            retriever is that the correct policy version was matched.
        warnings: Human-readable warnings surfaced during retrieval
            (e.g. ``["Policy expired before claim service date"]``).
    """

    claim_id: str
    policy_chunks: list[PolicyChunk]
    retrieval_method: str
    query_date: date | None = None
    policy_reference_from_claim: str = ""
    matched_policy_number: str = ""
    matched_policy_version: str = ""
    confidence: float = 0.0
    warnings: list[str] = Field(default_factory=list)
