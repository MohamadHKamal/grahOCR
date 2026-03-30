"""DSPy modules — programmatic prompt optimization for extraction tasks.

Each module wraps an extraction task as a DSPy Signature/Module that
can be optimized by MIPRO or BootstrapFewShot.
"""

from __future__ import annotations

import dspy


class ClaimFieldExtractorSignature(dspy.Signature):
    """Extract structured insurance claim fields from OCR tokens.

    Given spatial tokens with coordinates, reading order, language, and confidence,
    extract patient info, diagnosis codes, procedures, medications, amounts.
    Preserve source token references for provenance.
    """

    spatial_tokens_text: str = dspy.InputField(
        desc="Concatenated OCR tokens with coordinates and metadata"
    )
    document_language: str = dspy.InputField(
        desc="Primary document language: 'ar', 'en', or 'mixed'"
    )
    context_hints: str = dspy.InputField(
        desc="Pre-extracted regex fields and layout zone information"
    )

    claim_fields_json: str = dspy.OutputField(
        desc="JSON with extracted fields: patient_name, patient_id, diagnosis_codes, "
        "procedure_codes, medications, date_of_service, total_amount, etc."
    )


class ClaimFieldExtractor(dspy.Module):
    """DSPy module for extracting claim fields from spatial tokens."""

    def __init__(self):
        super().__init__()
        self.extractor = dspy.ChainOfThought(ClaimFieldExtractorSignature)

    def forward(
        self,
        spatial_tokens_text: str,
        document_language: str,
        context_hints: str = "",
    ) -> dspy.Prediction:
        return self.extractor(
            spatial_tokens_text=spatial_tokens_text,
            document_language=document_language,
            context_hints=context_hints,
        )


class ArabicMedicalNormalizerSignature(dspy.Signature):
    """Normalize Arabic medical terms to standard forms.

    Handles diacritical variations, abbreviations, and handwriting variants
    common in Arabic medical documents.
    """

    arabic_text: str = dspy.InputField(desc="Raw Arabic medical text from OCR")
    context: str = dspy.InputField(desc="Surrounding tokens for context")

    normalized_text: str = dspy.OutputField(
        desc="Standardized Arabic medical term"
    )
    confidence: float = dspy.OutputField(
        desc="Normalization confidence 0.0-1.0"
    )


class ArabicMedicalNormalizer(dspy.Module):
    """Normalizes Arabic medical terms accounting for OCR artifacts."""

    def __init__(self):
        super().__init__()
        self.normalizer = dspy.ChainOfThought(ArabicMedicalNormalizerSignature)

    def forward(self, arabic_text: str, context: str = "") -> dspy.Prediction:
        return self.normalizer(arabic_text=arabic_text, context=context)


class DiagnosisCodeMapperSignature(dspy.Signature):
    """Map extracted diagnosis descriptions to ICD-10 codes.

    Given a free-text diagnosis description (potentially in Arabic),
    identify the correct ICD-10 code.
    """

    diagnosis_text: str = dspy.InputField(
        desc="Free-text diagnosis description from the claim"
    )
    language: str = dspy.InputField(desc="Language of the diagnosis text")

    icd10_code: str = dspy.OutputField(desc="ICD-10 code (e.g., E11.9)")
    code_description: str = dspy.OutputField(desc="Standard English description of the code")
    mapping_confidence: float = dspy.OutputField(desc="Confidence in the mapping 0.0-1.0")


class DiagnosisCodeMapper(dspy.Module):
    """Maps diagnosis descriptions to ICD-10 codes."""

    def __init__(self):
        super().__init__()
        self.mapper = dspy.ChainOfThought(DiagnosisCodeMapperSignature)

    def forward(self, diagnosis_text: str, language: str = "en") -> dspy.Prediction:
        return self.mapper(diagnosis_text=diagnosis_text, language=language)


class PolicyVersionValidatorSignature(dspy.Signature):
    """Validate that extracted policy references match the correct temporal version.

    Detects Type B failures where RAG retrieves a current standard policy
    when the claim actually references an obsolete rider.
    """

    claim_text: str = dspy.InputField(
        desc="Extracted claim text with policy references"
    )
    policy_context: str = dspy.InputField(
        desc="Retrieved policy context from RAG"
    )

    is_correct_version: bool = dspy.OutputField(
        desc="Whether the retrieved policy matches the claim's temporal reference"
    )
    confidence: float = dspy.OutputField(
        desc="Confidence in the version match assessment 0.0-1.0"
    )
    mismatch_explanation: str = dspy.OutputField(
        desc="If mismatched, explanation of the temporal discrepancy"
    )


class PolicyVersionValidator(dspy.Module):
    """Validates policy version alignment between claim and RAG retrieval."""

    def __init__(self):
        super().__init__()
        self.validator = dspy.ChainOfThought(PolicyVersionValidatorSignature)

    def forward(self, claim_text: str, policy_context: str) -> dspy.Prediction:
        return self.validator(claim_text=claim_text, policy_context=policy_context)


class ChallengeGeneratorSignature(dspy.Signature):
    """Generate adversarial challenges for an extraction result.

    Given extracted fields and the original tokens, identify potential
    errors, OCR misreads, or logical inconsistencies.
    """

    extracted_fields: str = dspy.InputField(desc="JSON of extracted claim fields")
    token_sample: str = dspy.InputField(desc="Sample of original OCR tokens")
    known_issues: str = dspy.InputField(desc="Already-detected validation issues")

    challenges_json: str = dspy.OutputField(
        desc="JSON list of adversarial challenges with hypothesis, evidence, confidence"
    )


class ChallengeGenerator(dspy.Module):
    """DSPy-optimizable adversarial challenge generation."""

    def __init__(self):
        super().__init__()
        self.generator = dspy.ChainOfThought(ChallengeGeneratorSignature)

    def forward(
        self,
        extracted_fields: str,
        token_sample: str,
        known_issues: str = "",
    ) -> dspy.Prediction:
        return self.generator(
            extracted_fields=extracted_fields,
            token_sample=token_sample,
            known_issues=known_issues,
        )
