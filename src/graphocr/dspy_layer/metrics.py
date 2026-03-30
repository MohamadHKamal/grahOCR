"""Custom DSPy metrics for insurance claim extraction evaluation.

These metrics are used by DSPy optimizers (MIPRO, BootstrapFewShot) to
evaluate prompt quality and select the best demonstrations.
"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher


def field_level_f1(example: dict, prediction: dict, trace=None) -> float:
    """Compute field-level F1 score between expected and predicted claim fields.

    Handles both exact match and fuzzy match for text fields.
    Returns a score between 0.0 and 1.0.
    """
    expected_json = example.get("claim_fields_json", "{}")
    predicted_json = prediction.get("claim_fields_json", "{}")

    try:
        expected = json.loads(expected_json) if isinstance(expected_json, str) else expected_json
        predicted = json.loads(predicted_json) if isinstance(predicted_json, str) else predicted_json
    except json.JSONDecodeError:
        return 0.0

    if not isinstance(expected, dict) or not isinstance(predicted, dict):
        return 0.0

    all_fields = set(expected.keys()) | set(predicted.keys())
    if not all_fields:
        return 1.0

    scores: list[float] = []
    for field in all_fields:
        exp_val = str(expected.get(field, ""))
        pred_val = str(predicted.get(field, ""))

        if not exp_val and not pred_val:
            scores.append(1.0)
        elif not exp_val or not pred_val:
            scores.append(0.0)
        else:
            scores.append(_field_similarity(field, exp_val, pred_val))

    return sum(scores) / len(scores)


def exact_match(example: dict, prediction: dict, trace=None) -> float:
    """Exact string match metric for normalization tasks."""
    expected = example.get("normalized_text", "").strip()
    predicted = prediction.get("normalized_text", "").strip()
    return 1.0 if expected == predicted else 0.0


def code_accuracy(example: dict, prediction: dict, trace=None) -> float:
    """ICD-10/CPT code accuracy metric."""
    expected = example.get("icd10_code", "").strip().upper()
    predicted = prediction.get("icd10_code", "").strip().upper()

    if expected == predicted:
        return 1.0

    # Partial credit for matching category (first 3 chars)
    if expected[:3] == predicted[:3]:
        return 0.5

    return 0.0


def arabic_fuzzy_match(example: dict, prediction: dict, trace=None) -> float:
    """Fuzzy matching for Arabic text that accounts for common OCR errors.

    Strips diacritics and normalizes common character confusions before comparing.
    """
    expected = example.get("claim_fields_json", "")
    predicted = prediction.get("claim_fields_json", "")

    if isinstance(expected, str):
        expected = _normalize_arabic(expected)
    if isinstance(predicted, str):
        predicted = _normalize_arabic(predicted)

    return SequenceMatcher(None, expected, predicted).ratio()


def _field_similarity(field_name: str, expected: str, predicted: str) -> float:
    """Compute similarity for a specific field type."""
    # Exact match fields
    if field_name in ("diagnosis_codes", "procedure_codes", "icd10_code"):
        exp_codes = set(re.findall(r"[\w.]+", expected.upper()))
        pred_codes = set(re.findall(r"[\w.]+", predicted.upper()))
        if not exp_codes and not pred_codes:
            return 1.0
        if not exp_codes or not pred_codes:
            return 0.0
        intersection = exp_codes & pred_codes
        precision = len(intersection) / len(pred_codes)
        recall = len(intersection) / len(exp_codes)
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Numeric fields
    if field_name in ("total_amount",):
        try:
            exp_num = float(re.sub(r"[^\d.]", "", expected))
            pred_num = float(re.sub(r"[^\d.]", "", predicted))
            if exp_num == 0:
                return 1.0 if pred_num == 0 else 0.0
            return max(0, 1.0 - abs(exp_num - pred_num) / exp_num)
        except ValueError:
            return 0.0

    # Date fields
    if "date" in field_name.lower():
        exp_clean = re.sub(r"[^\d]", "", expected)
        pred_clean = re.sub(r"[^\d]", "", predicted)
        return 1.0 if exp_clean == pred_clean else 0.0

    # Text fields — fuzzy match
    exp_norm = _normalize_arabic(expected.lower().strip())
    pred_norm = _normalize_arabic(predicted.lower().strip())
    return SequenceMatcher(None, exp_norm, pred_norm).ratio()


def _normalize_arabic(text: str) -> str:
    """Normalize Arabic text for comparison.

    - Remove diacritics (tashkeel)
    - Normalize alef variants
    - Normalize taa marbuta/haa
    """
    # Remove Arabic diacritics
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)

    # Normalize alef variants to plain alef
    text = text.replace("\u0622", "\u0627")  # Alef with madda -> Alef
    text = text.replace("\u0623", "\u0627")  # Alef with hamza above -> Alef
    text = text.replace("\u0625", "\u0627")  # Alef with hamza below -> Alef

    # Normalize taa marbuta to haa
    text = text.replace("\u0629", "\u0647")

    return text


# Registry for DSPy optimizer metric selection
METRICS = {
    "field_level_f1": field_level_f1,
    "exact_match": exact_match,
    "code_accuracy": code_accuracy,
    "arabic_fuzzy_match": arabic_fuzzy_match,
}
