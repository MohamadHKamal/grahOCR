"""Calibrate the traffic controller threshold using ROC analysis.

Given a labeled dataset of (tokens, ground_truth_needs_vlm) pairs,
computes the optimal uncertainty threshold that maximizes the Youden
index (sensitivity + specificity - 1), then updates the config.

Usage:
    python scripts/calibrate_threshold.py --dataset labeled_claims.json
    python scripts/calibrate_threshold.py --synthetic 1000

The synthetic mode generates random token profiles to demonstrate the
calibration process; replace with real labeled data in production.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from graphocr.core.logging import setup_logging, get_logger
from graphocr.core.types import Language
from graphocr.layer3_inference.traffic_controller import route_document
from graphocr.models.token import BoundingBox, SpatialToken

logger = get_logger(__name__)


@dataclass
class LabeledSample:
    """A labeled sample: tokens + whether it actually needed VLM consensus."""

    tokens: list[SpatialToken]
    needs_vlm: bool  # Ground truth: did this doc need expensive processing?
    uncertainty_score: float = 0.0  # Filled during calibration


def generate_synthetic_dataset(n: int) -> list[LabeledSample]:
    """Generate synthetic labeled samples for calibration demo.

    In production, replace with real labeled data where human reviewers
    marked which documents the cheap rail got wrong.
    """
    samples = []
    for i in range(n):
        # Random document characteristics
        num_tokens = random.randint(10, 50)
        base_confidence = random.uniform(0.3, 0.99)
        handwriting_ratio = random.uniform(0.0, 0.8)
        is_mixed_lang = random.random() < 0.3

        tokens = []
        for j in range(num_tokens):
            conf = max(0.1, min(1.0, base_confidence + random.gauss(0, 0.1)))
            lang = Language.ARABIC if (is_mixed_lang and random.random() < 0.5) else Language.ENGLISH
            is_hw = random.random() < handwriting_ratio

            tokens.append(SpatialToken(
                text=f"tok_{j}",
                bbox=BoundingBox(
                    x_min=random.randint(0, 700),
                    y_min=random.randint(0, 900),
                    x_max=random.randint(50, 750),
                    y_max=random.randint(20, 950),
                    page_number=1,
                ),
                reading_order=j,
                confidence=conf,
                ocr_engine="test",
                language=lang,
                is_handwritten=is_hw,
            ))

        # Ground truth: docs need VLM if confidence is low, lots of handwriting,
        # or mixed language — with some noise
        avg_conf = sum(t.confidence for t in tokens) / len(tokens)
        needs_vlm = (
            avg_conf < 0.75
            or handwriting_ratio > 0.5
            or (is_mixed_lang and avg_conf < 0.85)
        )
        # Add 10% label noise
        if random.random() < 0.1:
            needs_vlm = not needs_vlm

        samples.append(LabeledSample(tokens=tokens, needs_vlm=needs_vlm))

    return samples


def compute_roc_curve(samples: list[LabeledSample]) -> dict:
    """Compute ROC curve and find the optimal threshold.

    Returns:
        Dict with thresholds, TPR, FPR, AUC, and optimal threshold.
    """
    # Score all samples
    for sample in samples:
        decision = route_document(sample.tokens)
        sample.uncertainty_score = decision.uncertainty_score

    scores = [s.uncertainty_score for s in samples]
    labels = [s.needs_vlm for s in samples]

    # Generate thresholds
    thresholds = np.linspace(0.0, 1.0, 200)

    tpr_list = []
    fpr_list = []
    youden_list = []

    for thresh in thresholds:
        # Prediction: uncertainty > threshold => route to VLM (positive)
        predictions = [score > thresh for score in scores]

        tp = sum(1 for p, l in zip(predictions, labels) if p and l)
        fp = sum(1 for p, l in zip(predictions, labels) if p and not l)
        fn = sum(1 for p, l in zip(predictions, labels) if not p and l)
        tn = sum(1 for p, l in zip(predictions, labels) if not p and not l)

        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        specificity = tn / max(tn + fp, 1)

        tpr_list.append(tpr)
        fpr_list.append(fpr)
        youden_list.append(tpr + specificity - 1)

    # AUC via trapezoidal rule
    auc = -np.trapz(tpr_list, fpr_list)

    # Optimal threshold: maximize Youden index (sensitivity + specificity - 1)
    best_idx = int(np.argmax(youden_list))
    optimal_threshold = float(thresholds[best_idx])

    # Also compute the threshold that gives ~90/10 split
    for i, thresh in enumerate(thresholds):
        predictions = [score > thresh for score in scores]
        vlm_ratio = sum(predictions) / len(predictions)
        if abs(vlm_ratio - 0.10) < 0.02:
            split_threshold = float(thresh)
            break
    else:
        split_threshold = optimal_threshold

    return {
        "thresholds": thresholds.tolist(),
        "tpr": tpr_list,
        "fpr": fpr_list,
        "youden": youden_list,
        "auc": float(auc),
        "optimal_threshold_youden": optimal_threshold,
        "optimal_youden_value": float(youden_list[best_idx]),
        "optimal_tpr": float(tpr_list[best_idx]),
        "optimal_fpr": float(fpr_list[best_idx]),
        "threshold_90_10_split": split_threshold,
        "n_samples": len(samples),
        "n_positive": sum(labels),
        "n_negative": len(labels) - sum(labels),
    }


def main():
    parser = argparse.ArgumentParser(description="Calibrate traffic controller threshold")
    parser.add_argument("--dataset", type=str, help="Path to labeled dataset JSON")
    parser.add_argument("--synthetic", type=int, default=0, help="Generate N synthetic samples")
    parser.add_argument("--output", type=str, default="calibration_results.json")
    args = parser.parse_args()

    setup_logging("INFO")

    if args.synthetic > 0:
        print(f"Generating {args.synthetic} synthetic samples...")
        samples = generate_synthetic_dataset(args.synthetic)
    elif args.dataset:
        print(f"Loading dataset from {args.dataset}...")
        raw = json.loads(Path(args.dataset).read_text())
        samples = []
        for item in raw:
            tokens = [SpatialToken(**t) for t in item["tokens"]]
            samples.append(LabeledSample(tokens=tokens, needs_vlm=item["needs_vlm"]))
    else:
        print("Using default 1000 synthetic samples...")
        samples = generate_synthetic_dataset(1000)

    print(f"Samples: {len(samples)} ({sum(s.needs_vlm for s in samples)} need VLM)")
    print("Computing ROC curve...")

    results = compute_roc_curve(samples)

    print(f"\n{'='*60}")
    print(f"  ROC Analysis Results")
    print(f"{'='*60}")
    print(f"  AUC:                         {results['auc']:.4f}")
    print(f"  Optimal threshold (Youden):  {results['optimal_threshold_youden']:.4f}")
    print(f"    - TPR at optimal:          {results['optimal_tpr']:.4f}")
    print(f"    - FPR at optimal:          {results['optimal_fpr']:.4f}")
    print(f"    - Youden index:            {results['optimal_youden_value']:.4f}")
    print(f"  Threshold for 90/10 split:   {results['threshold_90_10_split']:.4f}")
    print(f"{'='*60}")
    print(f"\n  Recommended config update:")
    print(f"    traffic_controller:")
    print(f"      cheap_rail_confidence_threshold: {1 - results['optimal_threshold_youden']:.4f}")
    print(f"      # Empirically derived via ROC analysis on {results['n_samples']} samples")
    print(f"      # AUC={results['auc']:.4f}, Youden={results['optimal_youden_value']:.4f}")

    # Save full results
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\n  Full results saved to: {args.output}")


if __name__ == "__main__":
    main()
