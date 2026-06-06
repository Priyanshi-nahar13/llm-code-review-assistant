"""
evaluation/evaluate.py

Evaluates the fine-tuned model on the held-out test set.
Outputs per-category precision, recall, F1, and overall metrics.

Usage:
    python evaluation/evaluate.py \
        --model  models/codellama-review-lora \
        --test   data/processed/test.jsonl \
        --output evaluation/results/
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

from src.model.inference import AntiPattern, InferenceEngine
from src.parser.diff_parser import DiffHunk, Language


# ─── Args ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",  default="models/codellama-review-lora")
    p.add_argument("--test",   default="data/processed/test.jsonl")
    p.add_argument("--output", default="evaluation/results/")
    p.add_argument("--limit",  type=int, default=None, help="Cap test examples")
    return p.parse_args()


# ─── Load Test Data ──────────────────────────────────────────────────────────

def load_test_data(path: str, limit=None) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    logger.info(f"Loaded {len(rows)} test examples")
    return rows


def row_to_hunk(row: dict) -> DiffHunk:
    lang_map = {
        "python":     Language.PYTHON,
        "javascript": Language.JAVASCRIPT,
        "go":         Language.GO,
    }
    return DiffHunk(
        file_path=row.get("file_path", "test.py"),
        language=lang_map.get(row.get("language", "python"), Language.PYTHON),
        old_start=row.get("old_start", 1),
        new_start=row.get("new_start", 1),
        removed_lines=row.get("removed_lines", []),
        added_lines=row.get("added_lines", []),
        context_lines=row.get("context_lines", []),
        raw_hunk=row.get("diff", ""),
    )


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate(args):
    Path(args.output).mkdir(parents=True, exist_ok=True)

    test_data = load_test_data(args.test, args.limit)
    engine    = InferenceEngine()

    y_true, y_pred = [], []

    for row in tqdm(test_data, desc="Evaluating"):
        hunk       = row_to_hunk(row)
        true_label = row.get("label", "clean")
        results    = engine.analyze_hunks([hunk])

        # Take highest-confidence non-clean prediction, else "clean"
        top_finding = None
        for r in results:
            for f in r.findings:
                if f.pattern.value != "clean":
                    if top_finding is None or f.confidence > top_finding.confidence:
                        top_finding = f

        pred_label = top_finding.pattern.value if top_finding else "clean"

        y_true.append(true_label)
        y_pred.append(pred_label)

    # ── Metrics ───────────────────────────────────────────────────────────────
    labels = [p.value for p in AntiPattern if p != AntiPattern.CLEAN]

    report = classification_report(
        y_true, y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    weighted_f1  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_pre = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)

    # False positive rate (predicted issue when label is clean)
    clean_mask = np.array([t == "clean" for t in y_true])
    fp_count   = sum(
        1 for t, p in zip(y_true, y_pred)
        if t == "clean" and p != "clean"
    )
    fpr = fp_count / max(clean_mask.sum(), 1)

    summary = {
        "weighted_f1":       round(weighted_f1,  4),
        "weighted_precision":round(weighted_pre, 4),
        "weighted_recall":   round(weighted_rec, 4),
        "false_positive_rate": round(fpr, 4),
        "total_examples":    len(y_true),
        "per_category":      report,
    }

    # ── Print ─────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"Weighted F1:        {weighted_f1:.3f}")
    logger.info(f"Weighted Precision: {weighted_pre:.3f}")
    logger.info(f"Weighted Recall:    {weighted_rec:.3f}")
    logger.info(f"False Positive Rate:{fpr:.3f} ({fp_count}/{int(clean_mask.sum())})")
    logger.info("=" * 60)
    logger.info("\n" + classification_report(y_true, y_pred, zero_division=0))

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = Path(args.output) / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Results saved to {out_path}")

    # Save predictions
    preds_path = Path(args.output) / "predictions.jsonl"
    with open(preds_path, "w") as f:
        for row, true, pred in zip(test_data, y_true, y_pred):
            f.write(json.dumps({
                "file": row.get("file_path"),
                "true": true,
                "pred": pred,
                "correct": true == pred,
            }) + "\n")
    logger.info(f"Predictions saved to {preds_path}")


if __name__ == "__main__":
    evaluate(parse_args())
