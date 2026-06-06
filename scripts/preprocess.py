"""
scripts/preprocess.py

Converts raw collected PR diffs into fine-tuning ready JSONL format.
Extracts per-hunk labels from review comments using keyword matching
and regex patterns.

Usage:
    python scripts/preprocess.py \
        --input  data/raw/pr_diffs.jsonl \
        --output data/processed/
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from loguru import logger
from tqdm import tqdm


# ─── Label Extraction ────────────────────────────────────────────────────────

PATTERN_KEYWORDS: dict[str, list[str]] = {
    "sql_injection": [
        "sql injection", "sqli", "cwe-89", "f-string.*sql", "format.*query",
        "parameterize", "prepared statement",
    ],
    "memory_leak": [
        "memory leak", "not freed", "not released", "clearinterval",
        "cleartimeout", "resource leak", "unclosed",
    ],
    "hardcoded_credentials": [
        "hardcoded", "hard-coded", "password in code", "secret in code",
        "api key in", "token in code",
    ],
    "unhandled_exception": [
        "unhandled exception", "bare except", "except:", "swallow.*error",
        "missing error handling", "no error handling",
    ],
    "race_condition": [
        "race condition", "thread safety", "concurrent", "synchronize",
        "mutex", "lock", "atomic",
    ],
    "n_plus_one_query": [
        "n+1", "n + 1", "query in loop", "select_related", "prefetch",
        "eager load",
    ],
    "xss": [
        "xss", "cross-site scripting", "cwe-79", "sanitize", "escape html",
        "innerhtml",
    ],
    "path_traversal": [
        "path traversal", "directory traversal", "cwe-22", "../",
        "os.path.join.*user",
    ],
}

COMPILED = {
    label: [re.compile(p, re.IGNORECASE) for p in patterns]
    for label, patterns in PATTERN_KEYWORDS.items()
}


def extract_label(comment_bodies: list[str]) -> str:
    """Return the best matching anti-pattern label from review comments."""
    full_text = " ".join(comment_bodies).lower()
    for label, patterns in COMPILED.items():
        for pat in patterns:
            if pat.search(full_text):
                return label
    return "clean"


def build_label_json(label: str) -> str:
    """Build the expected JSON output for this label."""
    if label == "clean":
        return json.dumps({"findings": [{"pattern": "clean", "confidence": 0.95,
                                          "line_start": 1, "line_end": 1,
                                          "explanation": "No issues found.",
                                          "suggestion": "", "cwe": None}]})
    templates = {
        "sql_injection": ("SQL injection via string interpolation in query.",
                          "Use parameterized queries.", "CWE-89"),
        "memory_leak":   ("Resource not properly released.",
                          "Ensure cleanup in finally/dispose.", None),
        "hardcoded_credentials": ("Credentials hardcoded in source.",
                                   "Use environment variables or secret manager.", "CWE-798"),
        "unhandled_exception": ("Exception may be swallowed silently.",
                                 "Log or re-raise the exception.", None),
        "race_condition": ("Shared state accessed without synchronization.",
                            "Use locks or atomic operations.", None),
        "n_plus_one_query": ("Query executed inside a loop (N+1 pattern).",
                              "Use eager loading / batch query.", None),
        "xss": ("User input rendered without escaping.",
                 "Sanitize output with proper escaping.", "CWE-79"),
        "path_traversal": ("User-controlled path may escape base directory.",
                            "Validate and normalize paths.", "CWE-22"),
    }
    expl, sugg, cwe = templates.get(label, ("Anti-pattern detected.", "Review this code.", None))
    return json.dumps({"findings": [{"pattern": label, "confidence": 0.85,
                                      "line_start": 1, "line_end": 5,
                                      "explanation": expl,
                                      "suggestion": sugg, "cwe": cwe}]})


# ─── Main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",      default="data/raw/pr_diffs.jsonl")
    p.add_argument("--output",     default="data/processed/")
    p.add_argument("--train-ratio", type=float, default=0.85)
    p.add_argument("--val-ratio",   type=float, default=0.10)
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


def preprocess(args):
    random.seed(args.seed)
    Path(args.output).mkdir(parents=True, exist_ok=True)

    records = []
    with open(args.input) as f:
        for line in tqdm(f, desc="Parsing raw diffs"):
            row = json.loads(line)
            comment_bodies = [c["body"] for c in row.get("comments", [])]
            label      = extract_label(comment_bodies)
            label_json = build_label_json(label)

            records.append({
                "repo":       row["repo"],
                "pr_number":  row["pr_number"],
                "language":   row["language"],
                "diff":       row["diff"],
                "label":      label,
                "label_json": label_json,
            })

    random.shuffle(records)
    n = len(records)
    n_train = int(n * args.train_ratio)
    n_val   = int(n * args.val_ratio)

    splits = {
        "train": records[:n_train],
        "val":   records[n_train:n_train + n_val],
        "test":  records[n_train + n_val:],
    }

    for split_name, rows in splits.items():
        out_path = Path(args.output) / f"{split_name}.jsonl"
        with open(out_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        logger.info(f"{split_name}: {len(rows)} examples → {out_path}")

    # Label distribution
    from collections import Counter
    dist = Counter(r["label"] for r in records)
    logger.info("Label distribution:")
    for label, count in dist.most_common():
        logger.info(f"  {label:35s}: {count:6d} ({count/n*100:.1f}%)")


if __name__ == "__main__":
    preprocess(parse_args())
