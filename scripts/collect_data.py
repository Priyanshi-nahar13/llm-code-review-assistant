"""
scripts/collect_data.py

Collects PR diffs from top GitHub repos for training data.

Usage:
    python scripts/collect_data.py \
        --limit 80000 \
        --output data/raw/ \
        --languages python javascript go
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from github import Github, RateLimitExceededException
from loguru import logger
from tqdm import tqdm


# ─── Top repos per language ───────────────────────────────────────────────────

SEED_REPOS = {
    "python": [
        "psf/requests", "pallets/flask", "django/django",
        "fastapi/fastapi", "pytorch/pytorch", "scikit-learn/scikit-learn",
        "pandas-dev/pandas", "numpy/numpy", "keras-team/keras",
        "celery/celery",
    ],
    "javascript": [
        "facebook/react", "vuejs/vue", "expressjs/express",
        "nodejs/node", "vercel/next.js", "nestjs/nest",
        "axios/axios", "lodash/lodash", "webpack/webpack",
    ],
    "go": [
        "gin-gonic/gin", "kubernetes/kubernetes", "moby/moby",
        "hashicorp/terraform", "gofiber/fiber", "labstack/echo",
    ],
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--limit",     type=int, default=80000)
    p.add_argument("--output",    default="data/raw/")
    p.add_argument("--languages", nargs="+", default=["python", "javascript", "go"])
    p.add_argument("--min-review-comments", type=int, default=1,
                   help="Only collect PRs that have at least N review comments")
    return p.parse_args()


def collect(args):
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        logger.error("GITHUB_TOKEN not set")
        return

    g = Github(token, per_page=100)
    Path(args.output).mkdir(parents=True, exist_ok=True)

    out_path = Path(args.output) / "pr_diffs.jsonl"
    collected = 0

    repos = []
    for lang in args.languages:
        repos.extend([(r, lang) for r in SEED_REPOS.get(lang, [])])

    with open(out_path, "w") as f_out:
        for repo_name, language in tqdm(repos, desc="Repos"):
            if collected >= args.limit:
                break
            try:
                repo = g.get_repo(repo_name)
                prs  = repo.get_pulls(state="closed", sort="updated", direction="desc")

                for pr in prs:
                    if collected >= args.limit:
                        break
                    if not pr.merged:
                        continue
                    if pr.review_comments < args.min_review_comments:
                        continue

                    try:
                        # Get raw diff
                        import requests as req
                        headers = {
                            "Authorization": f"token {token}",
                            "Accept": "application/vnd.github.v3.diff",
                        }
                        resp = req.get(pr.diff_url, headers=headers, timeout=15)
                        if resp.status_code != 200:
                            continue
                        raw_diff = resp.text[:50_000]   # cap at 50k chars

                        # Get review comments for labels
                        comments = [
                            {"body": c.body, "path": c.path, "line": c.original_line}
                            for c in pr.get_review_comments()
                        ]

                        record = {
                            "repo":      repo_name,
                            "pr_number": pr.number,
                            "language":  language,
                            "diff":      raw_diff,
                            "comments":  comments,
                            "merged_at": str(pr.merged_at),
                        }
                        f_out.write(json.dumps(record) + "\n")
                        collected += 1

                        # Respect rate limits
                        time.sleep(0.3)

                    except Exception as e:
                        logger.warning(f"PR#{pr.number} in {repo_name}: {e}")
                        continue

            except RateLimitExceededException:
                logger.warning("Rate limited — sleeping 60s")
                time.sleep(60)
            except Exception as e:
                logger.error(f"{repo_name}: {e}")
                continue

    logger.info(f"Collected {collected} PRs → {out_path}")


if __name__ == "__main__":
    collect(parse_args())
