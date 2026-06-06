"""
src/api/routes.py

API routes:
  POST /webhook      — GitHub webhook receiver
  POST /analyze      — Direct diff analysis (for testing)
  GET  /findings/{pr} — Get cached findings for a PR
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request, status
from loguru import logger
from pydantic import BaseModel

from src.bot.github_bot import GitHubBot
from src.model.inference import get_engine
from src.parser.diff_parser import parse_diff

router = APIRouter()

WEBHOOK_SECRET = os.getenv("GITHUB_WEBHOOK_SECRET", "")

# In-memory cache (use Redis in production)
_findings_cache: dict[str, list[dict]] = {}


# ─── Models ──────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    diff:      str
    pr_number: int   = 0
    repo:      str   = ""


class FindingOut(BaseModel):
    file:        str
    line_start:  int
    line_end:    int
    pattern:     str
    severity:    str
    confidence:  float
    explanation: str
    suggestion:  str
    cwe:         Optional[str] = None


class AnalyzeResponse(BaseModel):
    pr_number:   int
    repo:        str
    total_hunks: int
    findings:    list[FindingOut]
    latency_ms:  float


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _verify_signature(payload: bytes, sig_header: str) -> bool:
    """Verify GitHub webhook HMAC-SHA256 signature."""
    if not WEBHOOK_SECRET:
        logger.warning("GITHUB_WEBHOOK_SECRET not set — skipping signature check")
        return True
    if not sig_header or not sig_header.startswith("sha256="):
        return False
    expected = "sha256=" + hmac.new(
        WEBHOOK_SECRET.encode(), payload, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, sig_header)


def _results_to_out(results) -> list[FindingOut]:
    out = []
    for result in results:
        for f in result.actionable_findings:
            out.append(FindingOut(
                file=f.hunk_file,
                line_start=f.line_start,
                line_end=f.line_end,
                pattern=f.pattern.value,
                severity=f.severity,
                confidence=f.confidence,
                explanation=f.explanation,
                suggestion=f.suggestion,
                cwe=f.cwe,
            ))
    return out


# ─── Routes ──────────────────────────────────────────────────────────────────

@router.post("/webhook")
async def github_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_hub_signature_256: str = Header(default=""),
    x_github_event: str     = Header(default=""),
):
    """Receive GitHub webhook events and trigger async review."""
    body = await request.body()

    if not _verify_signature(body, x_hub_signature_256):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid signature")

    if x_github_event != "pull_request":
        return {"message": f"Ignored event: {x_github_event}"}

    payload = json.loads(body)
    action  = payload.get("action", "")

    if action not in ("opened", "synchronize", "reopened"):
        return {"message": f"Ignored action: {action}"}

    pr      = payload["pull_request"]
    pr_num  = pr["number"]
    repo    = payload["repository"]["full_name"]
    diff_url = pr["diff_url"]

    logger.info(f"PR#{pr_num} {action} in {repo} — queuing review")
    background_tasks.add_task(_run_review, pr_num, repo, diff_url)

    return {"message": "Review queued", "pr": pr_num, "repo": repo}


async def _run_review(pr_number: int, repo: str, diff_url: str):
    """Background task: fetch diff, run model, post comments."""
    import httpx

    try:
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"token {os.getenv('GITHUB_TOKEN', '')}"}
            resp = await client.get(diff_url, headers=headers, timeout=30)
            resp.raise_for_status()
            raw_diff = resp.text

        parsed  = parse_diff(raw_diff, pr_number=pr_number, repo=repo)
        engine  = get_engine()
        results = engine.analyze_hunks(parsed.all_hunks)

        findings_out = _results_to_out(results)
        _findings_cache[f"{repo}#{pr_number}"] = [f.dict() for f in findings_out]

        bot = GitHubBot()
        await bot.post_review_comments(repo, pr_number, results)

        logger.info(
            f"PR#{pr_number}: posted {len(findings_out)} finding(s)"
        )

    except Exception as e:
        logger.error(f"Review failed for PR#{pr_number}: {e}")


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_diff(body: AnalyzeRequest):
    """
    Direct diff analysis endpoint — useful for testing without GitHub.

    Send a raw git diff and get back structured findings.
    """
    import time

    parsed  = parse_diff(body.diff, pr_number=body.pr_number, repo=body.repo)
    engine  = get_engine()

    t0      = time.perf_counter()
    results = engine.analyze_hunks(parsed.all_hunks)
    elapsed = (time.perf_counter() - t0) * 1000

    findings_out = _results_to_out(results)

    return AnalyzeResponse(
        pr_number=body.pr_number,
        repo=body.repo,
        total_hunks=len(parsed.all_hunks),
        findings=findings_out,
        latency_ms=round(elapsed, 1),
    )


@router.get("/findings/{repo:path}/{pr_number}")
async def get_findings(repo: str, pr_number: int):
    """Retrieve cached findings for a specific PR."""
    key = f"{repo}#{pr_number}"
    if key not in _findings_cache:
        raise HTTPException(status_code=404, detail="No findings cached for this PR")
    return {"pr_number": pr_number, "repo": repo, "findings": _findings_cache[key]}
