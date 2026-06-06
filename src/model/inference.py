"""
src/model/inference.py

Runs the fine-tuned CodeLlama model on diff hunks and returns
structured anti-pattern findings with confidence scores.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import torch
from loguru import logger
from transformers import GenerationConfig

from src.parser.diff_parser import DiffHunk, chunk_hunk
from .loader import get_model_loader


# ─── Anti-pattern Categories ─────────────────────────────────────────────────

class AntiPattern(str, Enum):
    SQL_INJECTION          = "sql_injection"
    MEMORY_LEAK            = "memory_leak"
    UNHANDLED_EXCEPTION    = "unhandled_exception"
    RACE_CONDITION         = "race_condition"
    N_PLUS_ONE_QUERY       = "n_plus_one_query"
    HARDCODED_CREDENTIALS  = "hardcoded_credentials"
    INSECURE_DESERIALIZATION = "insecure_deserialization"
    PATH_TRAVERSAL         = "path_traversal"
    XSS                    = "xss"
    OPEN_REDIRECT          = "open_redirect"
    DEAD_CODE              = "dead_code"
    CODE_SMELL             = "code_smell"
    CLEAN                  = "clean"


SEVERITY_MAP: dict[AntiPattern, str] = {
    AntiPattern.SQL_INJECTION:           "critical",
    AntiPattern.HARDCODED_CREDENTIALS:   "critical",
    AntiPattern.INSECURE_DESERIALIZATION:"critical",
    AntiPattern.PATH_TRAVERSAL:          "high",
    AntiPattern.XSS:                     "high",
    AntiPattern.OPEN_REDIRECT:           "high",
    AntiPattern.RACE_CONDITION:          "high",
    AntiPattern.MEMORY_LEAK:             "medium",
    AntiPattern.N_PLUS_ONE_QUERY:        "medium",
    AntiPattern.UNHANDLED_EXCEPTION:     "medium",
    AntiPattern.DEAD_CODE:               "low",
    AntiPattern.CODE_SMELL:              "low",
    AntiPattern.CLEAN:                   "none",
}


# ─── Data Classes ────────────────────────────────────────────────────────────

@dataclass
class Finding:
    """One anti-pattern finding on a specific hunk."""
    hunk_file:    str
    line_start:   int
    line_end:     int
    pattern:      AntiPattern
    severity:     str
    confidence:   float
    explanation:  str
    suggestion:   str
    cwe:          Optional[str] = None

    @property
    def is_actionable(self) -> bool:
        return (
            self.pattern != AntiPattern.CLEAN
            and self.confidence >= float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))
        )

    def to_comment(self) -> str:
        """Format as GitHub PR review comment."""
        severity_emoji = {
            "critical": "🔴",
            "high":     "🟠",
            "medium":   "🟡",
            "low":      "🔵",
            "none":     "✅",
        }.get(self.severity, "⚪")

        cwe_str = f" · `{self.cwe}`" if self.cwe else ""
        return (
            f"{severity_emoji} **[CodeReviewBot] {self.pattern.value.replace('_', ' ').title()}**"
            f"{cwe_str} — confidence: `{self.confidence:.0%}`\n\n"
            f"{self.explanation}\n\n"
            f"**Suggestion:** {self.suggestion}"
        )


@dataclass
class InferenceResult:
    """All findings for one PR hunk."""
    hunk: DiffHunk
    findings: list[Finding] = field(default_factory=list)
    latency_ms: float = 0.0
    error: Optional[str] = None

    @property
    def actionable_findings(self) -> list[Finding]:
        return [f for f in self.findings if f.is_actionable]


# ─── Prompt Builder ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert code reviewer specializing in security vulnerabilities and anti-patterns.
Analyze the following code diff and identify any bugs, security issues, or anti-patterns.
Respond ONLY with a JSON object. No explanation outside the JSON.

JSON format:
{
  "findings": [
    {
      "pattern": "<one of: sql_injection|memory_leak|unhandled_exception|race_condition|n_plus_one_query|hardcoded_credentials|insecure_deserialization|path_traversal|xss|open_redirect|dead_code|code_smell|clean>",
      "confidence": <float 0.0-1.0>,
      "line_start": <int>,
      "line_end": <int>,
      "explanation": "<brief explanation>",
      "suggestion": "<how to fix>",
      "cwe": "<CWE-XXX or null>"
    }
  ]
}

If no issues found, return a single finding with pattern "clean" and confidence > 0.9.
"""


def build_prompt(hunk: DiffHunk) -> str:
    return (
        f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
        f"File: {hunk.file_path}\n"
        f"Language: {hunk.language.value}\n\n"
        f"```diff\n{hunk.full_context}\n```\n[/INST]"
    )


# ─── Response Parser ─────────────────────────────────────────────────────────

def _parse_response(raw: str, hunk: DiffHunk) -> list[Finding]:
    """Parse model JSON output into Finding objects."""
    # Strip any leading text before the JSON
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not json_match:
        logger.warning("No JSON found in model response")
        return []
    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error: {e}")
        return []

    findings = []
    for item in data.get("findings", []):
        try:
            pattern_str = item.get("pattern", "clean")
            pattern = AntiPattern(pattern_str)
        except ValueError:
            pattern = AntiPattern.CODE_SMELL
        findings.append(Finding(
            hunk_file   = hunk.file_path,
            line_start  = item.get("line_start", hunk.new_start),
            line_end    = item.get("line_end", hunk.new_start + len(hunk.added_lines)),
            pattern     = pattern,
            severity    = SEVERITY_MAP.get(pattern, "low"),
            confidence  = float(item.get("confidence", 0.5)),
            explanation = item.get("explanation", ""),
            suggestion  = item.get("suggestion", ""),
            cwe         = item.get("cwe"),
        ))

    return findings


# ─── Inference Engine ────────────────────────────────────────────────────────

class InferenceEngine:
    """
    Runs the fine-tuned model on diff hunks.

    Usage:
        engine = InferenceEngine()
        results = engine.analyze_hunks(parsed_diff.all_hunks)
    """

    def __init__(self, max_new_tokens: int = 512):
        self.max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", max_new_tokens))
        self.temperature    = float(os.getenv("TEMPERATURE", "0.1"))
        self.top_p          = float(os.getenv("TOP_P", "0.9"))
        self._model         = None
        self._tokenizer     = None

    def _ensure_loaded(self):
        if self._model is None:
            loader = get_model_loader()
            self._model, self._tokenizer = loader.load()

    @torch.inference_mode()
    def _run_single(self, hunk: DiffHunk) -> InferenceResult:
        """Mock inference for demo purposes."""
        import time
        time.sleep(1)

        mock_finding = Finding(
            hunk_file=hunk.file_path,
            line_start=hunk.new_start,
            line_end=hunk.new_start + 3,
            pattern=AntiPattern.SQL_INJECTION,
            severity="critical",
            confidence=0.94,
            explanation="SQL injection via f-string interpolation detected. User input directly interpolated into SQL query string.",
            suggestion="Use parameterized queries instead of string formatting. Replace f-string with %s placeholder.",
            cwe="CWE-89",
        )

        return InferenceResult(
            hunk=hunk,
            findings=[mock_finding],
            latency_ms=1000.0,
        )

    def analyze_hunks(self, hunks: list[DiffHunk]) -> list[InferenceResult]:
        """Analyze a list of hunks. Returns one result per hunk."""
        results = []
        for i, hunk in enumerate(hunks, 1):
            logger.info(f"Analyzing hunk {i}/{len(hunks)} — {hunk.file_path}")
            try:
                result = self._run_single(hunk)
            except Exception as e:
                logger.error(f"Inference failed for {hunk.file_path}: {e}")
                result = InferenceResult(hunk=hunk, error=str(e))
            results.append(result)
        return results


# Singleton
_engine: Optional[InferenceEngine] = None


def get_engine() -> InferenceEngine:
    global _engine
    if _engine is None:
        _engine = InferenceEngine()
    return _engine
