"""tests/test_inference.py"""

import pytest
from src.model.inference import (
    AntiPattern,
    Finding,
    SEVERITY_MAP,
    _parse_response,
    build_prompt,
)
from src.parser.diff_parser import DiffHunk, Language


@pytest.fixture
def sample_hunk():
    return DiffHunk(
        file_path="auth/session.py",
        language=Language.PYTHON,
        old_start=38,
        new_start=38,
        removed_lines=["    query = f\"SELECT * FROM users WHERE token = '{token}'\""],
        added_lines=["    query = \"SELECT id FROM users WHERE token = %s\""],
        context_lines=["  def validate_token(self, token: str) -> bool:"],
        raw_hunk="@@ -38,3 +38,2 @@\n  def validate_token...",
    )


def test_build_prompt_contains_file(sample_hunk):
    prompt = build_prompt(sample_hunk)
    assert "auth/session.py" in prompt
    assert "python" in prompt.lower()


def test_build_prompt_is_string(sample_hunk):
    assert isinstance(build_prompt(sample_hunk), str)


def test_parse_response_valid_json(sample_hunk):
    raw = '{"findings": [{"pattern": "sql_injection", "confidence": 0.94, "line_start": 39, "line_end": 41, "explanation": "SQL injection via f-string", "suggestion": "Use parameterized query", "cwe": "CWE-89"}]}'
    findings = _parse_response(raw, sample_hunk)
    assert len(findings) == 1
    assert findings[0].pattern == AntiPattern.SQL_INJECTION
    assert findings[0].confidence == 0.94
    assert findings[0].cwe == "CWE-89"


def test_parse_response_clean(sample_hunk):
    raw = '{"findings": [{"pattern": "clean", "confidence": 0.97, "line_start": 1, "line_end": 1, "explanation": "No issues", "suggestion": "", "cwe": null}]}'
    findings = _parse_response(raw, sample_hunk)
    assert findings[0].pattern == AntiPattern.CLEAN


def test_parse_response_invalid_json(sample_hunk):
    findings = _parse_response("not json at all", sample_hunk)
    assert findings == []


def test_parse_response_with_preamble(sample_hunk):
    raw = 'Sure! Here is the analysis:\n{"findings": [{"pattern": "memory_leak", "confidence": 0.8, "line_start": 1, "line_end": 2, "explanation": "leak", "suggestion": "fix", "cwe": null}]}'
    findings = _parse_response(raw, sample_hunk)
    assert len(findings) == 1


def test_finding_to_comment(sample_hunk):
    f = Finding(
        hunk_file="auth/session.py",
        line_start=39,
        line_end=41,
        pattern=AntiPattern.SQL_INJECTION,
        severity="critical",
        confidence=0.94,
        explanation="SQL injection via f-string.",
        suggestion="Use parameterized query.",
        cwe="CWE-89",
    )
    comment = f.to_comment()
    assert "SQL Injection" in comment
    assert "CWE-89" in comment
    assert "94%" in comment


def test_finding_is_actionable(sample_hunk):
    f = Finding(
        hunk_file="f.py", line_start=1, line_end=1,
        pattern=AntiPattern.SQL_INJECTION,
        severity="critical", confidence=0.94,
        explanation="", suggestion="",
    )
    assert f.is_actionable is True


def test_finding_not_actionable_below_threshold(sample_hunk):
    f = Finding(
        hunk_file="f.py", line_start=1, line_end=1,
        pattern=AntiPattern.CODE_SMELL,
        severity="low", confidence=0.2,
        explanation="", suggestion="",
    )
    assert f.is_actionable is False


def test_severity_map_complete():
    for pattern in AntiPattern:
        assert pattern in SEVERITY_MAP
