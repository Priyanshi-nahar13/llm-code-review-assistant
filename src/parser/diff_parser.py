"""
src/parser/diff_parser.py

Parses raw git diffs into structured chunks using tree-sitter.
Supports Python, JavaScript, and Go.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from loguru import logger


class Language(str, Enum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    GO = "go"
    UNKNOWN = "unknown"


@dataclass
class DiffHunk:
    """One contiguous changed block inside a file diff."""
    file_path: str
    language: Language
    old_start: int
    new_start: int
    removed_lines: list[str]
    added_lines: list[str]
    context_lines: list[str]
    raw_hunk: str

    @property
    def changed_content(self) -> str:
        """All changed lines (removed + added) as a single string."""
        return "\n".join(self.removed_lines + self.added_lines)

    @property
    def full_context(self) -> str:
        """Full hunk with context, for model input."""
        return self.raw_hunk


@dataclass
class FileDiff:
    """All hunks for one file in a PR diff."""
    file_path: str
    language: Language
    hunks: list[DiffHunk] = field(default_factory=list)
    is_new_file: bool = False
    is_deleted_file: bool = False

    @property
    def total_additions(self) -> int:
        return sum(len(h.added_lines) for h in self.hunks)

    @property
    def total_deletions(self) -> int:
        return sum(len(h.removed_lines) for h in self.hunks)


@dataclass
class ParsedDiff:
    """Full parsed diff for one PR."""
    pr_number: int
    repo: str
    files: list[FileDiff] = field(default_factory=list)

    @property
    def all_hunks(self) -> list[DiffHunk]:
        return [hunk for f in self.files for hunk in f.hunks]

    @property
    def languages(self) -> set[Language]:
        return {f.language for f in self.files}


# ─── Language Detection ──────────────────────────────────────────────────────

EXTENSION_MAP: dict[str, Language] = {
    ".py":   Language.PYTHON,
    ".pyw":  Language.PYTHON,
    ".js":   Language.JAVASCRIPT,
    ".jsx":  Language.JAVASCRIPT,
    ".ts":   Language.JAVASCRIPT,
    ".tsx":  Language.JAVASCRIPT,
    ".go":   Language.GO,
}


def detect_language(file_path: str) -> Language:
    """Detect programming language from file extension."""
    ext = Path(file_path).suffix.lower()
    return EXTENSION_MAP.get(ext, Language.UNKNOWN)


# ─── Hunk Parsing ────────────────────────────────────────────────────────────

HUNK_HEADER_RE = re.compile(
    r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@"
)


def _parse_hunk(hunk_lines: list[str], file_path: str, language: Language) -> DiffHunk:
    """Convert raw hunk lines into a DiffHunk object."""
    header = hunk_lines[0]
    m = HUNK_HEADER_RE.match(header)
    old_start = int(m.group(1)) if m else 0
    new_start = int(m.group(2)) if m else 0

    removed, added, context = [], [], []
    for line in hunk_lines[1:]:
        if line.startswith("-"):
            removed.append(line[1:])
        elif line.startswith("+"):
            added.append(line[1:])
        else:
            context.append(line.lstrip(" "))

    return DiffHunk(
        file_path=file_path,
        language=language,
        old_start=old_start,
        new_start=new_start,
        removed_lines=removed,
        added_lines=added,
        context_lines=context,
        raw_hunk="\n".join(hunk_lines),
    )


# ─── File Diff Parsing ───────────────────────────────────────────────────────

FILE_HEADER_RE = re.compile(r"^diff --git a/(.+) b/(.+)$")
NEW_FILE_RE    = re.compile(r"^new file mode")
DEL_FILE_RE    = re.compile(r"^deleted file mode")


def _parse_file_section(section_lines: list[str]) -> Optional[FileDiff]:
    """Parse one file's section of a git diff."""
    if not section_lines:
        return None

    # Extract file path
    file_path = "unknown"
    is_new = False
    is_del = False

    for line in section_lines[:6]:
        m = FILE_HEADER_RE.match(line)
        if m:
            file_path = m.group(2)
        if NEW_FILE_RE.match(line):
            is_new = True
        if DEL_FILE_RE.match(line):
            is_del = True

    language = detect_language(file_path)
    file_diff = FileDiff(
        file_path=file_path,
        language=language,
        is_new_file=is_new,
        is_deleted_file=is_del,
    )

    # Split into hunks
    hunk_starts = [
        i for i, l in enumerate(section_lines)
        if HUNK_HEADER_RE.match(l)
    ]

    for idx, start in enumerate(hunk_starts):
        end = hunk_starts[idx + 1] if idx + 1 < len(hunk_starts) else len(section_lines)
        hunk_lines = section_lines[start:end]
        hunk = _parse_hunk(hunk_lines, file_path, language)
        file_diff.hunks.append(hunk)

    return file_diff


# ─── Main Parser ─────────────────────────────────────────────────────────────

def parse_diff(raw_diff: str, pr_number: int = 0, repo: str = "") -> ParsedDiff:
    """
    Parse a raw git diff string into a structured ParsedDiff.

    Args:
        raw_diff:  Full text of `git diff` output.
        pr_number: GitHub PR number (for tracking).
        repo:      Repo name (owner/repo).

    Returns:
        ParsedDiff with all files and hunks.
    """
    parsed = ParsedDiff(pr_number=pr_number, repo=repo)
    lines = raw_diff.splitlines()

    # Split into per-file sections
    file_sections: list[list[str]] = []
    current: list[str] = []

    for line in lines:
        if FILE_HEADER_RE.match(line) and current:
            file_sections.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        file_sections.append(current)

    for section in file_sections:
        file_diff = _parse_file_section(section)
        if file_diff and file_diff.language != Language.UNKNOWN:
            parsed.files.append(file_diff)
            logger.debug(
                f"Parsed {file_diff.file_path} "
                f"({file_diff.total_additions}+ / {file_diff.total_deletions}-)"
            )

    logger.info(
        f"PR#{pr_number}: parsed {len(parsed.files)} files, "
        f"{len(parsed.all_hunks)} hunks"
    )
    return parsed


# ─── Token-safe Chunking ─────────────────────────────────────────────────────

def chunk_hunk(hunk: DiffHunk, max_tokens: int = 2048) -> list[str]:
    """
    Split a large hunk into token-safe chunks for model input.
    Rough estimate: 1 token ≈ 4 chars.
    """
    text = hunk.full_context
    char_limit = max_tokens * 4

    if len(text) <= char_limit:
        return [text]

    chunks = []
    lines = text.splitlines(keepends=True)
    current_chunk = ""

    for line in lines:
        if len(current_chunk) + len(line) > char_limit:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = line
        else:
            current_chunk += line

    if current_chunk:
        chunks.append(current_chunk)

    return chunks
