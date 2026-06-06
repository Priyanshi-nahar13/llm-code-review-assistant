"""tests/test_parser.py"""

import pytest
from src.parser.diff_parser import parse_diff, detect_language, Language, chunk_hunk


SAMPLE_DIFF = """\
diff --git a/auth/session.py b/auth/session.py
index abc123..def456 100644
--- a/auth/session.py
+++ b/auth/session.py
@@ -38,7 +38,6 @@ class SessionManager:
   def validate_token(self, token: str) -> bool:
-      query = f"SELECT * FROM users WHERE token = '{token}'"
-      result = self.db.execute(query)
-      return result.fetchone() is not None
+      query = "SELECT id FROM users WHERE token = %s"
+      return self.db.execute(query, (token,)).fetchone() is not None
"""


def test_parse_diff_returns_parsed():
    result = parse_diff(SAMPLE_DIFF, pr_number=1, repo="test/repo")
    assert result.pr_number == 1
    assert result.repo == "test/repo"
    assert len(result.files) == 1


def test_parsed_file_language():
    result = parse_diff(SAMPLE_DIFF)
    assert result.files[0].language == Language.PYTHON


def test_parsed_file_path():
    result = parse_diff(SAMPLE_DIFF)
    assert result.files[0].file_path == "auth/session.py"


def test_hunks_parsed():
    result = parse_diff(SAMPLE_DIFF)
    assert len(result.files[0].hunks) == 1
    hunk = result.files[0].hunks[0]
    assert len(hunk.removed_lines) == 3
    assert len(hunk.added_lines) == 2


def test_detect_language():
    assert detect_language("foo.py")  == Language.PYTHON
    assert detect_language("bar.js")  == Language.JAVASCRIPT
    assert detect_language("baz.go")  == Language.GO
    assert detect_language("qux.rs")  == Language.UNKNOWN


def test_unknown_language_filtered():
    diff = SAMPLE_DIFF.replace("session.py", "session.rs")
    result = parse_diff(diff)
    assert len(result.files) == 0  # .rs is UNKNOWN, filtered out


def test_chunk_hunk_small():
    result = parse_diff(SAMPLE_DIFF)
    hunk   = result.files[0].hunks[0]
    chunks = chunk_hunk(hunk, max_tokens=2048)
    assert len(chunks) == 1
    assert chunks[0] == hunk.full_context


def test_all_hunks_property():
    result = parse_diff(SAMPLE_DIFF)
    assert len(result.all_hunks) == 1


def test_empty_diff():
    result = parse_diff("")
    assert len(result.files) == 0
    assert len(result.all_hunks) == 0
