"""tests/test_parser.py - Lightweight tests, no ML dependencies"""

from src.parser.diff_parser import (
    parse_diff, detect_language, Language, chunk_hunk
)

SAMPLE_DIFF = """\
diff --git a/auth/session.py b/auth/session.py
index abc123..def456 100644
--- a/auth/session.py
+++ b/auth/session.py
@@ -38,7 +38,6 @@ class SessionManager:
   def validate_token(self, token):
-      query = f"SELECT * FROM users WHERE token = '{token}'"
+      query = "SELECT id FROM users WHERE token = %s"
"""


def test_parse_diff_returns_parsed():
    result = parse_diff(SAMPLE_DIFF, pr_number=1, repo="test/repo")
    assert result.pr_number == 1
    assert result.repo == "test/repo"


def test_parsed_file_language():
    result = parse_diff(SAMPLE_DIFF)
    assert result.files[0].language == Language.PYTHON


def test_parsed_file_path():
    result = parse_diff(SAMPLE_DIFF)
    assert result.files[0].file_path == "auth/session.py"


def test_detect_language():
    assert detect_language("foo.py")  == Language.PYTHON
    assert detect_language("bar.js")  == Language.JAVASCRIPT
    assert detect_language("baz.go")  == Language.GO
    assert detect_language("qux.rs")  == Language.UNKNOWN


def test_empty_diff():
    result = parse_diff("")
    assert len(result.files) == 0