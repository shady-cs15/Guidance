"""Unit tests for helpers in agent_func_lean_minif2f.py.

These tests are *fast* — they don't spawn any Lean process.
"""

import json
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

# Make sure the examples module is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from examples.python.agent_func_lean_minif2f import (
    _build_lean_file,
    _extract_tactic,
    _find_traced_repo_path,
    _get_theorem_map,
)


# ── _extract_tactic ────────────────────────────────────────────────────


class TestExtractTactic:
    """Test tactic extraction from LLM-style outputs."""

    def test_lean_code_block(self):
        text = "Let me try this:\n```lean\nomega\n```\nThat should work."
        assert _extract_tactic(text) == "omega"

    def test_generic_code_block(self):
        text = "Here's a tactic:\n```\nring\n```"
        assert _extract_tactic(text) == "ring"

    def test_raw_text(self):
        assert _extract_tactic("  omega  ") == "omega"

    def test_multiline_code_block(self):
        text = "```lean\napply h₀\nexact h₁\n```"
        assert _extract_tactic(text) == "apply h₀\nexact h₁"

    def test_empty_string(self):
        assert _extract_tactic("") == ""

    def test_whitespace_only(self):
        assert _extract_tactic("   \n\n  ") == ""

    def test_code_block_with_language_tag(self):
        text = "Try:\n```lean\nsimp [h₀, h₁]\n```"
        assert _extract_tactic(text) == "simp [h₀, h₁]"

    def test_prefers_lean_block_over_generic(self):
        text = "```lean\nomega\n```\n\nOr alternatively:\n```\nring\n```"
        assert _extract_tactic(text) == "omega"

    def test_code_block_with_extra_whitespace(self):
        text = "```lean\n  simp only [Nat.add_comm]  \n```"
        assert _extract_tactic(text) == "simp only [Nat.add_comm]"

    def test_complex_tactic(self):
        text = "```lean\nconstructor\n· intro h\n  exact h.left\n· intro h\n  exact ⟨h, rfl⟩\n```"
        result = _extract_tactic(text)
        assert "constructor" in result
        assert "intro h" in result


# ── _build_lean_file ───────────────────────────────────────────────────


class TestBuildLeanFile:
    """Test Lean source file construction."""

    def test_basic_structure(self):
        header = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators\n\n"
        stmt = "theorem foo (n : ℕ) : n = n := by"
        result = _build_lean_file(header, stmt)

        assert "import Lean4Repl" in result
        assert "import Mathlib" in result
        assert "set_option maxHeartbeats 0 in" in result
        assert "theorem foo" in result
        assert "lean_dojo_repl" in result
        assert "sorry" in result

    def test_imports_lean4repl_first(self):
        header = "import Mathlib\n"
        stmt = "theorem bar : True := by"
        result = _build_lean_file(header, stmt)
        lines = result.split("\n")
        assert lines[0] == "import Lean4Repl"

    def test_statement_preserved(self):
        header = "import Mathlib\n"
        stmt = "theorem baz (x y : ℤ) (h : x + y = 80) : x = 26 := by"
        result = _build_lean_file(header, stmt)
        assert stmt.rstrip() in result


# ── _get_theorem_map (with mocking) ───────────────────────────────────


class TestGetTheoremMap:
    """Test theorem map loading."""

    def test_loads_from_file(self, tmp_path):
        fake_map = {
            "thm_a": {
                "repo_url": "https://github.com/test/repo",
                "commit": "abc123",
                "file_path": "Test/A.lean",
                "full_name": "thm_a",
            }
        }
        p = tmp_path / "theorem_map.json"
        p.write_text(json.dumps(fake_map))

        import examples.python.agent_func_lean_minif2f as mod

        # Reset the singleton
        old_map = mod._theorem_map
        old_path = mod.THEOREM_MAP_PATH
        try:
            mod._theorem_map = None
            mod.THEOREM_MAP_PATH = str(p)
            result = _get_theorem_map()
            assert "thm_a" in result
            assert result["thm_a"]["commit"] == "abc123"
        finally:
            mod._theorem_map = old_map
            mod.THEOREM_MAP_PATH = old_path

    def test_caches_result(self, tmp_path):
        """Second call should return the cached dict (same object)."""
        fake_map = {"x": {"repo_url": "", "commit": "", "file_path": "", "full_name": "x"}}
        p = tmp_path / "map.json"
        p.write_text(json.dumps(fake_map))

        import examples.python.agent_func_lean_minif2f as mod

        old_map = mod._theorem_map
        old_path = mod.THEOREM_MAP_PATH
        try:
            mod._theorem_map = None
            mod.THEOREM_MAP_PATH = str(p)
            first = _get_theorem_map()
            second = _get_theorem_map()
            assert first is second  # same object → cached
        finally:
            mod._theorem_map = old_map
            mod.THEOREM_MAP_PATH = old_path


# ── _find_traced_repo_path ────────────────────────────────────────────


class TestFindTracedRepoPath:
    """Test traced repo path resolution."""

    def test_finds_correct_path(self, tmp_path):
        # Simulate cache layout: <cache>/owner-repo-commit/repo/
        slug = "yangky11-miniF2F-lean4-abc123"
        repo_dir = tmp_path / slug / "miniF2F-lean4"
        repo_dir.mkdir(parents=True)

        import examples.python.agent_func_lean_minif2f as mod

        old_cache = mod.LEAN_DOJO_CACHE
        try:
            mod.LEAN_DOJO_CACHE = str(tmp_path)
            result = _find_traced_repo_path(
                "https://github.com/yangky11/miniF2F-lean4", "abc123"
            )
            assert result == repo_dir
        finally:
            mod.LEAN_DOJO_CACHE = old_cache

    def test_raises_on_missing(self, tmp_path):
        import examples.python.agent_func_lean_minif2f as mod

        old_cache = mod.LEAN_DOJO_CACHE
        try:
            mod.LEAN_DOJO_CACHE = str(tmp_path)
            with pytest.raises(FileNotFoundError):
                _find_traced_repo_path(
                    "https://github.com/nobody/nothing", "deadbeef"
                )
        finally:
            mod.LEAN_DOJO_CACHE = old_cache
