"""Shared fixtures for the Lean agent tests."""

import json
import pathlib

import pytest

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
VALID_JSONL = DATA_DIR / "minif2f_valid.jsonl"
TEST_JSONL = DATA_DIR / "minif2f_test.jsonl"


def _load_jsonl(path: pathlib.Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


@pytest.fixture(scope="session")
def valid_rows():
    """All rows from data/minif2f_valid.jsonl."""
    if not VALID_JSONL.exists():
        pytest.skip(f"{VALID_JSONL} not found — run setup_minif2f_leandojo.py first")
    return _load_jsonl(VALID_JSONL)


@pytest.fixture(scope="session")
def test_rows():
    """All rows from data/minif2f_test.jsonl."""
    if not TEST_JSONL.exists():
        pytest.skip(f"{TEST_JSONL} not found — run setup_minif2f_leandojo.py first")
    return _load_jsonl(TEST_JSONL)


def _find_row_by_name(rows: list[dict], name: str) -> dict:
    """Find a JSONL row by theorem name."""
    for row in rows:
        label = json.loads(row["label"]) if isinstance(row["label"], str) else row["label"]
        if label["name"] == name:
            return row
    pytest.skip(f"Theorem '{name}' not found in dataset")


@pytest.fixture(scope="session")
def algebra_182_row(valid_rows):
    """Row for mathd_algebra_182 — provable with ``ring``."""
    return _find_row_by_name(valid_rows, "mathd_algebra_182")


@pytest.fixture(scope="session")
def numbertheory_33_row(valid_rows):
    """Row for mathd_numbertheory_33 — provable with ``omega``."""
    return _find_row_by_name(valid_rows, "mathd_numbertheory_33")


@pytest.fixture(scope="session")
def numbertheory_335_row(valid_rows):
    """Row for mathd_numbertheory_335 — provable with ``omega``."""
    return _find_row_by_name(valid_rows, "mathd_numbertheory_335")
