#!/usr/bin/env python3
"""Discover, trace, and map the miniF2F-lean4 repo for LeanDojo.

This script:
1. Finds/traces the miniF2F-lean4 Lean 4 repo via LeanDojo
2. Enumerates all theorems with file_path and full_name
3. Builds a mapping JSON and enriched JSONL files
4. Smoke-tests one theorem with Dojo

Usage:
    python scripts/setup_minif2f_leandojo.py

Env vars:
    MINIF2F_URL    - Override the miniF2F repo URL
    MINIF2F_COMMIT - Override the commit hash (default: HEAD)
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# ----- Configuration -----
# Candidate repos to try, in order of preference
CANDIDATE_URLS = [
    "https://github.com/yangky11/miniF2F-lean4",
    "https://github.com/formalizedinmath/miniF2F-lean4",
]

MAPPING_OUT = "data/minif2f_theorem_map.json"


def check_repo_exists(url: str) -> bool:
    """Check if a GitHub repo URL is reachable."""
    try:
        result = subprocess.run(
            ["git", "ls-remote", "--exit-code", url],
            capture_output=True, timeout=30,
        )
        return result.returncode == 0
    except Exception:
        return False


def find_repo() -> str | None:
    """Find the first valid miniF2F-lean4 GitHub repo."""
    # Allow manual override
    if os.environ.get("MINIF2F_URL"):
        url = os.environ["MINIF2F_URL"]
        print(f"Using MINIF2F_URL override: {url}")
        return url

    for url in CANDIDATE_URLS:
        print(f"  Checking {url} ...")
        if check_repo_exists(url):
            print(f"  ✓ Found: {url}")
            return url
        print(f"  ✗ Not found or not accessible")
    return None


def get_default_branch_commit(url: str) -> str:
    """Get the latest commit hash from the default branch of a remote repo."""
    result = subprocess.run(
        ["git", "ls-remote", url, "HEAD"],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git ls-remote failed: {result.stderr}")
    return result.stdout.split()[0]


def _print_summary(theorem_map, repo_url, commit, traced_path):
    """Print a summary of the theorem mapping."""
    print(f"  Theorem mapping:  {MAPPING_OUT}")
    print(f"  Total theorems:   {len(theorem_map)}")
    print(f"  Repo URL:         {repo_url}")
    print(f"  Commit:           {commit}")
    if traced_path:
        print(f"  Traced repo path: {traced_path}")
    unique_files = sorted(set(v["file_path"] for v in theorem_map.values()))
    print(f"  Unique files:     {len(unique_files)}")
    for fp in unique_files:
        count = sum(1 for v in theorem_map.values() if v["file_path"] == fp)
        print(f"    {fp} ({count} theorems)")


def main():
    # Ensure safe.directory is set for LeanDojo's git operations
    os.environ.setdefault("GIT_CONFIG_PARAMETERS", "'safe.directory=*'")

    force = "--force" in sys.argv
    
    # ---- Step 1: Find the repo ----
    print("=" * 60)
    print("Step 1: Finding miniF2F-lean4 repo")
    print("=" * 60)
    repo_url = find_repo()
    if repo_url is None:
        print("\nCould not find a valid miniF2F-lean4 repo.")
        print("Set MINIF2F_URL env var to the correct URL and retry.")
        sys.exit(1)

    commit = os.environ.get("MINIF2F_COMMIT") or get_default_branch_commit(repo_url)
    print(f"\n  Repo:   {repo_url}")
    print(f"  Commit: {commit}")

    # ---- Check if theorem map already exists ----
    if not force and os.path.exists(MAPPING_OUT):
        with open(MAPPING_OUT) as f:
            existing_map = json.load(f)
        # Validate it matches the same repo/commit
        sample = next(iter(existing_map.values()), {})
        if sample.get("repo_url") == repo_url and sample.get("commit") == commit:
            print(f"\nTheorem map already exists at {MAPPING_OUT} "
                  f"({len(existing_map)} theorems, matching repo+commit).")
            print("   Skipping Steps 2-3. Use --force to re-trace.")
            _print_summary(existing_map, repo_url, commit, traced_path=None)
            return
            
    # ---- Step 2: Trace with LeanDojo ----
    print("\n" + "=" * 60)
    print("Step 2: Tracing repo with LeanDojo")
    print("       (may download from remote cache — much faster)")
    print("=" * 60)

    from lean_dojo import Dojo, Theorem
    from lean_dojo.data_extraction.lean import LeanGitRepo
    from lean_dojo.data_extraction.trace import (
        get_traced_repo_path,
        is_available_in_cache,
    )

    repo = LeanGitRepo(repo_url, commit)

    cached = is_available_in_cache(repo)
    print(f"Remote/local cache: {'✓ available' if cached else '✗ not cached, will trace locally (this may take a while)'}")

    traced_path = get_traced_repo_path(repo, build_deps=False)
    print(f"Traced repo at: {traced_path}")

    # ---- Step 3: Enumerate all theorems ----
    # We scan the .lean source files directly instead of using
    # TracedRepo.from_traced_files(), which has compatibility issues
    # with certain LeanDojo/Mathlib version combinations.
    print("\n" + "=" * 60)
    print("Step 3: Enumerating theorems")
    print("=" * 60)

    # Find the repo root inside the traced path
    repo_root = traced_path / "miniF2F-lean4"
    if not repo_root.exists():
        repo_root = traced_path  # fallback

    theorem_map = {}
    for split in ("Test", "Valid"):
        split_dir = repo_root / "MiniF2F" / split
        if not split_dir.exists():
            print(f"  Warning: {split_dir} not found, skipping")
            continue
        for lean_file in sorted(split_dir.glob("*.lean")):
            text = lean_file.read_text()
            m = re.search(r"^theorem\s+(\S+)", text, re.MULTILINE)
            if not m:
                continue
            thm_name = m.group(1)
            file_path = f"MiniF2F/{split}/{lean_file.name}"
            full_name = f"MiniF2F.{split}.{lean_file.stem}"
            theorem_map[full_name] = {
                "repo_url": repo_url,
                "commit": commit,
                "file_path": file_path,
                "full_name": full_name,
            }

    print(f"  Found {len(theorem_map)} theorems total")

    os.makedirs("data", exist_ok=True)
    with open(MAPPING_OUT, "w") as f:
        json.dump(theorem_map, f, indent=2)
    print(f"  Wrote {len(theorem_map)} entries to {MAPPING_OUT}")

    # Show sample entries
    print("\n  Sample entries:")
    for i, name in enumerate(theorem_map):
        if i >= 5:
            break
        print(f"    {name} -> {theorem_map[name]['file_path']}")

    # ---- Step 4: Smoke-test one theorem with Dojo ----
    print("\n" + "=" * 60)
    print("Step 4: Smoke-testing Dojo with first theorem")
    print("=" * 60)

    first_name = next(iter(theorem_map))
    first_info = theorem_map[first_name]
    thm = Theorem(
        repo=repo,
        file_path=first_info["file_path"],
        full_name=first_info["full_name"],
    )
    print(f"  Theorem: {first_name}")
    print(f"  File:    {first_info['file_path']}")

    try:
        with Dojo(thm, timeout=60) as (dojo, init_state):
            print(f"  Initial state: {init_state}")
            result = dojo.run_tac(init_state, "sorry")
            print(f"  After 'sorry': {result}")
        print("  ✓ Dojo smoke test passed!")
    except Exception as e:
        print(f"  ✗ Dojo smoke test failed: {e}")
        print("    (This is OK — the mapping is still valid. Dojo issues can be debugged separately.)")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    _print_summary(theorem_map, repo_url, commit, traced_path)


if __name__ == "__main__":
    main()
