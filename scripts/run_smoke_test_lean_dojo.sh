#!/usr/bin/env bash
set -euo pipefail

# 1) get commit from your submodule so you're still pinned/reproducible
export REPO_URL="${REPO_URL:-https://github.com/leanprover-community/mathlib4}"
export REPO_COMMIT="${REPO_COMMIT:-$(git -C third_party/mathlib4 rev-parse HEAD)}"
export THEOREM_FILE="${THEOREM_FILE:-Mathlib/Data/Nat/Basic.lean}"
export THEOREM_NAME="${THEOREM_NAME:-Nat.succ_injective}"
export TACTIC="${TACTIC:-simp}"
export AUTO_TRACE="${AUTO_TRACE:-1}"

# Optional: keep temporary files in your own dir
mkdir -p /root/tmp
export TMPDIR=/root/tmp

python - <<'PY'
import os
from lean_dojo import Dojo, Theorem
from lean_dojo.data_extraction.lean import LeanGitRepo
from lean_dojo.data_extraction.trace import trace

repo_url = os.environ["REPO_URL"]
repo_commit = os.environ["REPO_COMMIT"]
theorem_file = os.environ["THEOREM_FILE"]
theorem_name = os.environ["THEOREM_NAME"]
tactic = os.environ.get("TACTIC", "simp")
auto_trace = os.environ.get("AUTO_TRACE", "1") == "1"

# IMPORTANT: no from_path(...) here
repo = LeanGitRepo(repo_url, repo_commit)

if auto_trace:
    trace(repo, build_deps=True)

thm = Theorem(repo=repo, file_path=theorem_file, full_name=theorem_name)

with Dojo(thm, timeout=30) as (dojo, state):
    print("=== Initial state ===")
    print(state)
    out = dojo.run_tac(state, tactic)
    print("=== After tactic ===")
    print(out)
PY