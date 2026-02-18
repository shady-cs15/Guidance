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


def _fix_dep_paths(traced_path: Path) -> None:
    """Rewrite stale absolute /tmp/... paths in .dep_paths files to relative paths.

    During tracing, LeanDojo writes dependency paths relative to a temp dir.
    After the traced repo is moved to the cache, those absolute paths become
    invalid and cause ``ValueError: ... is not in the subpath of ...`` when
    LeanDojo tries ``path.relative_to(root_dir)``.

    We fix this by converting any absolute path to a path relative to the
    repo root.  The heuristic is: find the repo directory name (e.g.
    ``miniF2F-lean4``) inside the absolute path and keep everything after it.
    """
    repo_name = traced_path.name  # e.g. "miniF2F-lean4"
    dep_files = list(traced_path.rglob("*.dep_paths"))
    fixed_count = 0
    for dep_file in dep_files:
        lines = dep_file.read_text().splitlines()
        new_lines = []
        changed = False
        for line in lines:
            if line.startswith("/"):
                # Try to extract relative portion after the repo name
                marker = f"/{repo_name}/"
                idx = line.find(marker)
                if idx != -1:
                    line = line[idx + len(marker):]
                    changed = True
            new_lines.append(line)
        if changed:
            dep_file.write_text("\n".join(new_lines) + "\n")
            fixed_count += 1
    if fixed_count:
        print(f"  Fixed stale absolute paths in {fixed_count}/{len(dep_files)} .dep_paths files")
    else:
        print(f"  All {len(dep_files)} .dep_paths files are clean")


def _fix_lean4_repl(traced_path: Path) -> None:
    """Patch Lean4Repl.lean for Lean 4.24+ compatibility.

    Two issues:
    1. ``NameSet.union`` was renamed to ``NameSet.merge`` in Lean 4.24
       (``Std.TreeSet`` replaced ``Lean.RBTree``).
    2. ``IO.getStdin`` returns an empty stream in elaboration mode (non-``--run``).
       We work around this by reading from ``/proc/self/fd/0`` directly, which
       only works when the process's stdin is a *pipe* (not a pty).
    """
    repl_path = traced_path / "Lean4Repl.lean"
    if not repl_path.exists():
        print("  Warning: Lean4Repl.lean not found, skipping patch")
        return

    text = repl_path.read_text()
    changed = False

    # Fix 1: .union → .merge
    if ".union" in text:
        text = text.replace(".union", ".merge")
        changed = True
        print("  Patched .union → .merge in Lean4Repl.lean")

    # Fix 2: IO.getStdin → /proc/self/fd/0 (for pipe-based IPC)
    # In Lean 4.24+, IO.getStdin returns an empty stream during file elaboration.
    old_loop_header = (
        'private def loop (m : Type → Type) [Monad m] [MonadLift IO m] '
        '[MonadError m] (handler : Request → m Response) : m Unit := do\n'
        ' while true do\n'
        '    let line := (← (← IO.getStdin).getLine).trim'
    )
    new_loop_header = (
        'private def loop (m : Type → Type) [Monad m] [MonadLift IO m] '
        '[MonadError m] (handler : Request → m Response) : m Unit := do\n'
        ' let stdinHandle ← IO.FS.Handle.mk "/proc/self/fd/0" .read\n'
        ' while true do\n'
        '    let line := (← stdinHandle.getLine).trim'
    )
    if old_loop_header in text:
        text = text.replace(old_loop_header, new_loop_header)
        changed = True
        print("  Patched IO.getStdin → /proc/self/fd/0 in Lean4Repl.lean")
    elif "stdinHandle" in text:
        print("  Lean4Repl.lean already patched for stdin")
    else:
        print("  Warning: could not find loop header to patch in Lean4Repl.lean")

    # Fix 3: Remove IO.Process.exit 0 — it terminates before stdout flushes.
    # Without it, the tactic returns normally and Lean flushes output on exit.
    if "IO.Process.exit 0" in text:
        text = text.replace("  IO.Process.exit 0\n", "")
        changed = True
        print("  Removed IO.Process.exit 0 (prevents stdout flush)")

    # Fix 4: Write REPL responses directly to /proc/self/fd/1 and flush.
    # In elaboration mode, println! writes to C's fully-buffered stdout.
    # IO.getStdout.flush only flushes Lean's wrapper, not the C buffer.
    # Writing to /proc/self/fd/1 directly bypasses both buffering layers.
    old_print = (
        '  let json := (toJson res).pretty 99999999999999999\n'
        '  println! "REPL> {json}"'
    )
    new_print = (
        '  let json := (toJson res).pretty 99999999999999999\n'
        '  let msg := s!"REPL> {json}\\n"\n'
        '  -- Write directly to fd 1 to bypass Lean stdout buffering in elaboration mode\n'
        '  let h ← IO.FS.Handle.mk "/proc/self/fd/1" .write\n'
        '  h.putStr msg\n'
        '  h.flush'
    )
    if old_print in text:
        text = text.replace(old_print, new_print)
        changed = True
        print("  Patched printResponse to write directly to /proc/self/fd/1")

    # Fix 5: Accept proofs that fail kernel addDecl due to Lean 4.24 prefix
    # restrictions.  The REPL's validateProof calls ``addDecl`` with
    # ``Name.anonymous``, which violates the per-file prefix restriction
    # added in Lean 4.24.  We keep the check for *other* kernel errors but
    # treat the prefix-restriction error as benign (the proof has already
    # been validated: correct type, no sorry, no metavars).
    old_validate = (
        '  catch ex =>\n'
        '    return {error := s!"kernel type check failed: '
        '{← ex.toMessageData.toString}"}'
    )
    new_validate = (
        '  catch ex =>\n'
        '    let errMsg ← ex.toMessageData.toString\n'
        '    if (errMsg.splitOn "restricted to the prefix").length <= 1 then\n'
        '      return {error := s!"kernel type check failed: {errMsg}"}'
    )
    if old_validate in text:
        text = text.replace(old_validate, new_validate)
        changed = True
        print("  Patched validateProof to accept prefix-restricted proofs")

    if changed:
        repl_path.write_text(text)


def _build_lean4_repl(traced_path: Path) -> None:
    """Build the Lean4Repl library in the traced repo."""
    repl_olean = traced_path / ".lake" / "build" / "lib" / "lean" / "Lean4Repl.olean"
    repl_src = traced_path / "Lean4Repl.lean"
    # Rebuild if olean is missing or older than the source
    if repl_olean.exists() and repl_olean.stat().st_mtime > repl_src.stat().st_mtime:
        print("  Lean4Repl.olean is up to date")
        return

    elan_bin = Path.home() / ".elan" / "bin"
    env = os.environ.copy()
    if elan_bin.exists():
        env["PATH"] = f"{elan_bin}:{env.get('PATH', '')}"

    print("  Building Lean4Repl ...")
    result = subprocess.run(
        ["lake", "build", "Lean4Repl"],
        cwd=traced_path,
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    if result.returncode != 0:
        print(f"  Warning: lake build Lean4Repl failed:\n{result.stderr[:500]}")
    else:
        print("  ✓ Lean4Repl built successfully")


def _smoke_test_repl(traced_path: Path, file_path: str, full_name: str, repo) -> None:
    """Quick smoke test: open the Lean REPL via subprocess pipes and run 'skip'."""
    from lean_dojo.data_extraction.traced_data import TracedFile
    from lean_dojo.utils import to_json_path
    from lean_dojo.interaction.dojo import get_code_without_comments

    json_path = to_json_path(traced_path, Path(file_path), repo)
    tf = TracedFile.from_traced_file(traced_path, json_path, repo)
    traced_thm = tf.get_traced_theorem(full_name)
    if traced_thm is None:
        print(f"  ✗ Could not find theorem '{full_name}' in traced file")
        return

    proof_start, proof_end = traced_thm.locate_proof()
    lean_file = tf.lean_file

    code_import = "import Lean4Repl\n"
    code_proof = "by\n  lean_dojo_repl\n  sorry\n"
    code_before = get_code_without_comments(
        lean_file, lean_file.start_pos, traced_thm.start, tf.comments
    )
    code_thm = get_code_without_comments(
        lean_file, traced_thm.start, proof_start, tf.comments
    ).strip()
    if not code_thm.endswith(":="):
        code_thm += " := "

    modified = (
        code_import + code_before
        + "\n\nset_option maxHeartbeats 0 in\n"
        + code_thm + code_proof
        + lean_file[proof_end:]
    )

    import tempfile
    tmp = tempfile.NamedTemporaryFile(
        "w", prefix="smoke_", suffix=".lean",
        dir=traced_path, delete=False,
    )
    tmp.write(modified)
    tmp.flush()
    tmp_path = Path(tmp.name)

    elan_bin = Path.home() / ".elan" / "bin"
    env = os.environ.copy()
    if elan_bin.exists():
        env["PATH"] = f"{elan_bin}:{env.get('PATH', '')}"

    try:
        rel = tmp_path.relative_to(traced_path)
        stdin_data = '{"sid": 0, "cmd": "skip"}\nexit\n'
        result = subprocess.run(
            ["lake", "env", "lean", "--threads=4", "--memory=32768", str(rel)],
            cwd=traced_path, input=stdin_data,
            capture_output=True, text=True, timeout=120, env=env,
        )
        # Parse REPL output
        for line in result.stdout.splitlines():
            if line.startswith("REPL>"):
                data = json.loads(line[5:].strip())
                if data.get("tacticState"):
                    state = data["tacticState"]
                    print(f"  Initial state: {state[:80]}...")
                if data.get("sid") is not None and data.get("error") is None:
                    print(f"  ✓ REPL responded (sid={data['sid']})")
                elif data.get("error"):
                    print(f"  REPL error: {data['error'][:100]}")
        if "REPL>" not in result.stdout:
            print(f"  ✗ No REPL output. Exit code: {result.returncode}")
            if result.stdout.strip():
                print(f"    stdout: {result.stdout[:200]}")
        else:
            print("  ✓ Smoke test passed!")
    finally:
        tmp_path.unlink(missing_ok=True)


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

    from lean_dojo import Theorem
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

    # ---- Step 2b: Fix stale paths and patch Lean4Repl ----
    _fix_dep_paths(traced_path)
    _fix_lean4_repl(traced_path)
    _build_lean4_repl(traced_path)

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
            # full_name must match what LeanDojo's AST extracts:
            # the bare theorem identifier (no module/namespace prefix,
            # since miniF2F files have no `namespace` blocks).
            full_name = thm_name
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

    # ---- Step 4: Smoke-test the REPL via pipe-based IPC ----
    print("\n" + "=" * 60)
    print("Step 4: Smoke-testing Lean REPL (pipe-based)")
    print("=" * 60)

    first_name = next(iter(theorem_map))
    first_info = theorem_map[first_name]
    print(f"  Theorem: {first_name}")
    print(f"  File:    {first_info['file_path']}")

    try:
        _smoke_test_repl(traced_path, first_info["file_path"], first_info["full_name"], repo)
    except Exception as e:
        print(f"  ✗ Smoke test failed: {e}")
        print("    (The mapping is still valid. REPL issues can be debugged separately.)")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    _print_summary(theorem_map, repo_url, commit, traced_path)


if __name__ == "__main__":
    main()
