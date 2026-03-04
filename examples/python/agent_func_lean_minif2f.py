"""Multi-turn Lean 4 theorem-proving agent for miniF2F.

This agent environment wraps the patched LeanDojo REPL (``lean_dojo_repl``
tactic) behind the OpenRLHF ``AgentInstanceBase`` interface so that an LLM
can interactively prove theorems.

The REPL is patched (by ``scripts/setup_minif2f_leandojo.py``) so that:
* stdin is read from ``/proc/self/fd/0`` (works when stdin is a pipe);
* responses are written directly to ``/proc/self/fd/1`` and flushed,
  bypassing Lean's internal stdout buffering in elaboration mode.

Environment variables
---------------------
MINIF2F_THEOREM_MAP : str
    Path to ``data/minif2f_theorem_map.json``.
LEAN_DOJO_CACHE : str
    Root of the LeanDojo cache dir.  Default: ``~/.cache/lean_dojo``.
LEAN_REPL_TIMEOUT : int
    Seconds to wait for any single REPL response.  Default: 300.
LEAN_THREADS : int
    ``--threads`` passed to ``lake env lean``.  Default: 4.
LEAN_MEMORY : int
    ``--memory`` (MiB) passed to ``lake env lean``.  Default: 32768.
LEAN_MAX_STEPS : int
    Maximum number of tactic steps before giving up.  Default: 100.
"""

import json
import logging
import os
import queue
import re
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from openrlhf.utils.agent import AgentInstanceBase, MultiTurnAgentExecutor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
THEOREM_MAP_PATH = os.environ.get(
    "MINIF2F_THEOREM_MAP", "data/minif2f_theorem_map.json"
)
LEAN_DOJO_CACHE = os.environ.get(
    "LEAN_DOJO_CACHE", str(Path.home() / ".cache" / "lean_dojo")
)
REPL_TIMEOUT = int(os.environ.get("LEAN_REPL_TIMEOUT", "300"))
LEAN_THREADS = int(os.environ.get("LEAN_THREADS", "4"))
LEAN_MEMORY = int(os.environ.get("LEAN_MEMORY", "32768"))
MAX_STEPS = int(os.environ.get("LEAN_MAX_STEPS", "100"))

# ---------------------------------------------------------------------------
# Singleton theorem map loader
# ---------------------------------------------------------------------------
_theorem_map: Optional[dict] = None


def _get_theorem_map() -> dict:
    global _theorem_map
    if _theorem_map is None:
        with open(THEOREM_MAP_PATH) as f:
            _theorem_map = json.load(f)
    return _theorem_map


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_traced_repo_path(repo_url: str, commit: str) -> Path:
    """Locate the traced repo directory inside the LeanDojo cache.

    LeanDojo stores traced repos at
    ``<cache>/<owner>-<repo>-<commit>/<repo>/``.
    """
    url_slug = repo_url.replace("https://github.com/", "").replace("/", "-")
    cache_dir = Path(LEAN_DOJO_CACHE)
    target = f"{url_slug}-{commit}"

    for entry in cache_dir.iterdir():
        if entry.is_dir() and entry.name == target:
            for sub in entry.iterdir():
                if sub.is_dir() and not sub.name.startswith("."):
                    return sub
    raise FileNotFoundError(
        f"Traced repo not found in {cache_dir} for {target}"
    )


_MULTI_LINE_TACTIC_STARTS = frozenset({
    "calc", "match", "suffices", "show", "by_cases", "rcases", "obtain",
})


def _extract_tactic(action_text: str) -> str:
    """Extract a single tactic from an LLM response.

    Strategy:
    1. Find the last ``````lean ... `````` (or generic ````` ... `````) block,
       skipping blocks that look like full program fragments rather than
       tactics (starting with ``import``, ``theorem``, ``#``, etc.).
    2. If the block contains multiple independent tactics on separate lines,
       return only the first one — the REPL expects one tactic at a time.
       Multi-line tactics (``calc``, ``match``, ``suffices``, …) are kept
       intact.
    3. If no valid code block is found, return "" to trigger [PARSE_ERROR].
       No raw-text fallback — the model is expected to use code blocks.
    """
    NON_TACTIC_PREFIXES = ("import ", "theorem ", "lemma ", "#", "open ", "set_option ", "##")

    # Collect all code blocks (lean-tagged first, then generic).
    blocks = list(re.finditer(r"```lean\s*\n?(.*?)```", action_text, re.DOTALL))
    if not blocks:
        blocks = list(re.finditer(r"```\s*\n?(.*?)```", action_text, re.DOTALL))

    # Walk backwards through blocks and pick the last one that looks like a tactic.
    for m in reversed(blocks):
        content = m.group(1).strip()
        if not content:
            continue
        if content.startswith(NON_TACTIC_PREFIXES):
            continue
        return _first_tactic(content)

    # No usable code block — return empty to trigger [PARSE_ERROR].
    return ""


def _first_tactic(block: str) -> str:
    """Return only the first independent tactic from a multi-line block.

    Known multi-line tactic starters (``calc``, ``match``, …) are returned
    in full because they span multiple lines by design.
    """
    lines = [l for l in block.splitlines() if l.strip()]
    if not lines:
        return ""

    first_word = lines[0].split()[0].rstrip(":") if lines[0].split() else ""
    if first_word in _MULTI_LINE_TACTIC_STARTS:
        return block

    return lines[0].strip()


def _build_lean_file(header: str, formal_statement: str) -> str:
    """Construct the ``.lean`` source that invokes the REPL tactic."""
    lines = ["import Lean4Repl"]

    # Append the original header (imports, opens, set_option, etc.)
    lines.append(header.rstrip())

    # Scope an unlimited heartbeat budget to just this theorem.
    lines.append("")
    lines.append("set_option maxHeartbeats 0 in")

    # The formal_statement should end with `:= by\n`
    stmt = formal_statement.rstrip()
    lines.append(stmt)

    # Insert the REPL tactic followed by sorry (as a fallback)
    lines.append("  lean_dojo_repl")
    lines.append("  sorry")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class AgentInstance(AgentInstanceBase):
    """A Lean 4 REPL environment for interactive theorem proving.

    Each instance manages one ``lean`` subprocess that runs the
    ``lean_dojo_repl`` tactic.  Communication is via stdin/stdout pipes.
    A background thread reads stdout lines into a queue to avoid
    dead-locking on buffered I/O.
    """

    def __init__(self, *args, **kwargs):
        self.proc: Optional[subprocess.Popen] = None
        self.tmp_file: Optional[Path] = None
        self.current_sid: int = 0
        self.traced_path: Optional[Path] = None
        self.step_idx: int = 0
        self.max_steps: int = MAX_STEPS
        self._repl_available: bool = False
        self._line_queue: Optional[queue.Queue] = None
        self._reader_thread: Optional[threading.Thread] = None

    # ---- lifecycle ---------------------------------------------------------

    async def reset(self, states: dict, **kwargs) -> dict:
        """Start a Lean REPL for the theorem; return initial observation."""
        prompt = states["observation"]
        label_str = states["label"]
        label = json.loads(label_str) if isinstance(label_str, str) else label_str

        name = label["name"]
        formal_statement = label["formal_statement"]
        header = label.get(
            "header",
            (
                "import Mathlib\nimport Aesop\n\n"
                "set_option maxHeartbeats 0\n\n"
                "open BigOperators Real Nat Topology Rat\n\n"
            ),
        )

        # Look up traced-repo location
        theorem_map = _get_theorem_map()
        thm_info = theorem_map.get(name)
        if thm_info is None:
            logger.warning("Theorem '%s' not in theorem map", name)
            return {"observation": prompt}

        try:
            self.traced_path = _find_traced_repo_path(
                thm_info["repo_url"], thm_info["commit"]
            )
        except FileNotFoundError as exc:
            logger.warning("Traced repo not found: %s", exc)
            return {"observation": prompt}

        # Write a temp .lean file for the REPL
        lean_src = _build_lean_file(header, formal_statement)
        fd, tmp_path = tempfile.mkstemp(
            prefix=f"repl_{name}_", suffix=".lean", dir=str(self.traced_path)
        )
        self.tmp_file = Path(tmp_path)
        with os.fdopen(fd, "w") as fh:
            fh.write(lean_src)

        # Start the Lean REPL process
        elan_bin = Path.home() / ".elan" / "bin"
        env = os.environ.copy()
        if elan_bin.exists():
            env["PATH"] = f"{elan_bin}:{env.get('PATH', '')}"

        rel = self.tmp_file.relative_to(self.traced_path)
        self.proc = subprocess.Popen(
            [
                "lake", "env", "lean",
                f"--threads={LEAN_THREADS}",
                f"--memory={LEAN_MEMORY}",
                str(rel),
            ],
            cwd=self.traced_path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
        )

        # Start a background reader thread (avoids deadlock with buffered I/O)
        self._line_queue = queue.Queue()
        self._reader_thread = threading.Thread(
            target=self._stdout_reader, daemon=True
        )
        self._reader_thread.start()

        # Wait for the initial proof state
        try:
            initial = self._read_repl_response(timeout=REPL_TIMEOUT)
            self._repl_available = True
            self.current_sid = initial.get("sid", 0)
            logger.info(
                "REPL ready for %s (sid=%d, goals=%s…)",
                name,
                self.current_sid,
                (initial.get("tacticState") or "")[:60],
            )
        except Exception as exc:
            logger.warning("REPL init failed for %s: %s", name, exc)
            self._cleanup()

        return {"observation": prompt}

    async def step(self, states: dict, **kwargs) -> Dict[str, Any]:
        """Send one tactic to the REPL and interpret the result."""
        action_text = states["action_text"]
        self.step_idx += 1

        # Guard: REPL not available
        if (
            not self._repl_available
            or self.proc is None
            or self.proc.poll() is not None
        ):
            return self._result(
                reward=0.0,
                done=True,
                feedback="\n\n[ERROR] Lean REPL is not available.\n",
                states=states,
                extra={"error_type": "repl_unavailable", "steps": self.step_idx},
            )

        # Extract tactic from LLM output
        tactic = _extract_tactic(action_text)
        if not tactic:
            if self.step_idx >= self.max_steps:
                self._cleanup()
                return self._result(
                    reward=0.0,
                    done=True,
                    feedback="\n\n[ERROR] Could not extract a tactic (max steps reached).\n",
                    states=states,
                    extra={"error_type": "no_tactic", "steps": self.step_idx},
                )
            return self._result(
                reward=0.0,
                done=False,
                feedback=(
                    "\n\n[PARSE_ERROR] Could not extract a tactic from your response.\n"
                    "Please provide exactly one tactic inside a ```lean code block.\n"
                ),
                states=states,
                extra={"error_type": "no_tactic", "steps": self.step_idx},
            )

        # Send tactic to the REPL
        try:
            req = json.dumps({"sid": self.current_sid, "cmd": tactic})
            self.proc.stdin.write(req.encode() + b"\n")
            self.proc.stdin.flush()
            resp = self._read_repl_response(timeout=REPL_TIMEOUT)
        except Exception as exc:
            logger.warning("REPL communication error: %s", exc)
            self._cleanup()
            return self._result(
                reward=0.0,
                done=True,
                feedback=f"\n\n[ERROR] REPL communication failed: {str(exc)[:200]}\n",
                states=states,
                extra={"error_type": "repl_comm_error", "steps": self.step_idx},
            )

        error = resp.get("error")
        tactic_state = resp.get("tacticState")
        new_sid = resp.get("sid")

        # ---- LeanError (recoverable — proof state unchanged, let model retry) -----
        if error:
            if self.step_idx >= self.max_steps:
                self._cleanup()
                return self._result(
                    reward=0.0,
                    done=True,
                    feedback=f"\n\n[LEAN_ERROR] {error}\n[MAX_STEPS] Reached maximum {self.max_steps} steps.\n",
                    states=states,
                    extra={"error_type": "lean_error", "lean_error": error[:500], "steps": self.step_idx},
                )
            return self._result(
                reward=0.0,
                done=False,
                feedback=(
                    f"\n\n[LEAN_ERROR] {error}\n\n"
                    "The tactic failed. The proof state is unchanged.\n"
                    "Try a different tactic.\n"
                ),
                states=states,
                extra={"error_type": "lean_error", "lean_error": error[:500], "steps": self.step_idx},
            )

        # ---- ProofFinished -----
        if tactic_state == "no goals":
            self._cleanup()
            return self._result(
                reward=1.0,
                done=True,
                feedback="\n\n[PROOF_COMPLETE] Proof finished successfully!\n",
                states=states,
                extra={"error_type": "none", "steps": self.step_idx},
            )

        # ---- ProofGivenUp / unexpected -----
        if tactic_state is None:
            self._cleanup()
            return self._result(
                reward=0.0,
                done=True,
                feedback="\n\n[PROOF_GIVEN_UP] No tactic state returned.\n",
                states=states,
                extra={"error_type": "proof_given_up", "steps": self.step_idx},
            )

        # ---- TacticState (continue proving) -----
        self.current_sid = new_sid if new_sid is not None else self.current_sid

        # Enforce step limit
        if self.step_idx >= self.max_steps:
            self._cleanup()
            return self._result(
                reward=0.0,
                done=True,
                feedback=(
                    f"\n\n[MAX_STEPS] Reached maximum {self.max_steps} steps.\n"
                    f"Last goal:\n{tactic_state}\n"
                ),
                states=states,
                extra={"error_type": "max_steps", "steps": self.step_idx},
            )

        feedback = (
            f"\n\nCurrent goal:\n```\n{tactic_state}\n```\n\n"
            "Provide the next tactic."
        )
        return self._result(
            reward=0.0,
            done=False,
            feedback=feedback,
            states=states,
            extra={"error_type": "none", "steps": self.step_idx},
        )

    # ---- internal helpers --------------------------------------------------

    def _stdout_reader(self) -> None:
        """Background thread: read lines from proc.stdout into a queue."""
        try:
            for raw_line in iter(self.proc.stdout.readline, b""):
                self._line_queue.put(
                    raw_line.decode("utf-8", errors="replace")
                )
        except Exception:
            pass
        self._line_queue.put(None)  # sentinel

    def _read_repl_response(self, timeout: int = 300) -> dict:
        """Consume lines from the reader queue until ``REPL> {…}``."""
        import time

        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            try:
                line = self._line_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if line is None:
                raise RuntimeError("REPL process exited unexpectedly")
            if "REPL>" in line:
                m = re.search(r"REPL>\s*(\{.*\})", line)
                if m:
                    return json.loads(m.group(1))

        raise TimeoutError(
            f"REPL did not respond within {timeout}s"
        )

    @staticmethod
    def _result(
        reward: float,
        done: bool,
        feedback: str,
        states: dict,
        extra: Optional[dict] = None,
    ) -> Dict[str, Any]:
        return {
            "rewards": torch.tensor(reward, dtype=torch.float32),
            "scores": torch.tensor(reward, dtype=torch.float32),
            "environment_feedback": feedback,
            "done": done,
            "sampling_params": states.get("sampling_params"),
            "extra_logs": extra or {},
        }

    def _cleanup(self) -> None:
        """Tear down the REPL subprocess and delete the temp file."""
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.stdin.write(b"exit\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
                try:
                    self.proc.wait(timeout=5)
                except Exception:
                    pass
        self.proc = None
        self._repl_available = False

        if self.tmp_file is not None:
            self.tmp_file.unlink(missing_ok=True)
            self.tmp_file = None

    def __del__(self):
        self._cleanup()


# ---------------------------------------------------------------------------
# Executor (entry-point for OpenRLHF)
# ---------------------------------------------------------------------------


class AgentExecutor(MultiTurnAgentExecutor):
    def __init__(self):
        super().__init__(AgentInstance)
