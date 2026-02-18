"""Integration tests for the Lean 4 REPL agent.

These tests spawn real ``lake env lean`` sub-processes and interact with the
patched ``lean_dojo_repl`` tactic, so they require:

* A fully set-up LeanDojo cache (``~/.cache/lean_dojo/...``)
* The ``data/minif2f_theorem_map.json`` file

Run with::

    pytest tests/test_lean_agent_integration.py -v -m integration

Each test is ~10-30 s because it starts a Lean process.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from examples.python.agent_func_lean_minif2f import AgentInstance

# All tests in this file are integration tests
pytestmark = [pytest.mark.integration]

# Default timeout for a single test (seconds)
TIMEOUT = 120

THEOREM_MAP_PATH = Path("data/minif2f_theorem_map.json")
LEAN_DOJO_CACHE = Path(
    os.environ.get("LEAN_DOJO_CACHE", str(Path.home() / ".cache" / "lean_dojo"))
)


def _require_environment():
    """Skip the whole module if the LeanDojo env isn't set up."""
    if not THEOREM_MAP_PATH.exists():
        pytest.skip("Theorem map not found — run setup_minif2f_leandojo.py first")
    if not LEAN_DOJO_CACHE.exists() or not any(LEAN_DOJO_CACHE.iterdir()):
        pytest.skip("LeanDojo cache is empty — import the cache first")


# Skip all tests if env not ready
_require_environment()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_states(row: dict, *, action_text: str = "") -> dict:
    """Build a ``states`` dict suitable for ``reset()`` or ``step()``."""
    return {
        "observation": row["prompt"],
        "label": row["label"],
        "observation_text": row["prompt"],
        "action_text": action_text,
        "sampling_params": None,
    }


def _run(coro):
    """Run an async coroutine synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Test: reset() returns observation and initialises the REPL
# ---------------------------------------------------------------------------


class TestReset:
    """Verify ``reset()`` starts the REPL and returns the prompt."""

    @pytest.fixture(autouse=True)
    def _agent(self):
        self.agent = AgentInstance()
        yield
        self.agent._cleanup()

    def test_reset_returns_observation(self, algebra_182_row):
        states = _make_states(algebra_182_row)
        result = _run(self.agent.reset(states))
        assert "observation" in result
        assert result["observation"] == algebra_182_row["prompt"]

    def test_reset_starts_repl(self, algebra_182_row):
        states = _make_states(algebra_182_row)
        _run(self.agent.reset(states))
        assert self.agent._repl_available is True
        assert self.agent.proc is not None
        assert self.agent.proc.poll() is None  # still running

    def test_reset_sets_initial_sid(self, algebra_182_row):
        states = _make_states(algebra_182_row)
        _run(self.agent.reset(states))
        # The initial REPL response has sid=0
        assert self.agent.current_sid == 0

    def test_reset_unknown_theorem(self):
        """Theorem not in the map → REPL not started, returns observation."""
        fake_label = json.dumps({
            "name": "nonexistent_theorem_xyz",
            "formal_statement": "theorem foo : True := by\n",
            "goal": "⊢ True",
            "header": "import Mathlib\n",
        })
        states = {
            "observation": "Prove something.",
            "label": fake_label,
            "sampling_params": None,
        }
        result = _run(self.agent.reset(states))
        assert result["observation"] == "Prove something."
        assert self.agent._repl_available is False


# ---------------------------------------------------------------------------
# Test: step() — ProofFinished (reward=1)
# ---------------------------------------------------------------------------


class TestProofFinished:
    """Theorems that can be closed in one tactic → reward 1.0, done=True."""

    @pytest.fixture(autouse=True)
    def _agent(self):
        self.agent = AgentInstance()
        yield
        self.agent._cleanup()

    def test_ring_closes_algebra_182(self, algebra_182_row):
        """mathd_algebra_182: 7*(3y+2) = 21y+14, solved by ``ring``."""
        _run(self.agent.reset(_make_states(algebra_182_row)))
        assert self.agent._repl_available

        result = _run(self.agent.step(
            _make_states(algebra_182_row, action_text="ring")
        ))
        assert result["done"] is True
        assert result["rewards"].item() == pytest.approx(1.0)
        assert "[PROOF_COMPLETE]" in result["environment_feedback"]

    def test_omega_closes_numbertheory_33(self, numbertheory_33_row):
        """mathd_numbertheory_33: n<398, n*7%398=1 → n=57, solved by ``omega``."""
        _run(self.agent.reset(_make_states(numbertheory_33_row)))
        assert self.agent._repl_available

        result = _run(self.agent.step(
            _make_states(numbertheory_33_row, action_text="omega")
        ))
        assert result["done"] is True
        assert result["rewards"].item() == pytest.approx(1.0)

    def test_omega_closes_numbertheory_335(self, numbertheory_335_row):
        """mathd_numbertheory_335: n%7=5 → 5n%7=4, solved by ``omega``."""
        _run(self.agent.reset(_make_states(numbertheory_335_row)))
        assert self.agent._repl_available

        result = _run(self.agent.step(
            _make_states(numbertheory_335_row, action_text="omega")
        ))
        assert result["done"] is True
        assert result["rewards"].item() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Test: step() — LeanError (bad tactic)
# ---------------------------------------------------------------------------


class TestLeanError:
    """Sending an invalid tactic should yield reward=0 and a LEAN_ERROR."""

    @pytest.fixture(autouse=True)
    def _agent(self):
        self.agent = AgentInstance()
        yield
        self.agent._cleanup()

    def test_bad_tactic_gives_lean_error(self, algebra_182_row):
        _run(self.agent.reset(_make_states(algebra_182_row)))
        assert self.agent._repl_available

        result = _run(self.agent.step(
            _make_states(algebra_182_row, action_text="this_is_not_a_tactic")
        ))
        assert result["done"] is True
        assert result["rewards"].item() == pytest.approx(0.0)
        assert "[LEAN_ERROR]" in result["environment_feedback"]
        assert result["extra_logs"]["error_type"] == "lean_error"


# ---------------------------------------------------------------------------
# Test: step() — TacticState (multi-step interaction)
# ---------------------------------------------------------------------------


class TestMultiStep:
    """A tactic that doesn't close the goal returns a new goal state."""

    @pytest.fixture(autouse=True)
    def _agent(self):
        self.agent = AgentInstance()
        yield
        self.agent._cleanup()

    def test_skip_returns_new_goal(self, algebra_182_row):
        """``skip`` doesn't change the goal → new TacticState, done=False."""
        _run(self.agent.reset(_make_states(algebra_182_row)))
        assert self.agent._repl_available

        result = _run(self.agent.step(
            _make_states(algebra_182_row, action_text="skip")
        ))
        assert result["done"] is False
        assert result["rewards"].item() == pytest.approx(0.0)
        assert "Current goal:" in result["environment_feedback"]
        assert self.agent.current_sid > 0

    def test_skip_then_ring(self, algebra_182_row):
        """``skip`` then ``ring`` should still close the proof."""
        _run(self.agent.reset(_make_states(algebra_182_row)))

        # Step 1: skip (goal unchanged)
        r1 = _run(self.agent.step(
            _make_states(algebra_182_row, action_text="skip")
        ))
        assert r1["done"] is False

        # Step 2: ring (closes proof)
        r2 = _run(self.agent.step(
            _make_states(algebra_182_row, action_text="ring")
        ))
        assert r2["done"] is True
        assert r2["rewards"].item() == pytest.approx(1.0)
        assert "[PROOF_COMPLETE]" in r2["environment_feedback"]


# ---------------------------------------------------------------------------
# Test: step() — tactic extracted from code block
# ---------------------------------------------------------------------------


class TestCodeBlockExtraction:
    """Tactics wrapped in markdown code blocks should be extracted."""

    @pytest.fixture(autouse=True)
    def _agent(self):
        self.agent = AgentInstance()
        yield
        self.agent._cleanup()

    def test_lean_code_block(self, algebra_182_row):
        _run(self.agent.reset(_make_states(algebra_182_row)))
        llm_output = (
            "I think `ring` should work here.\n\n"
            "```lean\nring\n```\n"
        )
        result = _run(self.agent.step(
            _make_states(algebra_182_row, action_text=llm_output)
        ))
        assert result["done"] is True
        assert result["rewards"].item() == pytest.approx(1.0)

    def test_generic_code_block(self, numbertheory_33_row):
        _run(self.agent.reset(_make_states(numbertheory_33_row)))
        llm_output = "This is solvable by omega:\n```\nomega\n```"
        result = _run(self.agent.step(
            _make_states(numbertheory_33_row, action_text=llm_output)
        ))
        assert result["done"] is True
        assert result["rewards"].item() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Test: step() — REPL not available
# ---------------------------------------------------------------------------


class TestReplUnavailable:
    """Calling step() when the REPL isn't running should fail gracefully."""

    def test_step_without_reset(self):
        agent = AgentInstance()
        # Don't call reset()
        states = {
            "observation_text": "...",
            "action_text": "omega",
            "label": "{}",
            "sampling_params": None,
        }
        result = _run(agent.step(states))
        assert result["done"] is True
        assert result["rewards"].item() == pytest.approx(0.0)
        assert "[ERROR]" in result["environment_feedback"]
        assert result["extra_logs"]["error_type"] == "repl_unavailable"


# ---------------------------------------------------------------------------
# Test: step() — no tactic extracted
# ---------------------------------------------------------------------------


class TestNoTactic:
    """Empty action_text → cannot extract tactic → error."""

    @pytest.fixture(autouse=True)
    def _agent(self):
        self.agent = AgentInstance()
        yield
        self.agent._cleanup()

    def test_empty_action(self, algebra_182_row):
        _run(self.agent.reset(_make_states(algebra_182_row)))
        result = _run(self.agent.step(
            _make_states(algebra_182_row, action_text="")
        ))
        assert result["done"] is True
        assert result["rewards"].item() == pytest.approx(0.0)
        assert result["extra_logs"]["error_type"] == "no_tactic"


# ---------------------------------------------------------------------------
# Test: cleanup is idempotent
# ---------------------------------------------------------------------------


class TestCleanup:
    """Cleanup should not raise even when called multiple times."""

    def test_double_cleanup(self, algebra_182_row):
        agent = AgentInstance()
        _run(agent.reset(_make_states(algebra_182_row)))
        agent._cleanup()
        agent._cleanup()  # should not raise

    def test_cleanup_removes_tmp_file(self, algebra_182_row):
        agent = AgentInstance()
        _run(agent.reset(_make_states(algebra_182_row)))
        tmp = agent.tmp_file
        assert tmp is not None and tmp.exists()
        agent._cleanup()
        assert not tmp.exists()
        assert agent.tmp_file is None


# ---------------------------------------------------------------------------
# Test: max steps enforcement
# ---------------------------------------------------------------------------


class TestMaxSteps:
    """When step_idx reaches max_steps, the episode ends."""

    @pytest.fixture(autouse=True)
    def _agent(self):
        self.agent = AgentInstance()
        self.agent.max_steps = 2  # Low limit for testing
        yield
        self.agent._cleanup()

    def test_max_steps_reached(self, algebra_182_row):
        _run(self.agent.reset(_make_states(algebra_182_row)))

        # Step 1: skip
        r1 = _run(self.agent.step(
            _make_states(algebra_182_row, action_text="skip")
        ))
        assert r1["done"] is False

        # Step 2: skip — this should hit max_steps
        r2 = _run(self.agent.step(
            _make_states(algebra_182_row, action_text="skip")
        ))
        assert r2["done"] is True
        assert r2["rewards"].item() == pytest.approx(0.0)
        assert "[MAX_STEPS]" in r2["environment_feedback"]
        assert r2["extra_logs"]["error_type"] == "max_steps"


# ---------------------------------------------------------------------------
# Test: return dict structure
# ---------------------------------------------------------------------------


class TestReturnStructure:
    """Verify all required keys are present in step() output."""

    @pytest.fixture(autouse=True)
    def _agent(self):
        self.agent = AgentInstance()
        yield
        self.agent._cleanup()

    def test_keys_present(self, algebra_182_row):
        _run(self.agent.reset(_make_states(algebra_182_row)))
        result = _run(self.agent.step(
            _make_states(algebra_182_row, action_text="ring")
        ))

        expected_keys = {
            "rewards", "scores", "environment_feedback",
            "done", "sampling_params", "extra_logs",
        }
        assert expected_keys.issubset(result.keys())

    def test_rewards_is_tensor(self, algebra_182_row):
        _run(self.agent.reset(_make_states(algebra_182_row)))
        result = _run(self.agent.step(
            _make_states(algebra_182_row, action_text="ring")
        ))
        assert isinstance(result["rewards"], torch.Tensor)
        assert result["rewards"].dtype == torch.float32

    def test_scores_is_tensor(self, algebra_182_row):
        _run(self.agent.reset(_make_states(algebra_182_row)))
        result = _run(self.agent.step(
            _make_states(algebra_182_row, action_text="ring")
        ))
        assert isinstance(result["scores"], torch.Tensor)
        assert result["scores"].dtype == torch.float32

    def test_extra_logs_is_dict(self, algebra_182_row):
        _run(self.agent.reset(_make_states(algebra_182_row)))
        result = _run(self.agent.step(
            _make_states(algebra_182_row, action_text="ring")
        ))
        assert isinstance(result["extra_logs"], dict)
        assert "error_type" in result["extra_logs"]


# ---------------------------------------------------------------------------
# Test: AgentExecutor class
# ---------------------------------------------------------------------------


class TestAgentExecutor:
    """Basic sanity check on AgentExecutor."""

    def test_executor_creates_instances(self):
        from examples.python.agent_func_lean_minif2f import AgentExecutor
        executor = AgentExecutor()
        assert executor.agent_instance_cls is AgentInstance
