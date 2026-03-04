"""Guided multi-turn Lean 4 theorem-proving agent for miniF2F.

Identical to ``agent_func_lean_minif2f.py`` but uses
``GuidedMultiTurnAgentExecutor`` to inject external-model guidance
(via OpenRouter) after each environment step.  Guidance tokens are
masked out during training, providing the model with reflective hints
without contributing to the PPO loss.

Extra environment variables (on top of the base Lean agent):
    OPENROUTER_API_KEY : str   – API key for OpenRouter.
    GUIDANCE_MODEL     : str   – Model identifier on OpenRouter.
    GUIDANCE_MAX_TOKENS: int   – Max tokens for guidance response (default 512).
"""

import importlib.util
from pathlib import Path

from openrlhf.utils.agent import GuidedMultiTurnAgentExecutor

_SIBLING = Path(__file__).with_name("agent_func_lean_minif2f.py")
_spec = importlib.util.spec_from_file_location("agent_func_lean_minif2f", str(_SIBLING))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
AgentInstance = _mod.AgentInstance


class AgentExecutor(GuidedMultiTurnAgentExecutor):
    def __init__(self):
        super().__init__(AgentInstance)
