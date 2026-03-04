"""External-model guidance for multi-turn RL exploration.

Calls an OpenRouter-compatible chat-completions endpoint to generate
first-person reflective guidance given the trajectory so far and the
latest environment feedback.  The guidance text is injected into the
agent context but masked out during training (same as env feedback).
"""

import os

import aiohttp

from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)

_DEFAULT_PROMPT_TEMPLATE = """\
You are reflecting on your proof attempt in Lean 4.

Here is your trajectory so far:
{trajectory_text}

The latest feedback from the environment:
{latest_feedback}

Reflect in first person on what went wrong and what you should try next.
Be concise and specific. Write as "I should..." / "I made the mistake of..."

IMPORTANT: Do NOT include any code blocks or tactic suggestions in your response. \
Only provide strategic, conceptual advice about the proof approach. \
Do NOT use triple backticks or write any Lean code.\
"""


class GuidanceClient:
    """Async client that fetches reflective guidance from an external LLM."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int | None = None,
        prompt_template: str | None = None,
        api_base: str | None = None,
    ):
        self.model = model or os.environ.get("GUIDANCE_MODEL", "qwen/qwen3.5-27b")
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.max_tokens = max_tokens or int(os.environ.get("GUIDANCE_MAX_TOKENS", "512"))
        self.prompt_template = prompt_template or os.environ.get(
            "GUIDANCE_PROMPT_TEMPLATE", _DEFAULT_PROMPT_TEMPLATE
        )
        self.api_base = api_base or os.environ.get(
            "GUIDANCE_API_BASE", "https://openrouter.ai/api/v1"
        )

        if not self.api_key:
            logger.warning("OPENROUTER_API_KEY not set — guidance calls will be skipped")
        if not self.model:
            logger.warning("GUIDANCE_MODEL not set — guidance calls will be skipped")
        else:
            logger.info("Guidance model: %s (reasoning excluded)", self.model)

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.model)

    async def get_guidance(self, trajectory_text: str, latest_feedback: str) -> str:
        """Return guidance text wrapped in delimiters, or empty string on failure."""
        if not self.enabled:
            return ""

        prompt = self.prompt_template.format(
            trajectory_text=trajectory_text,
            latest_feedback=latest_feedback,
        )

        try:
            guidance_text = await self._call_api(prompt)
        except Exception as exc:
            logger.warning("Guidance API call failed (skipping): %s", exc)
            return ""

        if not guidance_text:
            return ""

        return f"\n\n[GUIDANCE]\n{guidance_text}\n[/GUIDANCE]\n"

    async def _call_api(self, prompt: str, retries: int = 3) -> str:
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "include_reasoning": False,
        }
        timeout = aiohttp.ClientTimeout(total=120)

        for attempt in range(1, retries + 1):
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(url, json=payload, headers=headers) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                        return data["choices"][0]["message"]["content"]
            except Exception as exc:
                logger.warning("Guidance API attempt %d/%d failed: %s", attempt, retries, exc)
                if attempt == retries:
                    raise
                import asyncio

                await asyncio.sleep(1)
        return ""
