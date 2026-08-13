"""Thin async wrapper around LiteLLM with provider-key handling and retries."""

import logging
import os

import litellm

from .config import get_settings

logger = logging.getLogger(__name__)

# provider prefix -> (settings attr, standard env var)
_PROVIDER_KEY_ENV = {
    "openrouter": "OPENROUTER_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
}


async def complete(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int | None = None,
    json_mode: bool = False,
) -> tuple[str, dict | None]:
    """Run a chat completion and return (text, usage_metadata).

    usage_metadata is a dict with total_tokens and cost when available,
    or None when the provider does not expose usage (e.g. cached responses).
    Raises a clear error when the requested provider has no API key configured.
    """
    settings = get_settings()
    model = model or settings.model
    provider = model.split("/", 1)[0]

    env_key = _PROVIDER_KEY_ENV.get(provider)
    configured = os.environ.get(env_key) if env_key else True
    if not configured:
        raise RuntimeError(
            f"No API key configured for provider '{provider}'. "
            f"Set {env_key} in .env (see .env.example)."
        )

    kwargs: dict = {"temperature": temperature}
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        response = await litellm.acompletion(model=model, messages=messages, **kwargs)
        text = response.choices[0].message.content or ""
        usage = _extract_usage(response.usage)
        return text, usage
    except Exception as exc:  # litellm raises a grab-bag of exceptions
        logger.error("LLM call failed (model=%s): %s", model, exc)
        raise


def _extract_usage(usage) -> dict | None:
    """Extract total_tokens and cost from a LiteLLM usage object.

    Returns None when usage is missing or does not expose the fields we need
    (e.g. cached responses may omit usage entirely).
    """
    if usage is None:
        return None
    total = getattr(usage, "total_tokens", None)
    cost = getattr(usage, "cost", None)
    if total is None and cost is None:
        return None
    return {
        "total_tokens": total if total is not None else 0,
        "cost": cost if cost is not None else 0.0,
    }


async def complete_json(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.0,
) -> tuple[dict, dict | None]:
    """Run a chat completion and parse a JSON object out of the response.

    Retries once with a corrective hint when the model returns unparseable
    output (markdown fences, trailing prose, etc.).

    Returns (parsed_dict, usage_metadata). usage_metadata is the metadata from
    the final LLM call (the one that succeeded), or from the first call if the
    retry also fails.
    """
    import json

    raw, usage = await complete(messages=messages, model=model, temperature=temperature, json_mode=True)
    text = _strip_fence(raw.strip())

    try:
        return json.loads(text), usage
    except json.JSONDecodeError:
        logger.warning("LLM returned invalid JSON; retrying with a corrective hint")
        corrected = messages + [
            {
                "role": "user",
                "content": (
                    "Your previous reply was not valid JSON. Reply with ONLY a "
                    "valid JSON object, no markdown fences, no commentary."
                ),
            }
        ]
        raw, retry_usage = await complete(messages=corrected, model=model, temperature=temperature, json_mode=True)
        text = _strip_fence(raw.strip())
        return json.loads(text), retry_usage


def _strip_fence(text: str) -> str:
    """Extract JSON content from a fenced code block, handling embedded preamble.

    Uses re.search (not re.match) so preamble text before the fence doesn't
    break extraction. Returns the inner content of the fence, or the original
    text unchanged if no fence is present.
    """
    import re

    m = re.search(r"```(?:json)?\s*\n?(.*?)\s*```", text, re.DOTALL)
    return m.group(1) if m else text
