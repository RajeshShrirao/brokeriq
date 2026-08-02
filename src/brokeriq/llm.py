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
) -> str:
    """Run a chat completion and return the text content.

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
        return response.choices[0].message.content or ""
    except Exception as exc:  # litellm raises a grab-bag of exceptions
        logger.error("LLM call failed (model=%s): %s", model, exc)
        raise


async def complete_json(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.0,
) -> dict:
    """Run a chat completion and parse a JSON object out of the response.

    Retries once with a corrective hint when the model returns unparseable
    output (markdown fences, trailing prose, etc.).
    """
    import json
    import re

    raw = await complete(messages=messages, model=model, temperature=temperature, json_mode=True)
    text = raw.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)

    try:
        return json.loads(text)
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
        raw = await complete(messages=corrected, model=model, temperature=temperature, json_mode=True)
        text = raw.strip()
        fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
        if fenced:
            text = fenced.group(1)
        return json.loads(text)
