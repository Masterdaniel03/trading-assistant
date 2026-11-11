"""The LLM behind the research layer. Gemini for now (its free tier is enough).

Research code only needs an object with a `model` name and a
`generate_json(prompt, schema)` method, so another provider can be dropped in
later without touching anything else.
"""
from __future__ import annotations

from dataclasses import dataclass

from tradebot.retry import call_with_retry

DEFAULT_MODEL = "gemini-3.8-flash"
TIMEOUT_SECONDS = 60


class LLMError(Exception):
    pass


@dataclass(frozen=True)
class LLMReply:
    text: str            # the JSON the model wrote
    input_tokens: int
    output_tokens: int


class GeminiLLM:
    def __init__(self, api_key: str | None, model: str = DEFAULT_MODEL,
                 timeout_s: int = TIMEOUT_SECONDS):
        if not api_key:
            raise RuntimeError("Missing GEMINI_API_KEY in .env (the research command needs it)")
        from google import genai
        from google.genai import types

        self.model = model
        self._types = types
        self._client = genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=timeout_s * 1000))

    def generate_json(self, prompt: str, schema) -> LLMReply:
        """Ask for an answer shaped like `schema` (a pydantic model class).

        Rate limits and dropped connections are retried with backoff; the
        library's own retries are off by default, so this is the only layer.
        """
        config = self._types.GenerateContentConfig(
            response_mime_type="application/json", response_schema=schema)
        response = call_with_retry(self._client.models.generate_content, model=self.model,
                                   contents=prompt, config=config, what=f"gemini {self.model}")
        text = (response.text or "").strip()
        if not text:
            raise LLMError("the model returned an empty answer")
        usage = response.usage_metadata
        return LLMReply(text,
                        (usage.prompt_token_count or 0) if usage else 0,
                        (usage.candidates_token_count or 0) if usage else 0)
