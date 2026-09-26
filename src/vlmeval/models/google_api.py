"""Google Gemini adapter.

Uses the ``google-genai`` SDK. Images are passed as inline bytes parts, and
usage comes back on ``usage_metadata`` rather than a ``usage`` object, which is
the main reason this is a separate file instead of a configuration of the
OpenAI adapter.
"""

from __future__ import annotations

import os
import time
from typing import Any

from ..pricing import lookup
from ..types import ModelResponse
from .base import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, ModelAdapter, encode_image


class GoogleModel(ModelAdapter):
    """Gemini adapter.

    Args:
        model: Model id, e.g. ``"gemini-2.5-flash"``.
        name: Label for reports. Defaults to the model id.
        api_key: Defaults to ``$GOOGLE_API_KEY`` or ``$GEMINI_API_KEY``.
        max_side: Longest image side sent, in pixels.
        max_tokens: Cap on the reply.
        response_mime_type: Set to ``"application/json"`` to constrain the
            reply shape. Off by default because several Gemini models reject
            it alongside images, and the parsing layer already handles prose.
    """

    supports_images = True

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        name: str | None = None,
        api_key: str | None = None,
        max_side: int = 1024,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout: float = 60.0,
        response_mime_type: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "The Gemini adapter needs google-genai: pip install -e '.[google]'"
            ) from exc

        self._genai = genai
        self.model = model
        self.name = name or model
        self.max_side = max_side
        self.default_max_tokens = max_tokens
        self.default_temperature = temperature
        self.response_mime_type = response_mime_type

        key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key, "
                "or pass api_key=... explicitly."
            )
        self._client = genai.Client(api_key=key)

        self.cost_per_million = lookup("google", model)
        self.info = {
            "backend": "google",
            "model": model,
            "max_side": max_side,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "cost_in_per_mtok": self.cost_per_million[0],
            "cost_out_per_mtok": self.cost_per_million[1],
        }

    def generate(
        self,
        *,
        image: Any,
        prompt: str,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        from google.genai import types

        max_tokens = self.default_max_tokens if max_tokens is None else max_tokens
        temperature = self.default_temperature if temperature is None else temperature

        parts: list[Any] = []
        if image is not None:
            parts.append(
                types.Part.from_bytes(
                    data=encode_image(image, "PNG", self.max_side), mime_type="image/png"
                )
            )
        parts.append(types.Part.from_text(text=prompt))

        config_kwargs: dict[str, Any] = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            config_kwargs["system_instruction"] = system
        if self.response_mime_type:
            config_kwargs["response_mime_type"] = self.response_mime_type

        started = time.perf_counter()
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=types.Content(role="user", parts=parts),
                config=types.GenerateContentConfig(**config_kwargs),
            )
        except Exception as exc:  # noqa: BLE001 - surfaced as a recorded failure
            return ModelResponse(
                text="",
                latency_s=time.perf_counter() - started,
                error=f"{type(exc).__name__}: {exc}",
            )

        # `text` raises when every candidate was blocked or filtered, which is
        # a legitimate outcome the report needs to see rather than an exception.
        try:
            text = (response.text or "").strip()
        except Exception:  # noqa: BLE001
            text = ""

        usage = getattr(response, "usage_metadata", None)
        finish = None
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            finish = getattr(candidates[0], "finish_reason", None)
            finish = getattr(finish, "name", None) or str(finish) if finish else None

        return ModelResponse(
            text=text,
            prompt_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            completion_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            latency_s=time.perf_counter() - started,
            finish_reason=finish,
        )
