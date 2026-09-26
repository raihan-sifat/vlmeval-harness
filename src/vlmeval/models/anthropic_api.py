"""Anthropic adapter (Claude models).

The Messages API takes images as base64 blocks inside a content list and
requires ``max_tokens`` on every call, so the payload shape differs from
OpenAI's. Token usage arrives as ``input_tokens`` / ``output_tokens``.
"""

from __future__ import annotations

import base64
import os
import time
from typing import Any

from ..pricing import lookup
from ..types import ModelResponse
from .base import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, ModelAdapter, encode_image


class AnthropicModel(ModelAdapter):
    """Messages API adapter for Claude models.

    Args:
        model: Model id, e.g. ``"claude-haiku-4-5"``.
        name: Label for reports. Defaults to the model id.
        api_key: Defaults to ``$ANTHROPIC_API_KEY``.
        max_side: Longest image side sent, in pixels. Claude downsamples large
            images anyway, so capping here saves bandwidth without hurting
            accuracy on these tasks.
        max_tokens: Required by the API. The default suits single-token answers.
        thinking_budget: Enables extended thinking with this many tokens. Off
            by default: it multiplies latency and cost, and for a letter-answer
            task it usually changes nothing.
    """

    supports_images = True

    def __init__(
        self,
        model: str = "claude-haiku-4-5",
        name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_side: int = 1024,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout: float = 60.0,
        thinking_budget: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "The Anthropic adapter needs the anthropic package: "
                "pip install -e '.[anthropic]'"
            ) from exc

        self.model = model
        self.name = name or model
        self.max_side = max_side
        self.default_max_tokens = max_tokens
        self.default_temperature = temperature
        self.thinking_budget = thinking_budget

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key, "
                "or pass api_key=... explicitly."
            )
        url = base_url or os.environ.get("ANTHROPIC_BASE_URL") or None
        self._client = Anthropic(api_key=key, base_url=url, timeout=timeout)

        self.cost_per_million = lookup("anthropic", model)
        self.info = {
            "backend": "anthropic",
            "model": model,
            "max_side": max_side,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "thinking_budget": thinking_budget,
            "cost_in_per_mtok": self.cost_per_million[0],
            "cost_out_per_mtok": self.cost_per_million[1],
        }

    def _content(self, image: Any, prompt: str) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        if image is not None:
            data = encode_image(image, "PNG", self.max_side)
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.b64encode(data).decode("ascii"),
                    },
                }
            )
        blocks.append({"type": "text", "text": prompt})
        return blocks

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
        max_tokens = self.default_max_tokens if max_tokens is None else max_tokens
        temperature = self.default_temperature if temperature is None else temperature

        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": self._content(image, prompt)}],
        }
        if system:
            payload["system"] = system
        if temperature is not None:
            payload["temperature"] = temperature
        if self.thinking_budget:
            # Extended thinking is incompatible with a custom temperature.
            payload.pop("temperature", None)
            payload["thinking"] = {"type": "enabled", "budget_tokens": self.thinking_budget}
            payload["max_tokens"] = max(max_tokens, self.thinking_budget + 256)
        payload.update(kwargs)

        started = time.perf_counter()
        try:
            message = self._client.messages.create(**payload)
        except Exception as exc:  # noqa: BLE001 - surfaced as a recorded failure
            return ModelResponse(
                text="",
                latency_s=time.perf_counter() - started,
                error=f"{type(exc).__name__}: {exc}",
            )

        # Extended thinking returns interleaved thinking/text blocks; only the
        # text blocks are the answer.
        text = "".join(
            getattr(block, "text", "") for block in message.content if block.type == "text"
        )
        usage = message.usage
        return ModelResponse(
            text=text.strip(),
            prompt_tokens=getattr(usage, "input_tokens", 0) or 0,
            completion_tokens=getattr(usage, "output_tokens", 0) or 0,
            latency_s=time.perf_counter() - started,
            finish_reason=getattr(message, "stop_reason", None),
        )

    def close(self) -> None:
        self._client.close()
