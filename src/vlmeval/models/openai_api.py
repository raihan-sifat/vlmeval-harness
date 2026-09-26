"""OpenAI adapter (GPT-4o / GPT-4.1 / GPT-5 family and compatible endpoints).

The interesting engineering here is not the happy path but the API drift. The
GPT-5 family rejects ``temperature``, newer models renamed ``max_tokens`` to
``max_completion_tokens``, and gateways (LiteLLM, vLLM, OpenRouter) differ
again. Rather than pinning one API version, this adapter detects a rejected
parameter and retries without it, recording what it had to change in
``info['compat']`` so a result file shows when the call was not quite the
standard one.
"""

from __future__ import annotations

import base64
import os
import time
from typing import Any

from ..pricing import lookup
from ..types import ModelResponse
from .base import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, ModelAdapter, encode_image


class OpenAIModel(ModelAdapter):
    """Chat Completions adapter for OpenAI and OpenAI-compatible endpoints.

    Args:
        model: Model id, e.g. ``"gpt-4o"``. Any id your base URL accepts works.
        name: Label for reports. Defaults to the model id.
        api_key: Defaults to ``$OPENAI_API_KEY``.
        base_url: Defaults to ``$OPENAI_BASE_URL`` or the OpenAI endpoint.
        max_side: Longest image side sent, in pixels. Downscale here rather
            than letting the provider reject an oversized image.
        max_tokens: Cap on the reply. Kept small by default: these tasks ask
            for a single letter or a yes/no, and a long reply only adds cost
            and parsing ambiguity.
        reasoning_effort: Forwarded for reasoning models (``minimal`` ..
            ``high``). Left unset otherwise.
    """

    supports_images = True

    def __init__(
        self,
        model: str = "gpt-4o",
        name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_side: int = 1024,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout: float = 60.0,
        reasoning_effort: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "The OpenAI adapter needs the openai package: pip install -e '.[openai]'"
            ) from exc

        self.model = model
        self.name = name or model
        self.max_side = max_side
        self.default_max_tokens = max_tokens
        self.default_temperature = temperature
        self.reasoning_effort = reasoning_effort

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key, "
                "or pass api_key=... explicitly."
            )
        url = base_url or os.environ.get("OPENAI_BASE_URL") or None
        self._client = OpenAI(api_key=key, base_url=url, timeout=timeout)

        self.cost_per_million = lookup("openai", model)
        self.info = {
            "backend": "openai",
            "model": model,
            "base_url": url or "https://api.openai.com/v1",
            "max_side": max_side,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "reasoning_effort": reasoning_effort,
            "cost_in_per_mtok": self.cost_per_million[0],
            "cost_out_per_mtok": self.cost_per_million[1],
            "compat": [],
        }

    def _content(self, image: Any, prompt: str) -> list[dict[str, Any]]:
        if image is None:
            return [{"type": "text", "text": prompt}]
        data = encode_image(image, "PNG", self.max_side)
        b64 = base64.b64encode(data).decode("ascii")
        return [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]

    def _request(self, **payload: Any) -> Any:
        return self._client.chat.completions.create(**payload)

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

        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": self._content(image, prompt)})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        payload.update(kwargs)

        started = time.perf_counter()
        compat: list[str] = []
        # At most three shapes are tried: the canonical payload, then the two
        # known renames. Bounded, because an unbounded retry loop against a
        # metered API is a bad way to learn a schema.
        for _ in range(3):
            try:
                completion = self._request(**payload)
                break
            except Exception as exc:  # noqa: BLE001 - message drives the decision
                message = str(exc).lower()
                if (
                    "max_tokens" in message
                    and "unsupported" in message
                    and "max_tokens" in payload
                ):
                    # Guard on the parameter still being present, not on the one
                    # being renamed to. Checking the target instead makes the
                    # branch unreachable on the first failure -- which is the
                    # only time it is ever needed -- so the rename silently
                    # never happens and a GPT-5-style call just fails.
                    payload["max_completion_tokens"] = payload.pop("max_tokens")
                    compat.append("max_tokens->max_completion_tokens")
                    continue
                if "temperature" in message and "temperature" in payload:
                    payload.pop("temperature")
                    compat.append("dropped temperature")
                    continue
                return ModelResponse(
                    text="",
                    latency_s=time.perf_counter() - started,
                    error=f"{type(exc).__name__}: {exc}",
                )
        else:
            return ModelResponse(
                text="",
                latency_s=time.perf_counter() - started,
                error="exhausted OpenAI parameter compatibility retries",
            )

        if compat and compat not in self.info["compat"]:
            self.info["compat"] = [*self.info["compat"], *compat]

        choice = completion.choices[0]
        usage = completion.usage
        return ModelResponse(
            text=(choice.message.content or "").strip(),
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_s=time.perf_counter() - started,
            finish_reason=getattr(choice, "finish_reason", None),
        )

    def close(self) -> None:
        self._client.close()
