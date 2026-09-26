"""The model adapter contract.

Every backend -- a cloud API, a local Hugging Face checkpoint, or the offline
dummy used in CI -- implements one method: turn an image plus a prompt into a
:class:`~vlmeval.types.ModelResponse`. Everything downstream (tasks, metrics,
reports) is written against this contract only, so adding a provider is a
single file plus a registry entry, and no task code changes.

Two conventions keep cross-model comparison honest:

* Adapters report token usage when the provider exposes it. The runner turns
  that into cost estimates, so a cheap model that scores 1 point lower is a
  visibly different trade-off rather than an invisible one.
* Adapters never silently swallow a failure. A failed call returns a response
  with ``error`` set; the runner retries, then records the item as an error
  rather than dropping it, so ``n_reported`` always matches ``n_attempted``.
"""

from __future__ import annotations

import abc
from typing import Any

from ..types import ModelResponse

DEFAULT_MAX_TOKENS = 16
DEFAULT_TEMPERATURE = 0.0


class ModelAdapter(abc.ABC):
    """Base class for all vision-language backends.

    Subclasses must set :attr:`name` and implement :meth:`generate`.
    """

    #: Identifier used in result filenames, tables and cache keys. Keep it
    #: filesystem-safe: it is interpolated into paths.
    name: str = "unnamed-model"

    #: Whether the backend accepts images at all. A text-only control model is
    #: a legitimate thing to evaluate: the gap between it and a VLM of similar
    #: language ability isolates how much the vision pathway contributes.
    supports_images: bool = True

    #: Whether the adapter tolerates concurrent calls. Cloud SDK clients are
    #: thread-safe; a locally loaded model is not, and asking for 16 workers
    #: against a 4 GB GPU yields an out-of-memory error rather than a speedup.
    #: The runner reads this and clamps concurrency to 1 when it is False.
    parallel_safe: bool = True

    #: Rough USD per 1M prompt / completion tokens, for cost accounting. Cloud
    #: adapters should override with the provider's published list price.
    cost_per_million: tuple[float, float] = (0.0, 0.0)

    #: Free-form provenance recorded in the run manifest (model revision,
    #: resolution, quantisation, ...). Reproducibility depends on this.
    info: dict[str, Any]

    def __init__(self, **kwargs: Any) -> None:
        self.info = {}
        self._init(**kwargs)

    def _init(self, **kwargs: Any) -> None:
        """Backend-specific setup. Override instead of ``__init__``."""

    @abc.abstractmethod
    def generate(
        self,
        *,
        image: Any,
        prompt: str,
        system: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        **kwargs: Any,
    ) -> ModelResponse:
        """Return the model's reply to `prompt` about `image`."""

    def generate_with_context(
        self,
        *,
        image: Any,
        prompt: str,
        system: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Call :meth:`generate`, discarding harness-internal `context`.

        The runner needs to hand some adapters information that has no business
        reaching a provider -- the gold label, the item id, the category. Those
        travel through this method rather than through ``**kwargs``, and the
        default implementation drops them on the floor.

        That is the whole point of the split. An adapter that forwards unknown
        keyword arguments to a provider (a reasonable thing for a chat API to
        support) would otherwise leak the answer key into the request and quietly
        turn every evaluation into a 100% score. Here, an adapter has to opt in
        explicitly to see the context, and only the offline test double does.
        """
        return self.generate(
            image=image,
            prompt=prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """USD estimate for one call, from :attr:`cost_per_million`."""
        in_rate, out_rate = self.cost_per_million
        return (prompt_tokens * in_rate + completion_tokens * out_rate) / 1e6

    def close(self) -> None:
        """Release any held resources. Safe to call more than once."""

    def __enter__(self) -> ModelAdapter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


def encode_image(image: Any, format: str = "PNG", max_side: int | None = None) -> bytes:
    """Normalise any accepted image input to encoded bytes.

    Cloud providers want raw bytes; local models want a PIL image. Centralising
    the conversion here means adapters deal with one representation, and image
    resizing happens exactly once instead of once per provider quirk.
    """
    from io import BytesIO

    from PIL import Image

    if isinstance(image, (bytes, bytearray)):
        raw = bytes(image)
        if max_side:
            img = Image.open(BytesIO(raw))
            img = _resize(img, max_side)
            buf = BytesIO()
            img.save(buf, format=format)
            return buf.getvalue()
        return raw

    if isinstance(image, (str,)):
        from pathlib import Path

        image = Path(image)

    if isinstance(image, Image.Image):
        img = image
    else:
        img = Image.open(image)

    img = img.convert("RGB")
    if max_side:
        img = _resize(img, max_side)
    buf = BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


def _resize(img: Any, max_side: int) -> Any:
    """Downscale so the longest side is at most `max_side`, preserving aspect."""
    from PIL import Image

    longest = max(img.size)
    if longest <= max_side:
        return img
    scale = max_side / float(longest)
    size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
    return img.resize(size, Image.LANCZOS)


def load_pil(image: Any) -> Any:
    """Return a PIL image for any accepted image input, RGB, capped in size.

    The cap protects against pathological inputs (a 20000px scan will otherwise
    be rejected or silently truncated by some providers) and keeps local GPU
    memory use predictable.
    """
    from PIL import Image

    if isinstance(image, Image.Image):
        img = image
    elif isinstance(image, (bytes, bytearray)):
        from io import BytesIO

        img = Image.open(BytesIO(bytes(image)))
    else:
        img = Image.open(image)
    img = img.convert("RGB")
    return _resize(img, 2048)
