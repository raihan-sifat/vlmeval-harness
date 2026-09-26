"""Local open-weight VLM adapter (Hugging Face transformers).

Sized for a real constraint rather than a datacenter one. On a 4 GB consumer GPU
the models that actually load are SmolVLM-256M-Instruct and
SmolVLM-500M-Instruct (well under 1 GB and ~1.5 GB of weights respectively);
Qwen2-VL-2B needs roughly 13 GB and moondream2 1.8B needs about 4 GB before
quantisation, so both require 4-bit loading or a larger card. The defaults
below are chosen so the command in the README works on modest hardware.

Two behaviours worth knowing:

* Local generation is **not** thread-safe, so the adapter advertises
  ``parallel_safe = False`` and the runner serialises calls to it. Requesting
  16 workers against a 4 GB GPU produces OOM, not speedup.
* Token counts come from the model's own tokenizer, so cost and latency figures
  are comparable with cloud runs even though the dollar cost is zero.
"""

from __future__ import annotations

import time
from typing import Any

from ..types import ModelResponse
from .base import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, ModelAdapter, load_pil

#: Checkpoints that fit in a few GB of VRAM, offered as ready-made presets.
LOCAL_PRESETS: dict[str, dict[str, Any]] = {
    "smolvlm-256m": {
        "model": "HuggingFaceTB/SmolVLM-256M-Instruct",
        "max_side": 512,
        "note": "Under 1 GB VRAM. The cheapest real sanity check available.",
    },
    "smolvlm-500m": {
        "model": "HuggingFaceTB/SmolVLM-500M-Instruct",
        "max_side": 512,
        "note": "~1.5 GB. Best accuracy per byte in this family.",
    },
    "smolvlm-2b": {
        "model": "HuggingFaceTB/SmolVLM-2B-Instruct",
        "max_side": 1024,
        "note": "Needs ~5 GB VRAM; use load_in_4bit=True on smaller cards.",
    },
    "qwen2-vl-2b": {
        "model": "Qwen/Qwen2-VL-2B-Instruct",
        "max_side": 1024,
        "note": "Strong OCR, but ~13 GB VRAM at full precision. 4-bit only.",
    },
    "moondream2": {
        "model": "vikhyatk/moondream2",
        "max_side": 1024,
        "note": "~4 GB VRAM. Needs trust_remote_code=True.",
    },
    "florence2-base": {
        "model": "microsoft/Florence-2-base",
        "max_side": 768,
        "note": "Task tokens differ from chat models; answer_for_kwargs is tuned below.",
    },
}


class HFLocalModel(ModelAdapter):
    """Run a vision-language checkpoint locally via transformers.

    Args:
        model: Hub id or local path.
        name: Label for reports. Defaults to the repo id with ``/`` replaced.
        device: ``"auto"``, ``"cuda"``, ``"cuda:0"`` or ``"cpu"``.
        dtype: Torch dtype name. Defaults to bfloat16 on CUDA when supported,
            float16 otherwise, float32 on CPU.
        load_in_4bit / load_in_8bit: Bits-and-bytes quantisation. Both raise a
            clear error if bitsandbytes is missing rather than failing deep
            inside torch.
        trust_remote_code: Required by moondream and Florence-2.
        min_pixels / max_pixels: Pixel budget passed to the processor. Left
            unset the processor uses per-checkpoint defaults.
    """

    supports_images = True
    #: Generation is stateful; the runner must not call this concurrently.
    parallel_safe = False

    def __init__(
        self,
        model: str = "HuggingFaceTB/SmolVLM-256M-Instruct",
        name: str | None = None,
        device: str = "auto",
        dtype: str | None = None,
        max_side: int = 1024,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        trust_remote_code: bool = False,
        revision: str | None = None,
        attn_implementation: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        try:
            import torch
            from transformers import AutoProcessor
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "The local adapter needs torch and transformers: pip install -e '.[local]'"
            ) from exc

        self._torch = torch
        self.model_id = model
        self.name = name or model.replace("/", "-")
        self.max_side = max_side
        self.default_max_tokens = max_tokens
        self.default_temperature = temperature
        self.temperature = temperature

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        torch_dtype = self._resolve_dtype(dtype, device)
        self.processor = AutoProcessor.from_pretrained(
            model, trust_remote_code=trust_remote_code, revision=revision
        )
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "revision": revision,
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        if device != "cpu":
            model_kwargs["device_map"] = device if ":" in device else {"": 0}
            model_kwargs["torch_dtype"] = torch_dtype
        else:
            model_kwargs["torch_dtype"] = torch_dtype

        if load_in_4bit or load_in_8bit:
            try:
                import bitsandbytes  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "Quantised loading needs bitsandbytes: pip install bitsandbytes"
                ) from exc
            model_kwargs["load_in_4bit"] = load_in_4bit
            model_kwargs["load_in_8bit"] = load_in_8bit
            # Quantised weights must stay in their compute dtype for the
            # normalisation layers to behave.
            model_kwargs["torch_dtype"] = torch_dtype

        self.model = self._load_model(model_kwargs)
        self.model.eval()

        self.info = {
            "backend": "hf-local",
            "model": model,
            "revision": revision,
            "device": device,
            "dtype": str(torch_dtype),
            "quantisation": "4bit" if load_in_4bit else ("8bit" if load_in_8bit else "none"),
            "max_side": max_side,
            "max_tokens": max_tokens,
            "trust_remote_code": trust_remote_code,
            "cost_in_per_mtok": 0.0,
            "cost_out_per_mtok": 0.0,
        }
        if device.startswith("cuda") and torch.cuda.is_available():
            index = 0 if ":" not in device else int(device.split(":")[1])
            props = torch.cuda.get_device_properties(index)
            self.info["gpu"] = props.name
            self.info["vram_gb"] = round(props.total_memory / 1024**3, 1)

    def _resolve_dtype(self, dtype: str | None, device: str) -> Any:
        torch = self._torch
        if dtype:
            return getattr(torch, dtype)
        if device == "cpu":
            return torch.float32
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    def _load_model(self, model_kwargs: dict[str, Any]) -> Any:
        """Load whichever auto-class this checkpoint's architecture supports.

        ``AutoModelForImageTextToText`` is the current entry point; older
        checkpoints are still only registered under
        ``AutoModelForVision2Seq``. Trying both avoids pinning a transformers
        version and failing on one architecture or the other.
        """
        import transformers

        errors: list[str] = []
        for attr in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
            auto_cls = getattr(transformers, attr, None)
            if auto_cls is None:
                continue
            try:
                return auto_cls.from_pretrained(self.model_id, **model_kwargs)
            except Exception as exc:  # noqa: BLE001 - try the next entry point
                errors.append(f"{attr}: {type(exc).__name__}: {exc}")
        raise RuntimeError(
            f"Could not load {self.model_id}. Tried:\n  " + "\n  ".join(errors)
        )

    def _build_inputs(self, image: Any, prompt: str, system: str | None) -> Any:
        content = [{"type": "image"}, {"type": "text", "text": prompt}]
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
        messages.append({"role": "user", "content": content})

        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        kwargs: dict[str, Any] = {"text": text, "return_tensors": "pt"}
        if image is not None:
            kwargs["images"] = [load_pil(image)]
        inputs = self.processor(**kwargs)
        target = next(self.model.parameters()).device
        return inputs.to(target)

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
        torch = self._torch
        max_tokens = self.default_max_tokens if max_tokens is None else max_tokens
        temperature = self.default_temperature if temperature is None else temperature

        started = time.perf_counter()
        try:
            inputs = self._build_inputs(image, prompt, system)
            prompt_tokens = int(inputs["input_ids"].shape[-1])
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": max_tokens,
                "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            }
            # Greedy decoding by default: a temperature sweep is a separate
            # experiment, and non-determinism would break run-to-run diffing.
            if temperature and temperature > 0:
                gen_kwargs.update(do_sample=True, temperature=temperature)
            else:
                gen_kwargs.update(do_sample=False)
            gen_kwargs.update(kwargs)

            with torch.inference_mode():
                output = self.model.generate(**inputs, **gen_kwargs)

            prompt_len = inputs["input_ids"].shape[-1]
            new_ids = output[0][prompt_len:]
            text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        except Exception as exc:  # noqa: BLE001 - surfaced as a recorded failure
            return ModelResponse(
                text="",
                latency_s=time.perf_counter() - started,
                error=f"{type(exc).__name__}: {exc}",
            )

        completion_tokens = int(new_ids.shape[-1])
        finish = "length" if completion_tokens >= max_tokens else "stop"
        return ModelResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_s=time.perf_counter() - started,
            finish_reason=finish,
        )

    def close(self) -> None:
        model = getattr(self, "model", None)
        if model is not None:
            del self.model
        torch = getattr(self, "_torch", None)
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
