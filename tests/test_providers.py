"""Provider adapters, exercised against stub SDKs.

These are the files that spend money, and they are the files the offline `echo`
path never touches. A syntax error once sat unnoticed in `google_api.py`
precisely because no test could import it without the SDK installed.

Every SDK is therefore replaced with a recording fake, which buys three things
the echo backend cannot give:

* the exact request payload is asserted, so a silent API-shape regression is a
  test failure rather than a surprise bill;
* API errors are turned into `ModelResponse(ok=False)` instead of propagating,
  which is what keeps one bad item from killing a 5000-item run;
* the adapters' `generate_with_context` is proven to ignore the harness's gold
  label, so an answer key cannot reach a paid endpoint.
"""

from __future__ import annotations

import sys
import types
import unittest
from contextlib import contextmanager
from typing import Any, Iterator

from tests.helpers import *  # noqa: F401,F403 - path bootstrap


@contextmanager
def stub_modules(**names: Any) -> Iterator[None]:
    """Temporarily install fake modules under `names` in `sys.modules`."""
    saved = {name: sys.modules.get(name) for name in names}
    sys.modules.update(names)
    try:
        yield
    finally:
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


class SDKStubTest(unittest.TestCase):
    """Base class whose stubs stay installed for the whole test.

    The adapters import their SDK lazily -- inside `generate`, not just the
    constructor -- so a stub that is removed once the model is built disappears
    before the code under test needs it.
    """

    def install(self, **names: Any) -> None:
        saved = {name: sys.modules.get(name) for name in names}
        sys.modules.update(names)

        def restore() -> None:
            for name, original in saved.items():
                if original is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = original

        self.addCleanup(restore)


def ns(**fields: Any) -> Any:
    """A simple attribute bag, standing in for an SDK response object."""
    return types.SimpleNamespace(**fields)


class Recorder:
    """Collects request payloads and replays a scripted list of results."""

    def __init__(self, *results: Any) -> None:
        self.payloads: list[dict[str, Any]] = []
        self._results = list(results)

    def create(self, **payload: Any) -> Any:
        self.payloads.append(payload)
        result = self._results.pop(0) if self._results else ns()
        if isinstance(result, Exception):
            raise result
        return result


def openai_completion(text: str = "C") -> Any:
    return ns(
        choices=[ns(message=ns(content=text), finish_reason="stop")],
        usage=ns(prompt_tokens=11, completion_tokens=3),
    )


class TestOpenAIAdapter(SDKStubTest):
    def _model(self, recorder: Recorder, **kwargs: Any) -> Any:
        completions = ns(create=recorder.create)
        client = ns(chat=ns(completions=completions), close=lambda: None)
        self.install(openai=ns(OpenAI=lambda **_: client))
        from vlmeval.models.openai_api import OpenAIModel

        return OpenAIModel(model="gpt-4o", api_key="test-key", **kwargs)

    def test_missing_api_key_names_the_fix(self) -> None:
        import os

        with stub_modules(openai=ns(OpenAI=lambda **_: None)):
            from vlmeval.models.openai_api import OpenAIModel

            saved = os.environ.pop("OPENAI_API_KEY", None)
            try:
                with self.assertRaises(ValueError) as ctx:
                    OpenAIModel(model="gpt-4o")
            finally:
                if saved is not None:
                    os.environ["OPENAI_API_KEY"] = saved
        self.assertIn("OPENAI_API_KEY", str(ctx.exception))

    def test_payload_shape(self) -> None:
        recorder = Recorder(openai_completion())
        model = self._model(recorder)
        response = model.generate(image=None, prompt="What colour?")
        payload = recorder.payloads[0]
        self.assertEqual(payload["model"], "gpt-4o")
        self.assertEqual(payload["messages"][-1]["role"], "user")
        self.assertEqual(payload["messages"][-1]["content"][0]["text"], "What colour?")
        self.assertIn("max_tokens", payload)
        self.assertTrue(response.ok)

    def test_system_prompt_becomes_a_system_message(self) -> None:
        recorder = Recorder(openai_completion())
        model = self._model(recorder)
        model.generate(image=None, prompt="Q", system="Answer from the image only.")
        self.assertEqual(recorder.payloads[0]["messages"][0]["role"], "system")
        self.assertEqual(
            recorder.payloads[0]["messages"][0]["content"], "Answer from the image only."
        )

    def test_reply_is_stripped_and_usage_recorded(self) -> None:
        recorder = Recorder(openai_completion("  C  "))
        model = self._model(recorder)
        response = model.generate(image=None, prompt="Q")
        self.assertEqual(response.text, "C")
        self.assertEqual(response.prompt_tokens, 11)
        self.assertEqual(response.completion_tokens, 3)

    def test_api_error_becomes_a_failed_response(self) -> None:
        recorder = Recorder(RuntimeError("rate limited"))
        model = self._model(recorder)
        response = model.generate(image=None, prompt="Q")
        self.assertFalse(response.ok)
        self.assertIn("RuntimeError", response.error or "")

    def test_renames_max_tokens_when_the_api_rejects_it(self) -> None:
        recorder = Recorder(
            RuntimeError("unsupported parameter: max_tokens"), openai_completion()
        )
        model = self._model(recorder)
        response = model.generate(image=None, prompt="Q")
        self.assertTrue(response.ok)
        self.assertIn("max_completion_tokens", recorder.payloads[1])
        self.assertNotIn("max_tokens", recorder.payloads[1])
        self.assertIn("max_tokens->max_completion_tokens", model.info["compat"])

    def test_drops_temperature_when_the_api_rejects_it(self) -> None:
        recorder = Recorder(RuntimeError("temperature is not supported"), openai_completion())
        model = self._model(recorder)
        response = model.generate(image=None, prompt="Q")
        self.assertTrue(response.ok)
        self.assertNotIn("temperature", recorder.payloads[1])
        self.assertIn("dropped temperature", model.info["compat"])

    def test_retries_are_bounded(self) -> None:
        recorder = Recorder(*[RuntimeError("unsupported parameter: temperature")] * 5)
        model = self._model(recorder)
        response = model.generate(image=None, prompt="Q")
        self.assertFalse(response.ok)
        self.assertLessEqual(len(recorder.payloads), 3)

    def test_gold_never_reaches_the_payload(self) -> None:
        """The paid endpoint must not be able to see the answer key."""
        recorder = Recorder(openai_completion())
        model = self._model(recorder)
        model.generate_with_context(
            image=None,
            prompt="What colour?",
            context={"gold": "C", "display_gold": "C", "uid": "item-1"},
        )
        self.assertNotIn("C", str(recorder.payloads[0]["messages"][0]["content"][0]))


class TestAnthropicAdapter(SDKStubTest):
    def _model(self, recorder: Recorder, **kwargs: Any) -> Any:
        client = ns(messages=ns(create=recorder.create), close=lambda: None)
        self.install(anthropic=ns(Anthropic=lambda **_: client))
        from vlmeval.models.anthropic_api import AnthropicModel

        return AnthropicModel(model="claude-haiku-4-5", api_key="test-key", **kwargs)

    def test_max_tokens_is_always_sent(self) -> None:
        recorder = Recorder(
            ns(
                content=[ns(type="text", text="yes"), ns(type="thinking", thinking="hmm")],
                usage=ns(input_tokens=9, output_tokens=2),
                stop_reason="end_turn",
            )
        )
        model = self._model(recorder)
        response = model.generate(image=None, prompt="Is there a dog?")
        payload = recorder.payloads[0]
        self.assertIn("max_tokens", payload)
        self.assertEqual(payload["model"], "claude-haiku-4-5")
        # Only text blocks become the answer; thinking blocks must not leak in.
        self.assertEqual(response.text, "yes")
        self.assertEqual(response.prompt_tokens, 9)
        self.assertEqual(response.completion_tokens, 2)

    def test_error_becomes_a_failed_response(self) -> None:
        recorder = Recorder(RuntimeError("overloaded"))
        model = self._model(recorder)
        self.assertFalse(model.generate(image=None, prompt="Q").ok)

    def test_gold_never_reaches_the_payload(self) -> None:
        recorder = Recorder(
            ns(content=[ns(type="text", text="yes")], usage=ns(input_tokens=1, output_tokens=1))
        )
        model = self._model(recorder)
        model.generate_with_context(
            image=None, prompt="Is there a dog?", context={"gold": "yes"}
        )
        self.assertNotIn("gold", str(recorder.payloads[0]))


class TestGoogleAdapter(SDKStubTest):
    def _model(self, recorder: Recorder, **kwargs: Any) -> Any:
        def _generate_content(**payload: Any) -> Any:
            return recorder.create(**payload)

        client = ns(models=ns(generate_content=_generate_content), close=lambda: None)
        genai_types = ns(
            Part=ns(
                from_text=lambda text: ns(text=text),
                from_bytes=lambda data, mime_type: ns(data=data, mime_type=mime_type),
            ),
            Content=lambda **kw: ns(**kw),
            GenerateContentConfig=lambda **kw: ns(**kw),
        )
        genai = ns(Client=lambda **_: client, types=genai_types)
        self.install(google=ns(genai=genai), **{"google.genai": genai, "google.genai.types": genai_types})
        from vlmeval.models.google_api import GoogleModel

        return GoogleModel(model="gemini-2.5-flash", api_key="test-key", **kwargs)

    def test_request_shape(self) -> None:
        recorder = Recorder(
            ns(
                text="B",
                usage_metadata=ns(prompt_token_count=8, candidates_token_count=2),
            )
        )
        model = self._model(recorder)
        response = model.generate(image=None, prompt="What shape?")
        payload = recorder.payloads[0]
        self.assertEqual(payload["model"], "gemini-2.5-flash")
        self.assertEqual(payload["contents"].role, "user")
        self.assertTrue(response.ok)
        self.assertEqual(response.text, "B")

    def test_error_becomes_a_failed_response(self) -> None:
        recorder = Recorder(RuntimeError("quota exceeded"))
        model = self._model(recorder)
        self.assertFalse(model.generate(image=None, prompt="Q").ok)


class TestEveryModuleCompiles(unittest.TestCase):
    def test_package_has_no_syntax_errors(self) -> None:
        """A stub-free backstop.

        The provider tests above cover the adapters that spend money; this
        catches a syntax error in anything at all, including modules whose
        dependencies are unavailable on the machine running the tests.
        """
        import pathlib

        import vlmeval

        root = pathlib.Path(vlmeval.__file__).parent
        checked = 0
        for path in sorted(root.rglob("*.py")):
            with self.subTest(module=str(path.relative_to(root))):
                compile(path.read_text(encoding="utf-8"), str(path), "exec")
            checked += 1
        self.assertGreater(checked, 10, "expected to find the whole package")


if __name__ == "__main__":
    unittest.main()
