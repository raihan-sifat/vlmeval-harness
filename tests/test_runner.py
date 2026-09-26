"""End-to-end runner behaviour.

The regressions here were all found by running the harness, not by reading it:
a run that scored nothing, a shared SQLite connection that corrupted under
threads, and an answer key that had no channel to the offline model.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from tests.helpers import FlakyModel, RecordingModel, mcq_example, pope_example  # noqa: F401

from vlmeval.registry import build_model, build_task
from vlmeval.runner import Runner
from vlmeval.runstate import Budget
from vlmeval.types import Example, ModelResponse


def _run(model, task, examples, **kwargs):
    kwargs.setdefault("workers", 4)
    kwargs.setdefault("max_retries", 0)
    kwargs.setdefault("sink_dir", None)
    return Runner(task=task, model=model, **kwargs).run(examples, "ds")


class TestRunnerProducesResults(unittest.TestCase):
    def test_scores_even_without_a_sink(self) -> None:
        """Regression: results were read back only from the JSONL sink.

        With ``sink_dir=None`` there is no sink, so every metric came back with
        ``n=0`` and the run silently reported nothing. The sink is a durability
        mechanism, not the source of truth for scoring.
        """
        examples = [mcq_example(f"e{i}") for i in range(5)]
        result = _run(RecordingModel("A"), build_task("mcq"), examples)
        self.assertEqual(result.counts["n_results"], 5)
        self.assertEqual(result.metrics["accuracy"].n, 5)
        self.assertEqual(len(result.items), 5)

    def test_works_with_a_sink_too(self) -> None:
        tmp = tempfile.mkdtemp()
        try:
            examples = [mcq_example(f"e{i}") for i in range(4)]
            result = _run(RecordingModel("A"), build_task("mcq"), examples, sink_dir=tmp)
            self.assertEqual(result.counts["n_results"], 4)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_permutation_variants_all_appear(self) -> None:
        examples = [mcq_example("e0")]
        result = _run(RecordingModel("A"), build_task("mcq", num_permutations=2), examples)
        self.assertEqual(result.counts["n_results"], 3)
        self.assertEqual({i.variant for i in result.items}, {"base", "perm1", "perm2"})

    def test_results_are_ordered_by_dataset_not_completion(self) -> None:
        """Metrics must not depend on which worker finished first."""
        examples = [mcq_example(f"e{i:02d}") for i in range(30)]
        task = build_task("mcq", num_permutations=2)
        first = _run(RecordingModel("A"), task, examples, workers=8)
        second = _run(RecordingModel("A"), task, examples, workers=1)
        self.assertEqual(
            [(i.uid, i.variant) for i in first.items],
            [(i.uid, i.variant) for i in second.items],
        )

    def test_binary_task_scores(self) -> None:
        examples = [pope_example(f"p{i}", "yes" if i % 2 else "no") for i in range(6)]
        result = _run(RecordingModel("yes"), build_task("pope"), examples)
        self.assertEqual(result.counts["n_results"], 6)
        self.assertAlmostEqual(result.metrics["accuracy"].value, 0.5)

    def test_empty_dataset_does_not_crash(self) -> None:
        result = _run(RecordingModel("A"), build_task("mcq"), [])
        self.assertEqual(result.counts["n_results"], 0)
        self.assertIsNone(result.metrics["accuracy"].value)


class TestNoGoldLeak(unittest.TestCase):
    def test_runner_never_leaks_gold_to_adapters(self) -> None:
        """An adapter that implements only `generate` must not see the answer.

        The runner passes the gold label as harness-internal context. The base
        ``generate_with_context`` discards it, so a provider-facing payload can
        never contain it unless an adapter explicitly opts in. The offline echo
        model is the only thing that does.
        """
        examples = [mcq_example("e0", gold="C")]
        model = RecordingModel("A")
        _run(model, build_task("mcq"), examples)

        self.assertEqual(model.calls, 1)
        # The provider-visible payload is the prompt alone.
        self.assertNotIn("echo-answer", model.prompts[0])
        for letter in ("C",):
            self.assertNotIn(
                f"answer is {letter}", model.prompts[0],
                "the prompt must not reveal the answer",
            )

    def test_context_is_available_to_adapters_that_opt_in(self) -> None:
        examples = [mcq_example("e0", gold="C")]
        model = RecordingModel("A")
        _run(model, build_task("mcq"), examples)
        # RecordingModel does override the hook, so it *can* see the context;
        # the guarantee is that `generate` still received only the prompt.
        self.assertTrue(model.contexts)
        self.assertEqual(model.contexts[0]["display_gold"], "C")

    def test_stored_prompts_contain_no_answer_marker(self) -> None:
        examples = [mcq_example("e0", gold="C")]
        result = _run(RecordingModel("A"), build_task("mcq", num_permutations=2), examples)
        for item in result.items:
            self.assertNotIn("echo-answer", item.prompt)


class TestRetriesAndBudgets(unittest.TestCase):
    def test_retries_a_retryable_error(self) -> None:
        examples = [mcq_example("e0")]
        model = FlakyModel(failures=2)
        result = _run(model, build_task("mcq"), examples, max_retries=3, backoff_base=0.0)
        self.assertEqual(model.calls, 3, "should have retried until it succeeded")
        self.assertEqual(result.counts["n_failed"], 0)

    def test_gives_up_and_records_the_failure(self) -> None:
        examples = [mcq_example("e0")]
        model = FlakyModel(failures=99)
        result = _run(model, build_task("mcq"), examples, max_retries=1, backoff_base=0.0)
        self.assertEqual(result.counts["n_failed"], 1)
        # Every item is unparsed, which scores zero rather than disappearing.
        self.assertEqual(result.metrics["accuracy"].value, 0.0)
        self.assertEqual(result.metrics["unparsed_rate"].value, 1.0)

    def test_budget_stops_the_run_and_is_reported(self) -> None:
        examples = [mcq_example(f"e{i}") for i in range(20)]
        result = _run(
            RecordingModel("A"), build_task("mcq"), examples,
            budget=Budget(max_requests=5), workers=1,
        )
        self.assertEqual(result.counts["n_results"], 5)
        self.assertIsNotNone(result.counts["stopped_reason"])

    def test_token_budget_stops_the_run(self) -> None:
        examples = [mcq_example(f"e{i}") for i in range(20)]
        result = _run(
            RecordingModel("A"), build_task("mcq"), examples,
            budget=Budget(max_tokens_total=10), workers=1,
        )
        self.assertLess(result.counts["n_results"], 20)


class TestResume(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, examples, **task_kwargs):
        task = build_task("mcq", **task_kwargs)
        model = build_model({"backend": "echo", "model": "m", "skill": 0.7,
                             "chatter": 0.0, "refusal": 0.0})
        return Runner(task=task, model=model, workers=4, max_retries=0,
                      cache=None, sink_dir=self.tmp, resume=True).run(examples, "ds")

    def test_identical_config_resumes_from_the_sink(self) -> None:
        examples = [mcq_example(f"e{i}") for i in range(5)]
        first = self._run(examples, num_permutations=1)
        self.assertEqual(first.counts["n_cached"], 0)
        second = self._run(examples, num_permutations=1)
        self.assertEqual(second.counts["n_cached"], first.counts["n_requests"])
        self.assertEqual(second.counts["n_results"], first.counts["n_results"])

    def test_changed_config_does_not_reuse_stale_rows(self) -> None:
        """Regression: the sink file was keyed only by dataset and model.

        Re-running with a different permutation count resumed onto rows computed
        under the old one and reported them as the new configuration's results.
        """
        examples = [mcq_example(f"e{i}") for i in range(5)]
        first = self._run(examples, num_permutations=1)
        second = self._run(examples, num_permutations=3)
        self.assertEqual(second.counts["n_cached"], 0, "must not reuse incompatible rows")
        self.assertEqual(second.counts["n_results"], 5 * 4)
        self.assertEqual(first.counts["n_requests"], 5 * 2)

    def test_resume_keeps_metrics_identical(self) -> None:
        examples = [mcq_example(f"e{i}") for i in range(8)]
        first = self._run(examples, num_permutations=1)
        second = self._run(examples, num_permutations=1)
        for name in ("accuracy", "consistency", "unparsed_rate"):
            with self.subTest(metric=name):
                self.assertEqual(first.metrics[name].value, second.metrics[name].value)

    def test_torn_final_line_does_not_break_resume(self) -> None:
        """A hard kill leaves a partial JSON line; resume must not choke."""
        examples = [mcq_example(f"e{i}") for i in range(4)]
        self._run(examples, num_permutations=1)
        raws = list(Path(self.tmp).rglob("*.jsonl"))
        self.assertEqual(len(raws), 1)
        with open(raws[0], "a", encoding="utf-8") as handle:
            handle.write('{"uid": "e0", "variant": "ba')
        result = self._run(examples, num_permutations=1)
        self.assertEqual(result.counts["n_results"], 4 * 2)

    def test_sink_rows_are_valid_jsonl(self) -> None:
        examples = [mcq_example(f"e{i}") for i in range(3)]
        self._run(examples, num_permutations=1)
        raw = next(Path(self.tmp).rglob("*.jsonl"))
        rows = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines() if line]
        self.assertEqual(len(rows), 6)
        for row in rows:
            self.assertIn("uid", row)
            self.assertIn("response", row)


class TestEchoModel(unittest.TestCase):
    def test_uses_the_spec_model_name(self) -> None:
        """Regression: `echo:demo` was reported as `echo-v1`."""
        self.assertEqual(build_model("echo:demo").name, "demo")

    def test_skill_controls_accuracy(self) -> None:
        examples = [mcq_example(f"e{i:02d}") for i in range(40)]
        good = _run(build_model({"backend": "echo", "model": "g", "skill": 1.0,
                                 "chatter": 0.0, "refusal": 0.0}),
                    build_task("mcq"), examples)
        bad = _run(build_model({"backend": "echo", "model": "b", "skill": 0.0,
                                "chatter": 0.0, "refusal": 0.0}),
                   build_task("mcq"), examples)
        self.assertEqual(good.metrics["accuracy"].value, 1.0)
        self.assertEqual(bad.metrics["accuracy"].value, 0.0)

    def test_is_consistent_across_permutations(self) -> None:
        """A stable model agrees with itself when options are reordered."""
        examples = [mcq_example(f"e{i:02d}") for i in range(30)]
        result = _run(
            build_model({"backend": "echo", "model": "c", "skill": 0.8,
                         "chatter": 0.0, "refusal": 0.0}),
            build_task("mcq", num_permutations=3), examples,
        )
        self.assertGreater(result.metrics["consistency"].value, 0.7)

    def test_position_prior_is_detected_by_the_bias_audit(self) -> None:
        """A model that always picks the first option must be flagged.

        With the gold answer deliberately *not* in the first slot, canonical
        accuracy collapses to chance and consistency collapses to nearly zero --
        the signature of reading position rather than content.
        """
        examples = [mcq_example(f"e{i:02d}", gold="C") for i in range(40)]
        result = _run(
            build_model({"backend": "echo", "model": "p", "skill": 0.0,
                         "position_bias": 1.0, "chatter": 0.0, "refusal": 0.0}),
            build_task("mcq", num_permutations=3), examples,
        )
        self.assertLessEqual(result.metrics["accuracy"].value, 0.35)
        self.assertLessEqual(result.metrics["consistency"].value, 0.1)

    def test_canonical_accuracy_alone_cannot_see_a_position_prior(self) -> None:
        """The trap that makes pooled variant accuracy worth reporting.

        When the gold option happens to sit in the first slot, a model that
        always answers "A" scores a perfect 100% on the canonical variant. The
        per-variant figures are the only thing that exposes it, because
        reordering moves the gold out of slot one.

        The pooled number does not fall to the 25% chance rate, and should not:
        the canonical variant is itself a free hit for a first-option model, so
        with 4 variants the expected value is (1 + 3/4) / 4 ~ 0.44. The
        meaningful assertion is that it collapses far below the canonical 1.0.
        """
        examples = [mcq_example(f"e{i:02d}", gold="A") for i in range(40)]
        result = _run(
            build_model({"backend": "echo", "model": "p", "skill": 0.0,
                         "position_bias": 1.0, "chatter": 0.0, "refusal": 0.0}),
            build_task("mcq", num_permutations=3), examples,
        )
        self.assertEqual(result.metrics["accuracy"].value, 1.0,
                         "canonical accuracy is fooled when gold sits first")
        pooled = result.metrics["variant_accuracy"].value
        self.assertLess(pooled, 0.55,
                        f"pooled variant accuracy must expose the prior, got {pooled}")
        self.assertLessEqual(result.metrics["consistency"].value, 0.1)

    def test_orders_vary_across_items(self) -> None:
        """Regression: every item received the same set of permutations.

        One global seed made the order audit correlated -- the gold answer sat
        in the same display slot for every item, so a position prior and a
        lucky gold slot became indistinguishable.
        """
        task = build_task("mcq", num_permutations=2)
        first = task.queries(mcq_example("e0"))[1].meta["order"]
        second = task.queries(mcq_example("e1"))[1].meta["order"]
        self.assertNotEqual(tuple(first), tuple(second))
        # Determinism is preserved for a given item.
        again = build_task("mcq", num_permutations=2).queries(mcq_example("e0"))[1].meta["order"]
        self.assertEqual(tuple(first), tuple(again))

    def test_refusals_show_up_as_unparsed(self) -> None:
        examples = [mcq_example(f"e{i:02d}") for i in range(40)]
        result = _run(
            build_model({"backend": "echo", "model": "r", "skill": 1.0,
                         "chatter": 0.0, "refusal": 1.0}),
            build_task("mcq"), examples,
        )
        self.assertEqual(result.metrics["unparsed_rate"].value, 1.0)
        self.assertEqual(result.metrics["accuracy"].value, 0.0)

    def test_chatter_still_parses(self) -> None:
        examples = [mcq_example(f"e{i:02d}") for i in range(40)]
        result = _run(
            build_model({"backend": "echo", "model": "t", "skill": 1.0,
                         "chatter": 1.0, "refusal": 0.0}),
            build_task("mcq"), examples,
        )
        self.assertEqual(result.metrics["unparsed_rate"].value, 0.0)
        self.assertEqual(result.metrics["accuracy"].value, 1.0)

    def test_deterministic_across_processes(self) -> None:
        examples = [mcq_example(f"e{i}") for i in range(10)]
        spec = {"backend": "echo", "model": "d", "skill": 0.6}
        a = _run(build_model(dict(spec)), build_task("mcq"), examples)
        b = _run(build_model(dict(spec)), build_task("mcq"), examples)
        self.assertEqual(
            [i.pred for i in a.items], [i.pred for i in b.items]
        )


if __name__ == "__main__":
    unittest.main()
