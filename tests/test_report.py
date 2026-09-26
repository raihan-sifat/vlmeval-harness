"""Report artifacts: what gets written, and what can be read back.

The split between `metrics.json` and `results.jsonl` is a deliberate trade --
a summary you can parse cheaply, and item detail you only open when you need
it -- so it is pinned here rather than left to drift.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests.helpers import *  # noqa: F401,F403 - path bootstrap

from vlmeval.data import synthetic
from vlmeval.report import load_run, render_markdown, write_run
from vlmeval.runner import Runner
from vlmeval.registry import build_model, build_task


def _run_one_run(tmp: Path, model_spec: str = "echo:demo") -> object:
    """Evaluate a handful of items with the offline backend."""
    data_dir = tmp / "data"
    examples = synthetic.generate_mcq(data_dir, n_items=4, seed=5)
    task = build_task("mcq")
    model = build_model(model_spec)
    runner = Runner(model, task, workers=2, sink_dir=tmp / "raw")
    return runner.run(examples, dataset="unit")


class TestWriteRun(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.tmp = Path(self.dir.name)
        self.result = _run_one_run(self.tmp)

    def test_writes_every_artifact(self) -> None:
        run_dir = write_run(self.result, out_dir=self.tmp / "out")
        for name in ("metrics.json", "summary.md", "summary.csv", "results.jsonl",
                     "manifest.json"):
            with self.subTest(artifact=name):
                self.assertTrue((run_dir / name).is_file(), f"{name} was not written")

    def test_metrics_json_omits_items_and_stays_small(self) -> None:
        """The regression this guards: a 750 KB metrics file for a 260-item run."""
        run_dir = write_run(self.result, out_dir=self.tmp / "out")
        data = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        self.assertNotIn("items", data)
        self.assertIn("metrics", data)
        self.assertIn("by_category", data)
        self.assertLess(
            (run_dir / "metrics.json").stat().st_size,
            64_000,
            "metrics.json grew large; keep item detail in results.jsonl",
        )

    def test_results_jsonl_holds_one_record_per_item(self) -> None:
        run_dir = write_run(self.result, out_dir=self.tmp / "out")
        lines = [
            line
            for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(lines), len(self.result.items))
        first = json.loads(lines[0])
        for key in ("uid", "gold", "pred", "correct", "prompt", "response"):
            self.assertIn(key, first)

    def test_items_can_be_included_when_asked(self) -> None:
        run_dir = write_run(self.result, out_dir=self.tmp / "out2", include_items=True)
        data = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["items"]), len(self.result.items))

    def test_round_trips_through_load_run(self) -> None:
        run_dir = write_run(self.result, out_dir=self.tmp / "out3")
        loaded = load_run(run_dir)
        self.assertEqual(loaded.run_id, self.result.run_id)
        self.assertEqual(loaded.task, self.result.task)
        self.assertEqual(loaded.model, self.result.model)
        self.assertEqual(loaded.items, self.result.items)
        self.assertEqual(
            loaded.metrics["accuracy"].value, self.result.metrics["accuracy"].value
        )

    def test_load_run_without_jsonl_still_returns_metrics(self) -> None:
        """A metrics-only directory is summarised, not rejected."""
        run_dir = write_run(self.result, out_dir=self.tmp / "out4")
        (run_dir / "results.jsonl").unlink()
        loaded = load_run(run_dir)
        self.assertEqual(loaded.items, [])
        self.assertIn("accuracy", loaded.metrics)

    def test_load_run_skips_a_corrupt_jsonl_line(self) -> None:
        run_dir = write_run(self.result, out_dir=self.tmp / "out5")
        path = run_dir / "results.jsonl"
        path.write_text(
            path.read_text(encoding="utf-8") + "{not json}\n", encoding="utf-8"
        )
        loaded = load_run(run_dir)
        self.assertEqual(len(loaded.items), len(self.result.items))

    def test_summary_markdown_includes_the_headline(self) -> None:
        text = render_markdown(self.result)
        self.assertIn(self.result.model, text)
        self.assertIn("accuracy", text.lower())

    def test_run_id_is_sanitised_into_the_directory(self) -> None:
        result = self.result
        result.run_id = "../escape"
        run_dir = write_run(result, out_dir=self.tmp / "out6")
        self.assertEqual(run_dir.parent, self.tmp / "out6")
        self.assertNotIn("..", str(run_dir))


class TestEchoCalibration(unittest.TestCase):
    """`echo:oracle` and `echo:bad` are the harness's own regression fixtures.

    They are calibrated to exactly 1.0 and exactly 0.0, so a change that breaks
    parsing, scoring, or option decoding moves them immediately. This is the
    cheapest possible detector for "the harness is lying about the score".
    """

    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.tmp = Path(self.dir.name)

    def _accuracy(self, spec: str) -> float:
        result = _run_one_run(self.tmp, spec)
        return result.metrics["accuracy"].value

    def test_oracle_scores_exactly_one(self) -> None:
        self.assertEqual(self._accuracy("echo:oracle"), 1.0)

    def test_bad_scores_exactly_zero(self) -> None:
        self.assertEqual(self._accuracy("echo:bad"), 0.0)

    def test_permutation_keeps_a_consistent_model_consistent(self) -> None:
        """Option order must not change a model's mind.

        If permutations leaked the canonical label, or decoding were asymmetric,
        `oracle` would stop being perfect as soon as the options moved.
        """
        data_dir = self.tmp / "perm"
        examples = synthetic.generate_mcq(data_dir, n_items=6, seed=11)
        task = build_task("mcq", num_permutations=3, seed=99)
        runner = Runner(build_model("echo:oracle"), task, workers=2,
                        sink_dir=self.tmp / "raw2")
        result = runner.run(examples, dataset="unit")
        self.assertEqual(result.metrics["accuracy"].value, 1.0)
        self.assertEqual(result.metrics["consistency"].value, 1.0)


if __name__ == "__main__":
    unittest.main()
