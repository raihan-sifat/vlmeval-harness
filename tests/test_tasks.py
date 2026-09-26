"""Task construction: prompting, permutations, gold handling, fingerprints.

Every test here corresponds to a defect found while running the harness, so
they double as the specification for what went wrong.
"""

from __future__ import annotations

import unittest

from tests.helpers import *  # noqa: F401,F403 - path bootstrap

from vlmeval.prompts import LETTERS
from vlmeval.registry import build_task
from vlmeval.tasks.pope import POPETask
from vlmeval.types import Example


class TestMCQQueries(unittest.TestCase):
    def setUp(self) -> None:
        self.example = Example(
            uid="e0",
            question="Which shape is largest?",
            choices=["circle", "square", "triangle", "star"],
            label="B",
            image="i.png",
        )

    def test_gold_is_canonical_not_display(self) -> None:
        """`Query.gold` must live in the coordinate system `decode` produces.

        Otherwise a permuted variant is scored against the wrong key whenever
        the correct option moved, and the item is silently marked incorrect
        even when the model answered perfectly.
        """
        task = build_task("mcq", num_permutations=3)
        for query in task.queries(self.example):
            with self.subTest(variant=query.variant):
                self.assertEqual(query.gold, "B", "gold must be the canonical letter")
                # The displayed letter is different, and decode maps back to B.
                display = query.meta["display_gold"]
                self.assertEqual(query.apply(display), "B")

    def test_decode_round_trips_every_variant(self) -> None:
        task = build_task("mcq", num_permutations=3)
        for query in task.queries(self.example):
            for slot, canonical in enumerate(query.meta["order"]):
                with self.subTest(variant=query.variant, slot=slot):
                    shown = LETTERS[slot]
                    self.assertEqual(query.apply(shown), LETTERS[canonical])

    def test_permutations_are_distinct_and_include_canonical(self) -> None:
        task = build_task("mcq", num_permutations=5)
        orders = [tuple(q.meta["order"]) for q in task.queries(self.example)]
        self.assertEqual(orders[0], tuple(range(4)), "first variant must be canonical")
        self.assertEqual(len(set(orders)), len(orders), "permutations must be distinct")

    def test_num_permutations_counts_extra_variants(self) -> None:
        task = build_task("mcq", num_permutations=2)
        self.assertEqual(len(task.queries(self.example)), 3)
        self.assertEqual(len(build_task("mcq").queries(self.example)), 1)

    def test_deduplicates_when_fewer_orders_exist(self) -> None:
        # 3 options give only 6 orders, so asking for 10 must not repeat.
        example = Example(uid="e", question="q?", choices=["a", "b", "c"], label="A", image="i")
        orders = [tuple(q.meta["order"]) for q in build_task("mcq", num_permutations=10).queries(example)]
        self.assertEqual(len(set(orders)), len(orders))
        self.assertLessEqual(len(orders), 6)

    def test_parses_only_valid_letters(self) -> None:
        task = build_task("mcq")
        query = task.queries(self.example)[0]
        self.assertEqual(task.parse("The answer is C.", query), "C")
        self.assertIsNone(task.parse("no idea", query))

    def test_rejects_examples_without_choices(self) -> None:
        bare = Example(uid="e", question="q?", label="yes", image="i")
        with self.assertRaises(ValueError):
            build_task("mcq").queries(bare)

    def test_gold_may_be_written_as_option_text(self) -> None:
        example = Example(
            uid="e", question="q?", choices=["circle", "square"], label="square", image="i"
        )
        query = build_task("mcq").queries(example)[0]
        self.assertEqual(query.gold, "B")

    def test_missing_gold_yields_none_not_a_guess(self) -> None:
        example = Example(uid="e", question="q?", choices=["a", "b"], label="", image="i")
        query = build_task("mcq").queries(example)[0]
        self.assertIsNone(query.gold)

    def test_prompt_never_contains_the_gold_letter_marker(self) -> None:
        """No answer key may reach a stored prompt."""
        for query in build_task("mcq", num_permutations=2).queries(self.example):
            self.assertNotIn("echo-answer", query.prompt)
            self.assertNotIn("<!", query.prompt)


class TestMCQScoring(unittest.TestCase):
    def test_unparsed_counts_against_accuracy(self) -> None:
        """Abstention must not raise the score.

        Excluding unparsed items from the denominator means a model answering
        half the items correctly and refusing the rest scores 100%.
        """
        from vlmeval.metrics.mcq import accuracy_metric

        strict = accuracy_metric([True, True, None, None], name="accuracy")
        self.assertEqual(strict.n, 4, "unparsed items must stay in the denominator")
        self.assertAlmostEqual(strict.value, 0.5)

    def test_accuracy_and_parsed_accuracy_are_consistent(self) -> None:
        from vlmeval.metrics.mcq import accuracy_metric, parsed_accuracy_metric

        flags = [True, False, True, None, None, True]
        strict = accuracy_metric(flags, name="accuracy")
        parsed = parsed_accuracy_metric(flags, name="accuracy_parsed")
        self.assertEqual((strict.n, parsed.n), (6, 4))
        self.assertAlmostEqual(strict.value, parsed.value * parsed.n / strict.n)

    def test_all_unparsed_is_zero_not_none(self) -> None:
        from vlmeval.metrics.mcq import accuracy_metric

        self.assertEqual(accuracy_metric([None, None], name="accuracy").value, 0.0)


class TestPOPETask(unittest.TestCase):
    def test_normalizes_truthy_spellings_to_yes(self) -> None:
        for label in ("yes", "Yes", "YES", "true", "1", "present", " y "):
            with self.subTest(label=label):
                self.assertEqual(POPETask._normalize(label), "yes")

    def test_normalizes_falsy_spellings_to_no(self) -> None:
        for label in ("no", "No", "false", "0", "absent"):
            with self.subTest(label=label):
                self.assertEqual(POPETask._normalize(label), "no")

    def test_normalize_never_inverts_the_label(self) -> None:
        """Regression: a table of negations made "1" normalise to "0"."""
        for label in ("1", "true", "yes", "present"):
            self.assertNotEqual(POPETask._normalize(label), "no")
        for label in ("0", "false", "no", "absent"):
            self.assertNotEqual(POPETask._normalize(label), "yes")

    def test_unknown_label_is_none(self) -> None:
        self.assertIsNone(POPETask._normalize("maybe"))
        self.assertIsNone(POPETask._normalize(None))

    def test_rejects_examples_with_choices(self) -> None:
        bad = Example(uid="p", question="q?", choices=["a"], label="yes", image="i")
        with self.assertRaises(ValueError):
            POPETask().queries(bad)

    def test_by_category_keeps_every_item(self) -> None:
        """Grouping must partition the items, not drop or duplicate any."""
        from vlmeval.tasks.base import ItemOutcome
        from vlmeval.types import ItemResult, ModelResponse

        outcomes = []
        for i, cat in enumerate(["random", "popular", "adversarial"] * 4):
            uid = f"p{i}"
            result = ItemResult(
                uid=uid, task="pope", model="m", category=cat,
                gold="yes", pred="yes", correct=True, prompt="q", variant="base",
                response=ModelResponse(text="yes"),
            )
            outcomes.append(ItemOutcome(result=result, decoded={"base": "yes"}))
        groups = POPETask().by_category(outcomes)
        self.assertEqual(set(groups), {"random", "popular", "adversarial"})
        self.assertEqual(sum(groups[c]["accuracy"].n for c in groups), len(outcomes))

    def test_unlabelled_items_land_under_all(self) -> None:
        from vlmeval.tasks.base import ItemOutcome
        from vlmeval.types import ItemResult, ModelResponse

        result = ItemResult(
            uid="p0", task="pope", model="m", category="", gold="no",
            pred="no", correct=True, prompt="q", variant="base",
            response=ModelResponse(text="no"),
        )
        groups = POPETask().by_category([ItemOutcome(result=result, decoded={"base": "no"})])
        self.assertEqual(set(groups), {"all"})


class TestFingerprint(unittest.TestCase):
    def test_stable_for_equal_config(self) -> None:
        self.assertEqual(build_task("mcq", num_permutations=2).fingerprint(),
                         build_task("mcq", num_permutations=2).fingerprint())

    def test_changes_with_permutations(self) -> None:
        """Regression: the fingerprint ignored num_permutations entirely.

        Because MCQTask takes that knob as a named parameter it never reached
        ``self.options``, so ``describe()`` omitted it -- meaning a run could
        change its permutation count and still resume onto incompatible rows.
        """
        self.assertNotEqual(build_task("mcq", num_permutations=1).fingerprint(),
                            build_task("mcq", num_permutations=3).fingerprint())

    def test_changes_with_seed_and_template(self) -> None:
        self.assertNotEqual(build_task("mcq", seed=1).fingerprint(),
                            build_task("mcq", seed=2).fingerprint())
        self.assertNotEqual(build_task("mcq", style="plain").fingerprint(),
                            build_task("mcq", style="grounded").fingerprint())

    def test_differs_across_tasks(self) -> None:
        self.assertNotEqual(build_task("mcq").fingerprint(), build_task("pope").fingerprint())

    def test_records_its_own_knobs_in_the_manifest(self) -> None:
        options = build_task("mcq", num_permutations=4, seed=7).describe()["options"]
        self.assertEqual(options["num_permutations"], 4)
        self.assertEqual(options["seed"], 7)


if __name__ == "__main__":
    unittest.main()
