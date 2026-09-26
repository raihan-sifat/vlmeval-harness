"""Statistics: intervals and paired significance tests.

A leaderboard that ranks models on a point estimate with no interval invites
over-reading a 0.4-point gap as a real difference. These tests pin the
properties the report relies on.
"""

from __future__ import annotations

import unittest

from tests.helpers import *  # noqa: F401,F403 - path bootstrap

from vlmeval.stats import (
    bootstrap_ci,
    holm_bonferroni,
    mcnemar_test,
    paired_bootstrap_delta,
    percentile,
)


class TestBootstrapCI(unittest.TestCase):
    def test_brackets_the_mean(self) -> None:
        values = [1.0] * 60 + [0.0] * 40
        low, high = bootstrap_ci(values, n_boot=500, seed=1)
        self.assertLessEqual(low, 0.6)
        self.assertGreaterEqual(high, 0.6)
        self.assertLess(low, high)

    def test_is_deterministic_for_a_fixed_seed(self) -> None:
        values = [1.0, 0.0] * 50
        self.assertEqual(
            bootstrap_ci(values, n_boot=300, seed=7),
            bootstrap_ci(values, n_boot=300, seed=7),
        )

    def test_degenerate_input(self) -> None:
        # Empty reports "no interval", not a zero-width one: claiming the latter
        # would read as a precise estimate of nothing.
        self.assertEqual(bootstrap_ci([], n_boot=10), (None, None))
        low, high = bootstrap_ci([1.0] * 20, n_boot=100, seed=1)
        self.assertAlmostEqual(low, 1.0)
        self.assertAlmostEqual(high, 1.0)

    def test_narrower_for_more_data(self) -> None:
        small = [1.0] * 30 + [0.0] * 30
        large = [1.0] * 300 + [0.0] * 300
        s_low, s_high = bootstrap_ci(small, n_boot=400, seed=3)
        l_low, l_high = bootstrap_ci(large, n_boot=400, seed=3)
        self.assertLess((l_high - l_low), (s_high - s_low))


class TestMcnemar(unittest.TestCase):
    def test_identical_models_are_not_significant(self) -> None:
        a = [True] * 40 + [False] * 10
        out = mcnemar_test(a, list(a))
        self.assertEqual(out["p_value"], 1.0)
        self.assertFalse(out["significant"])
        self.assertAlmostEqual(out["delta"], 0.0)

    def test_clear_difference_is_significant(self) -> None:
        a = [True] * 45 + [False] * 5
        b = [False] * 45 + [True] * 5
        out = mcnemar_test(a, b)
        self.assertTrue(out["significant"])
        self.assertLess(out["p_value"], 0.05)
        self.assertEqual(out["only_a"], 45)
        self.assertEqual(out["only_b"], 5)
        self.assertAlmostEqual(out["delta"], 0.8)

    def test_few_discordant_items_cannot_reach_significance(self) -> None:
        """A gap carried by five items is not evidence.

        Exactly the case where an unpaired test would mislead: the pooled
        accuracies differ, but five discordant pairs cannot reject the null.
        """
        a = [True] * 45 + [False] * 5
        b = [True] * 40 + [False] * 10
        out = mcnemar_test(a, b)
        self.assertEqual(out["only_a"], 5)
        self.assertEqual(out["only_b"], 0)
        self.assertAlmostEqual(out["delta"], 0.1)
        self.assertFalse(out["significant"], "p should be 2 * 0.5^5 = 0.0625")

    def test_counts_every_item(self) -> None:
        a = [True, True, False, False]
        b = [True, False, True, False]
        out = mcnemar_test(a, b)
        self.assertEqual(
            out["both_correct"] + out["only_a"] + out["only_b"] + out["neither"],
            len(a),
        )
        self.assertEqual(out["n_paired"], 4)

    def test_agreement_produces_no_discordant_pairs(self) -> None:
        same = [True] * 30 + [False] * 30
        out = mcnemar_test(same, same)
        self.assertEqual(out["only_a"] + out["only_b"], 0)
        self.assertEqual(out["p_value"], 1.0)

    def test_mismatched_lengths_raise(self) -> None:
        with self.assertRaises(ValueError):
            mcnemar_test([True, False], [True])

    def test_p_value_is_a_probability(self) -> None:
        a = [True] * 30 + [False] * 20
        b = [False] * 20 + [True] * 30
        self.assertGreaterEqual(mcnemar_test(a, b)["p_value"], 0.0)
        self.assertLessEqual(mcnemar_test(a, b)["p_value"], 1.0)


class TestPairedBootstrapDelta(unittest.TestCase):
    def test_no_difference_when_models_match(self) -> None:
        a = [1.0] * 50 + [0.0] * 50
        out = paired_bootstrap_delta(a, list(a), n_boot=400, seed=5)
        self.assertAlmostEqual(out["delta"], 0.0)
        self.assertLessEqual(out["ci_low"], 0.0)
        self.assertGreaterEqual(out["ci_high"], 0.0)

    def test_detects_a_real_gap(self) -> None:
        a = [1.0] * 80 + [0.0] * 20
        b = [1.0] * 20 + [0.0] * 80
        out = paired_bootstrap_delta(a, b, n_boot=400, seed=5)
        self.assertGreater(out["delta"], 0.0)
        self.assertGreater(out["ci_low"], 0.0, "interval should exclude zero")

    def test_is_deterministic(self) -> None:
        a = [1.0, 0.0] * 40
        b = [True] * 30 + [False] * 50
        self.assertEqual(
            paired_bootstrap_delta(a, b, n_boot=200, seed=2),
            paired_bootstrap_delta(a, b, n_boot=200, seed=2),
        )

    def test_unparsed_items_are_excluded_from_both_sides(self) -> None:
        out = paired_bootstrap_delta([True, None, True], [True, True, True], n_boot=100, seed=1)
        self.assertEqual(out["n"], 2)

    def test_no_usable_pairs(self) -> None:
        out = paired_bootstrap_delta([None, None], [None, None], n_boot=50)
        self.assertEqual(out["n"], 0)
        self.assertIsNone(out["delta"])


class TestHolmBonferroni(unittest.TestCase):
    def test_fewer_significant_after_correction(self) -> None:
        p_values = {"a": 0.001, "b": 0.01, "c": 0.02, "d": 0.04}
        out = holm_bonferroni(p_values)
        self.assertEqual(out["n_comparisons"], 4)
        for name, adjusted in out["adjusted"].items():
            with self.subTest(comparison=name):
                self.assertGreaterEqual(adjusted, p_values[name])

    def test_adjusted_is_monotone_in_raw(self) -> None:
        p_values = {"a": 0.001, "b": 0.02, "c": 0.03, "d": 0.04, "e": 0.045}
        out = holm_bonferroni(p_values)
        ranked = [out["adjusted"][k] for k in sorted(p_values, key=p_values.get)]
        self.assertEqual(ranked, sorted(ranked))

    def test_significance_can_only_be_lost_not_gained(self) -> None:
        raw = {"a": 0.001, "b": 0.02, "c": 0.03}
        out = holm_bonferroni(raw)
        for name, adjusted in out["adjusted"].items():
            with self.subTest(comparison=name):
                self.assertEqual(out["significant"][name], adjusted < 0.05)

    def test_empty_input(self) -> None:
        out = holm_bonferroni({})
        self.assertEqual(out["adjusted"], {})
        self.assertEqual(out["n_comparisons"], 0)

    def test_strong_result_survives_correction(self) -> None:
        out = holm_bonferroni({f"m{i}": 1e-9 for i in range(10)})
        self.assertTrue(any(out["significant"].values()))

    def test_correction_never_exceeds_one(self) -> None:
        out = holm_bonferroni({"a": 0.9, "b": 0.95})
        for adjusted in out["adjusted"].values():
            self.assertLessEqual(adjusted, 1.0)


class TestPercentile(unittest.TestCase):
    def test_endpoints(self) -> None:
        data = [1.0, 2.0, 3.0, 4.0]
        self.assertAlmostEqual(percentile(data, 0.0), 1.0)
        self.assertAlmostEqual(percentile(data, 1.0), 4.0)
        self.assertAlmostEqual(percentile(data, 0.5), 2.5)

    def test_out_of_range_is_clamped_not_fatal(self) -> None:
        data = [1.0, 2.0, 3.0, 4.0]
        self.assertAlmostEqual(percentile(data, 100.0), 4.0)
        self.assertAlmostEqual(percentile(data, -5.0), 1.0)

    def test_empty_is_none_so_json_stays_valid(self) -> None:
        """NaN would serialise as the bare token ``NaN`` and break JSON.parse."""
        self.assertIsNone(percentile([], 0.5))

    def test_single_value(self) -> None:
        self.assertAlmostEqual(percentile([7.0], 0.5), 7.0)


if __name__ == "__main__":
    unittest.main()
