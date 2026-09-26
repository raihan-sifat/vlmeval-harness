"""Registry: turning user-supplied strings and YAML into working objects.

Bad configuration must fail loudly and early. A model spec that silently falls
back to a default is how a "GPT-5" row on a leaderboard ends up being something
else entirely.
"""

from __future__ import annotations

import unittest

from tests.helpers import *  # noqa: F401,F403 - path bootstrap

from vlmeval.models.echo import ECHO_PRESETS, EchoModel
from vlmeval.registry import (
    TASK_ALIASES,
    available_models,
    available_tasks,
    build_model,
    build_task,
    sanitize,
)


class TestBuildModel(unittest.TestCase):
    def test_shorthand_string(self) -> None:
        """Regression: the shorthand path never set `backend`.

        It raised ``KeyError: 'backend'`` for every ``echo:demo``-style spec,
        so the documented default invocation could not run at all.
        """
        model = build_model("echo:demo")
        self.assertEqual(model.name, "demo")

    def test_shorthand_uses_the_model_id_as_the_name(self) -> None:
        self.assertEqual(build_model("echo:my-model").name, "my-model")

    def test_mapping_form(self) -> None:
        model = build_model({"backend": "echo", "model": "m", "skill": 0.25})
        self.assertEqual(model.name, "m")
        self.assertEqual(model.skill, 0.25)

    def test_mapping_rejects_missing_backend(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            build_model({"model": "m"})
        self.assertIn("backend", str(ctx.exception))

    def test_mapping_rejects_missing_model(self) -> None:
        with self.assertRaises(ValueError):
            build_model({"backend": "echo"})

    def test_unknown_backend_is_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            build_model("nosuchbackend:model")
        self.assertIn("nosuchbackend", str(ctx.exception))

    def test_bare_name_without_colon_is_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            build_model("gpt-5")
        self.assertIn("backend:model", str(ctx.exception))

    def test_non_string_non_mapping_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            build_model(42)  # type: ignore[arg-type]

    def test_name_is_sanitised_for_filenames(self) -> None:
        model = build_model("echo:vendor/model:v1")
        self.assertNotIn("/", model.name)
        self.assertNotIn(":", model.name)

    def test_available_models_lists_every_backend(self) -> None:
        listed = available_models()
        for expected in ("openai", "anthropic", "google", "echo", "hf_local"):
            self.assertIn(expected, listed)


class TestPriceOverrides(unittest.TestCase):
    """`cost_in` / `cost_out` must work for any backend, including local ones.

    A stale price table is a quiet way to make a cheap model look expensive, so
    the override is applied centrally and recorded in the model's provenance
    rather than being accepted and forgotten.
    """

    def test_override_replaces_both_rates(self) -> None:
        model = build_model({"backend": "echo", "model": "demo", "cost_in": 1.5, "cost_out": 7.5})
        self.assertEqual(model.cost_per_million, (1.5, 7.5))
        self.assertAlmostEqual(model.estimate_cost(1_000_000, 1_000_000), 9.0)

    def test_override_applies_to_one_rate_only(self) -> None:
        model = build_model({"backend": "echo", "model": "demo", "cost_out": 4.0})
        self.assertEqual(model.cost_per_million[1], 4.0)
        self.assertEqual(model.cost_per_million[0], EchoModel().cost_per_million[0])

    def test_override_is_recorded_for_provenance(self) -> None:
        model = build_model({"backend": "echo", "model": "demo", "cost_in": 2.0, "cost_out": 3.0})
        override = model.info["cost_override"]
        self.assertEqual(override["cost_in_per_mtok"], 2.0)
        self.assertEqual(override["cost_out_per_mtok"], 3.0)
        self.assertIn("list_in_per_mtok", override)

    def test_no_override_leaves_info_untouched(self) -> None:
        self.assertNotIn("cost_override", build_model("echo:demo").info)

    def test_override_names_are_not_passed_to_the_adapter(self) -> None:
        """`EchoModel` has no cost_in parameter; it must not receive one."""
        model = build_model({"backend": "echo", "model": "demo", "cost_in": 1.0})
        self.assertFalse(hasattr(model, "cost_in"))


class TestInlineJsonSpec(unittest.TestCase):
    """A mapping is the only way to pass adapter options from a shell."""

    def test_json_object_is_accepted(self) -> None:
        model = build_model('{"backend": "echo", "model": "demo", "skill": 0.25}')
        self.assertEqual(model.name, "demo")
        self.assertEqual(model.skill, 0.25)

    def test_malformed_json_is_refused_with_a_reason(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            build_model('{"backend": "echo",')
        self.assertIn("JSON", str(ctx.exception))

    def test_json_array_falls_through_to_the_shorthand_error(self) -> None:
        """Not a `{` prefix, so it is treated as a shorthand -- and says so."""
        with self.assertRaises(ValueError) as ctx:
            build_model("[1, 2]")
        self.assertIn("backend:model", str(ctx.exception))

    def test_json_presets_still_apply(self) -> None:
        self.assertEqual(build_model('{"backend": "echo", "model": "oracle"}').skill, 1.0)


class TestEchoPresets(unittest.TestCase):
    """The named echo presets must be genuinely different models.

    Regression: ``echo:demo``, ``echo:chatty`` and ``echo:bad`` all built
    ``EchoModel`` with stock defaults, so a benchmark over them reported three
    identical scores. Nothing in the harness was broken -- the benchmark simply
    had one model under three names, which reads exactly like a broken harness.
    """

    def test_presets_differ_from_each_other(self) -> None:
        knobs = {
            name: (
                build_model(f"echo:{name}").skill,
                build_model(f"echo:{name}").chatter,
                build_model(f"echo:{name}").refusal,
                build_model(f"echo:{name}").position_bias,
                build_model(f"echo:{name}").yes_bias,
            )
            for name in ECHO_PRESETS
        }
        self.assertGreater(
            len(set(knobs.values())),
            1,
            f"every echo preset resolved to the same behaviour: {knobs}",
        )

    def test_oracle_is_always_right(self) -> None:
        model = build_model("echo:oracle")
        self.assertEqual(model.skill, 1.0)
        self.assertEqual(model.refusal, 0.0)

    def test_bad_is_never_right(self) -> None:
        self.assertEqual(build_model("echo:bad").skill, 0.0)

    def test_blind_exploits_a_position(self) -> None:
        self.assertEqual(build_model("echo:blind").position_bias, 1.0)

    def test_hallucinator_affirms_every_binary_question(self) -> None:
        """POPE's subject is object hallucination, so the harness must be able to
        produce a model that exhibits it on demand."""
        model = build_model("echo:hallucinator")
        self.assertEqual(model.yes_bias, 1.0)
        for gold in ("yes", "no"):
            with self.subTest(gold=gold):
                response = model.generate_with_context(
                    image=None,
                    prompt="Is there a dog in the image? Answer yes or no.",
                    context={"gold": gold, "display_gold": gold, "uid": f"p-{gold}"},
                )
                self.assertEqual(response.text.strip().lower(), "yes")

    def test_yes_bias_does_not_affect_lettered_questions(self) -> None:
        """`yes_bias` is a binary-task failure mode; an MCQ has no yes/no answer."""
        model = build_model("echo:hallucinator")
        response = model.generate_with_context(
            image=None,
            prompt="What colour is it?\n\nA. red\nB. blue\n",
            context={
                "gold": "A",
                "display_gold": "A",
                "uid": "mcq-1",
                "num_choices": 2,
            },
        )
        self.assertIn(response.text.strip().upper(), {"A", "B"})

    def test_chatty_always_wraps_the_answer(self) -> None:
        self.assertEqual(build_model("echo:chatty").chatter, 1.0)

    def test_unknown_echo_name_keeps_defaults(self) -> None:
        """An unrecognised name is a free-form label, not an error."""
        model = build_model("echo:my-experiment")
        self.assertEqual(model.name, "my-experiment")
        self.assertEqual(model.skill, EchoModel().skill)

    def test_explicit_options_beat_the_preset(self) -> None:
        model = build_model({"backend": "echo", "model": "oracle", "skill": 0.5})
        self.assertEqual(model.skill, 0.5)

    def test_preset_lookup_is_case_insensitive(self) -> None:
        self.assertEqual(build_model("echo:ORACLE").skill, 1.0)

    def test_every_preset_builds_and_answers(self) -> None:
        for name in ECHO_PRESETS:
            with self.subTest(preset=name):
                model = build_model(f"echo:{name}")
                response = model.generate_with_context(
                    image=None,
                    prompt="What colour is the hexagon?\n\nA. red\nB. blue\n",
                    context={
                        "gold": "A",
                        "display_gold": "A",
                        "uid": "item-1",
                        "num_choices": 2,
                    },
                )
                self.assertTrue(response.ok)
                self.assertTrue(response.text)


class TestBuildTask(unittest.TestCase):
    def test_bare_name(self) -> None:
        self.assertEqual(build_task("mcq").name, "mcq")
        self.assertEqual(build_task("pope").name, "pope")

    def test_keyword_overrides(self) -> None:
        task = build_task("mcq", num_permutations=3)
        self.assertEqual(task.num_permutations, 3)

    def test_mapping_form(self) -> None:
        task = build_task({"task": "mcq", "num_permutations": 2})
        self.assertEqual(task.num_permutations, 2)

    def test_kwargs_beat_the_mapping(self) -> None:
        task = build_task({"task": "mcq", "num_permutations": 2}, num_permutations=5)
        self.assertEqual(task.num_permutations, 5)

    def test_aliases_resolve(self) -> None:
        for alias, canonical in TASK_ALIASES.items():
            with self.subTest(alias=alias):
                self.assertEqual(build_task(alias).name, build_task(canonical).name)

    def test_unknown_task_is_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            build_task("nosuchtask")
        self.assertIn("nosuchtask", str(ctx.exception))

    def test_available_tasks_is_sorted(self) -> None:
        tasks = available_tasks()
        self.assertEqual(tasks, sorted(tasks))
        self.assertIn("mcq", tasks)


class TestSanitize(unittest.TestCase):
    def test_strips_path_separators(self) -> None:
        for raw in ("a/b", "a\\b", "../../etc/passwd", "a:b"):
            with self.subTest(raw=raw):
                out = sanitize(raw)
                self.assertNotIn("/", out)
                self.assertNotIn("\\", out)
                self.assertNotIn(":", out)
                self.assertNotIn("..", out)

    def test_strips_spaces(self) -> None:
        self.assertNotIn(" ", sanitize("a b c"))

    def test_is_idempotent(self) -> None:
        once = sanitize("weird/name: v2")
        self.assertEqual(once, sanitize(once))

    def test_keeps_readable_characters(self) -> None:
        self.assertEqual(sanitize("gpt-5-mini"), "gpt-5-mini")


if __name__ == "__main__":
    unittest.main()
