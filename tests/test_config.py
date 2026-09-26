"""Configuration files: the YAML subset reader and preset precedence.

These paths are hard to exercise by hand and easy to get quietly wrong. A
preset that silently loses to a stale default, or a config typo that is ignored
until after a paid run, both look like working software right up until the
moment they cost something.
"""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from tests.helpers import *  # noqa: F401,F403 - path bootstrap

from vlmeval import cli
from vlmeval.config import load_config, resolve_preset
from vlmeval.yamlite import YamlSubsetError, loads


class TestYamlSubset(unittest.TestCase):
    """The reader must handle the shipped config shape and refuse the rest."""

    def test_scalars(self) -> None:
        data = loads(
            "a_string: hello\n"
            "an_int: 42\n"
            "a_float: 0.25\n"
            "a_true: true\n"
            "a_false: false\n"
            "a_null: null\n"
            "a_quoted: 'single'\n"
        )
        self.assertEqual(data["a_string"], "hello")
        self.assertEqual(data["an_int"], 42)
        self.assertEqual(data["a_float"], 0.25)
        self.assertIs(data["a_true"], True)
        self.assertIs(data["a_false"], False)
        self.assertIsNone(data["a_null"])
        self.assertEqual(data["a_quoted"], "single")

    def test_comments_are_removed_but_hashes_inside_values_survive(self) -> None:
        data = loads("# leading comment\nmodel: gpt-4o  # trailing\ntag: a#b\n")
        self.assertEqual(data["model"], "gpt-4o")
        self.assertEqual(data["tag"], "a#b")

    def test_nested_mappings(self) -> None:
        data = loads("defaults:\n  workers: 4\n  nested:\n    deep: true\n")
        self.assertEqual(data["defaults"]["workers"], 4)
        self.assertIs(data["defaults"]["nested"]["deep"], True)

    def test_sequence_of_mappings(self) -> None:
        """The shape every entry in configs/models.yaml uses."""
        data = loads(
            "models:\n"
            "  - name: first\n"
            "    spec:\n"
            "      backend: echo\n"
            "      model: strong\n"
            "    task: mcq\n"
            "  - name: second\n"
            "    spec:\n"
            "      backend: openai\n"
            "      model: gpt-4o\n"
            "    task: pope\n"
        )
        self.assertEqual(len(data["models"]), 2)
        self.assertEqual(data["models"][0]["name"], "first")
        self.assertEqual(data["models"][0]["spec"]["backend"], "echo")
        self.assertEqual(data["models"][0]["task"], "mcq")
        self.assertEqual(data["models"][1]["spec"]["model"], "gpt-4o")
        self.assertEqual(data["models"][1]["task"], "pope")

    def test_sequence_of_scalars(self) -> None:
        self.assertEqual(loads("items:\n  - a\n  - b\n")["items"], ["a", "b"])

    def test_colon_inside_quotes_is_not_a_key_separator(self) -> None:
        self.assertEqual(loads('url: "https://example.com/x"')["url"], "https://example.com/x")

    def test_empty_document(self) -> None:
        self.assertIsNone(loads("\n# only a comment\n"))

    def test_flow_collections_are_refused_not_guessed(self) -> None:
        for text in ("a: [1, 2]\n", "a: {b: 1}\n"):
            with self.subTest(text=text), self.assertRaises(YamlSubsetError):
                loads(text)

    def test_tabs_are_refused(self) -> None:
        with self.assertRaises(YamlSubsetError):
            loads("a:\n\tb: 1\n")

    def test_multiple_documents_are_refused(self) -> None:
        with self.assertRaises(YamlSubsetError):
            loads("a: 1\n---\nb: 2\n")

    def test_missing_colon_is_refused(self) -> None:
        with self.assertRaises(YamlSubsetError):
            loads("a: 1\njust_a_word\n")


CONFIG_TEXT = """\
defaults:
  workers: 4
  temperature: 0.0
  max_usd: 5.0
models:
  - name: echo-strong
    spec:
      backend: echo
      model: strong
    task: mcq
    source: synthetic_mcq
  - name: echo-hallucinator
    spec:
      backend: echo
      model: hallucinator
    task: pope
    source: synthetic_pope
    max_requests: 5
"""


class TestLoadConfig(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = Path(self.dir.name) / "models.yaml"
        self.path.write_text(CONFIG_TEXT, encoding="utf-8")

    def test_reads_defaults_and_presets(self) -> None:
        config = load_config(self.path)
        self.assertEqual(config["defaults"]["workers"], 4)
        self.assertEqual(sorted(config["presets"]), ["echo-hallucinator", "echo-strong"])

    def test_preset_merges_over_defaults(self) -> None:
        config = load_config(self.path)
        merged = resolve_preset(config, "echo-hallucinator")
        self.assertEqual(merged["workers"], 4)          # from defaults
        self.assertEqual(merged["max_usd"], 5.0)        # from defaults
        self.assertEqual(merged["max_requests"], 5)     # preset wins
        self.assertEqual(merged["spec"]["model"], "hallucinator")

    def test_unknown_preset_lists_the_real_ones(self) -> None:
        config = load_config(self.path)
        with self.assertRaises(ValueError) as ctx:
            resolve_preset(config, "nope")
        message = str(ctx.exception)
        self.assertIn("echo-strong", message)
        self.assertIn("nope", message)

    def test_missing_file_is_reported_clearly(self) -> None:
        with self.assertRaises(FileNotFoundError) as ctx:
            load_config(Path(self.dir.name) / "absent.yaml")
        self.assertIn("--config", str(ctx.exception))

    def test_config_without_models_is_refused(self) -> None:
        bad = Path(self.dir.name) / "bad.yaml"
        bad.write_text("defaults:\n  workers: 4\n", encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            load_config(bad)
        self.assertIn("models", str(ctx.exception))

    def test_preset_without_spec_is_refused(self) -> None:
        bad = Path(self.dir.name) / "nospec.yaml"
        bad.write_text("models:\n  - name: x\n    task: mcq\n", encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            load_config(bad)
        self.assertIn("spec", str(ctx.exception))

    def test_preset_without_name_is_refused(self) -> None:
        bad = Path(self.dir.name) / "noname.yaml"
        bad.write_text("models:\n  - spec:\n      backend: echo\n", encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            load_config(bad)
        self.assertIn("name", str(ctx.exception))

    def test_unknown_preset_key_is_refused(self) -> None:
        bad = Path(self.dir.name) / "typo.yaml"
        bad.write_text(
            "models:\n  - name: x\n    spec:\n      backend: echo\n    permutaions: 3\n",
            encoding="utf-8",
        )
        with self.assertRaises(ValueError) as ctx:
            load_config(bad)
        self.assertIn("permutaions", str(ctx.exception))


class TestPresetPrecedence(unittest.TestCase):
    """`--config` must fill gaps, never override what the user typed."""

    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = Path(self.dir.name) / "models.yaml"
        self.path.write_text(CONFIG_TEXT, encoding="utf-8")
        from vlmeval.data import synthetic

        self.data = Path(self.dir.name) / "data"
        synthetic.write_jsonl(synthetic.generate_mcq(self.data, n_items=3, seed=1), self.data / "mcq.jsonl")

    def _run(self, argv: list[str]) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cli.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_preset_supplies_model_task_and_source(self) -> None:
        code, out, _ = self._run(
            ["run", "--config", str(self.path), "--preset", "echo-strong",
             "--source", "dir", "--path", str(self.data), "--dry-run"]
        )
        self.assertEqual(code, 0)
        self.assertIn("echo:strong", out)

    def test_explicit_flag_beats_the_preset(self) -> None:
        """`--source`, `--path` and `--style` come from the flag, not the file."""
        code, out, _ = self._run(
            ["run", "--config", str(self.path), "--preset", "echo-strong",
             "--source", "dir", "--path", str(self.data), "--style", "grounded",
             "--dry-run"]
        )
        self.assertEqual(code, 0)
        self.assertIn("dir", out)
        self.assertIn(str(self.data), out)
        self.assertIn("grounded", out)

    def test_equals_syntax_is_also_recognised_as_explicit(self) -> None:
        code, out, _ = self._run(
            ["run", f"--config={self.path}", "--preset=echo-strong",
             "--source", "dir", f"--path={self.data}", "--style=grounded", "--dry-run"]
        )
        self.assertEqual(code, 0)
        self.assertIn("grounded", out)

    def test_unknown_preset_exits_non_zero_without_a_traceback(self) -> None:
        code, _, err = self._run(
            ["run", "--config", str(self.path), "--preset", "nope", "--dry-run"]
        )
        self.assertEqual(code, 1)
        self.assertIn("Configuration problem", err)
        self.assertNotIn("Traceback", err)

    def test_missing_config_exits_non_zero_without_a_traceback(self) -> None:
        code, _, err = self._run(
            ["run", "--config", str(Path(self.dir.name) / "absent.yaml"), "--dry-run"]
        )
        self.assertEqual(code, 1)
        self.assertNotIn("Traceback", err)

    def test_no_model_anywhere_names_the_fix(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            self._run(["run", "--dry-run"])
        self.assertIn("--model", str(ctx.exception))


class TestShippedConfigIsValid(unittest.TestCase):
    """The config we tell people to copy has to actually load."""

    def test_configs_models_yaml_parses(self) -> None:
        from vlmeval.config import repo_root

        path = repo_root() / "configs" / "models.yaml"
        if not path.is_file():
            self.skipTest("configs/models.yaml not present")
        config = load_config(path)
        self.assertIn("presets", config)
        self.assertTrue(config["presets"], "shipped config defines no presets")
        for name, entry in config["presets"].items():
            with self.subTest(preset=name):
                self.assertIn("backend", entry["spec"])
                self.assertIn("model", entry["spec"])


if __name__ == "__main__":
    unittest.main()
