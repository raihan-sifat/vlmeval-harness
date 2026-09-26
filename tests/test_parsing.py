"""Answer extraction: the layer every score depends on.

A benchmark that parses its own outputs badly reports confidently wrong
numbers, so these cases are pinned rather than assumed.
"""

from __future__ import annotations

import unittest

from tests.helpers import *  # noqa: F401,F403 - path bootstrap

from vlmeval.parsing import extract_choice, extract_yes_no


class TestExtractChoice(unittest.TestCase):
    def test_bare_letter(self) -> None:
        self.assertEqual(extract_choice("A", 4), "A")
        self.assertEqual(extract_choice("b", 4), "B")

    def test_letter_in_prose(self) -> None:
        self.assertEqual(extract_choice("The answer is C.", 4), "C")
        self.assertEqual(
            extract_choice("Looking carefully, I would say option **B** here.", 4), "B"
        )

    def test_parenthesised_and_dotted_letters(self) -> None:
        self.assertEqual(extract_choice("(D)", 4), "D")
        self.assertEqual(extract_choice("B. blue", 4), "B")

    def test_explicit_answer_marker(self) -> None:
        self.assertEqual(extract_choice("Answer: D", 4), "D")
        self.assertEqual(extract_choice("Final answer is (A).", 4), "A")

    def test_bare_number_maps_to_index(self) -> None:
        self.assertEqual(extract_choice("2", 4), "B")
        self.assertEqual(extract_choice("Option 3", 4), "C")

    def test_rejects_out_of_range_letter(self) -> None:
        # 5 choices means A-E; "F" cannot be an answer and must not be read as one.
        self.assertIsNone(extract_choice("F", 5))
        self.assertEqual(extract_choice("E", 5), "E")

    def test_rejects_unparseable(self) -> None:
        for text in ("", "   ", "I don't know", "maybe the blue one?"):
            with self.subTest(text=text):
                self.assertIsNone(extract_choice(text, 4))

    def test_refusal_is_unparsed(self) -> None:
        self.assertIsNone(
            extract_choice("I'm sorry, I cannot determine the answer from this image.", 4)
        )

    def test_letter_inside_a_word_is_not_an_answer(self) -> None:
        # "Based" contains 'a'/'b'; word boundaries keep this from parsing.
        self.assertIsNone(extract_choice("Based on nothing in particular", 4))

    def test_first_valid_letter_wins(self) -> None:
        self.assertEqual(extract_choice("Could be A, but maybe C.", 4), "A")


class TestExtractYesNo(unittest.TestCase):
    def test_bare(self) -> None:
        self.assertEqual(extract_yes_no("yes"), "yes")
        self.assertEqual(extract_yes_no("No."), "no")

    def test_in_prose(self) -> None:
        self.assertEqual(extract_yes_no("Yes, there is a dog."), "yes")
        self.assertEqual(extract_yes_no("There is no cat present."), "no")

    def test_negation_wins_over_keyword_order(self) -> None:
        # "no" appears before "car" but the sentence denies existence.
        self.assertEqual(extract_yes_no("No, there is not a car in the image."), "no")
        self.assertEqual(extract_yes_no("There is not a dog."), "no")

    def test_rejects_other_content(self) -> None:
        for text in ("", "maybe", "unclear", "123"):
            with self.subTest(text=text):
                self.assertIsNone(extract_yes_no(text))

    def test_refusal_is_unparsed(self) -> None:
        self.assertIsNone(extract_yes_no("I'm not able to answer that."))


if __name__ == "__main__":
    unittest.main()
