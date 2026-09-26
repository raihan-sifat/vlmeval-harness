"""Answer extraction from free-form model output.

A large share of the effort in any VLM benchmark lives here. Models asked for
"just the letter" routinely answer ``"(C)"``, ``"C. a red triangle"``, ``"The
answer is C because ..."`` or ``"I cannot determine this."``; a harness that
does `if text == "C"` measures its own string handling rather than the model.

The parsers below are ordered from most to least explicit match, so that an
explicit marker always beats an incidental mention of the same letter elsewhere
in the text. When nothing matches they return ``None`` rather than guessing:
a forced guess inflates the score of verbose models and hides unparseable
outputs, so the runner counts those items and reports the rate separately.
"""

from __future__ import annotations

import re
import string
import unicodedata

_LETTERS = string.ascii_uppercase
_ARTICLES = {"a", "an", "the"}
_CONTRACTIONS = {
    "dont": "don't",
    "isnt": "isn't",
    "arent": "aren't",
    "wont": "won't",
    "cant": "can't",
    "its": "it's",
    "thereis": "there is",
}


def normalize_text(text: str) -> str:
    """Lowercase, strip accents and collapse whitespace.

    Unicode folding matters for short-answer tasks: "Café" and "cafe" must
    compare equal, and a stray non-breaking space from an extracted reference
    must not fail an otherwise correct answer.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("\u00a0", " ").lower().strip()
    return re.sub(r"\s+", " ", text)


def _letter_class(num_choices: int) -> str:
    """The set of letters a question with `num_choices` options may use."""
    n = max(0, min(int(num_choices), 26))
    return "".join(_LETTERS[:n])


def extract_choice(text: str, num_choices: int) -> str | None:
    """Extract an option letter from a model reply.

    Args:
        text: Raw model output.
        num_choices: How many options were offered, bounding the valid letters
            (``4`` -> A-D). Passing the true count prevents an answer like "E"
            from being credited when the question only had four options.

    Returns:
        An uppercase letter, or ``None`` if no confident answer was found.
    """
    letters = _letter_class(num_choices)
    if not text or not letters:
        return None

    raw = unicodedata.normalize("NFKC", text).strip()
    # Strip markdown emphasis so "**C**" and "`C`" parse like plain "C".
    flat = raw.replace("*", "").replace("_", "").replace("`", "").strip()

    # Tier 1: explicit answer markers, strongest first.
    anchored = (
        rf"answer\s*(?:is|:)\s*\(?([{letters}])\)?\b"
        rf"|correct\s*(?:answer|option|choice)?\s*(?:is|:)\s*\(?([{letters}])\)?\b"
        rf"|option\s*\(?([{letters}])\)?\s*(?:is|:)"
        rf"|choice\s*\(?([{letters}])\)?\s*(?:is|:)"
        rf"|^\s*\(?([{letters}])\)?\s*[.):\-]"
        rf"|^\s*\(?([{letters}])\)?\s*$"
    )
    match = re.search(anchored, flat, flags=re.IGNORECASE | re.MULTILINE)
    if match:
        return next(g for g in match.groups() if g).upper()

    # Tier 2: a parenthesised letter anywhere, e.g. "the answer (C) is red".
    match = re.search(rf"\(\s*([{letters}])\s*\)", flat)
    if match:
        return match.group(1).upper()

    # Tier 3: a bare standalone letter as the first token.
    match = re.match(rf"^\s*([{letters}])\b", flat)
    if match:
        return match.group(1).upper()

    # Tier 4: a letter introduced by a cue word, e.g. "could be A",
    # "option **B**", "I would pick C". The cue words are matched
    # case-insensitively but the letter must be upper-case, which is what keeps
    # the English article "a" in "there is a dog" from parsing as option A.
    match = re.search(
        rf"(?i:\b(?:be|is|answer|option|choice|choose|select|pick|goes\s+with))\s+\(?([{letters}])\)?\b",
        flat,
    )
    if match:
        return match.group(1).upper()

    # Tier 5: a bare option *number*, which many models use interchangeably with
    # letters. Anchored to a standalone token on purpose -- searching free prose
    # for a digit would read "there are 2 shapes" as an answer of B.
    match = re.search(
        rf"^\s*(?i:option|choice|answer)?\s*\(?([1-9])\)?\s*(?:[.):\-]|$)",
        flat,
    )
    if match:
        index = int(match.group(1)) - 1
        if 0 <= index < len(letters):
            return letters[index]

    return None


_YES = {"yes", "yeah", "yep", "y", "true", "correct", "affirmative", "1", "present"}
_NO = {"no", "nope", "n", "false", "incorrect", "negative", "0", "absent"}


def extract_yes_no(text: str) -> str | None:
    """Extract a yes/no verdict from a model reply.

    Negation is handled as its own tier because models frequently answer "There
    is no triangle in the image", which contains the queried object but answers
    *no*. Ordered scan (explicit verdict, then negation cues, then loose token
    membership) keeps that case correct without a full NLP dependency.
    """
    if not text:
        return None
    flat = normalize_text(text)

    # Tier 1: an explicit verdict, ideally at the start of the reply.
    verdict = re.search(
        r"^\s*(yes|no|y|n)\b"
        r"|\b(?:answer|verdict)\s*(?:is|:)\s*(yes|no)\b"
        r"|\b(yes|no)\s*[,.\u2013-]",
        flat,
    )
    if verdict:
        token = next(g for g in verdict.groups() if g)
        if token in _YES:
            return "yes"
        if token in _NO:
            return "no"

    # Tier 2: negation cues that invert an otherwise affirmative phrasing.
    negated = re.search(
        r"\b(?:does|do|did|is|are|was|were|has|have|can|could|will|would)\s*n(?:o|')?t\b"
        r"|\b(?:no|not|never|without)\b[^.]{0,40}?\b(?:there|present|visible|contain|shown|any)\b"
        r"|\b(?:cannot|can't|unable\s+to)\b",
        flat,
    )
    if negated:
        return "no"

    # Tier 3: loose membership. Order of appearance breaks ties, so a reply
    # that mentions "no" before "yes" is read as a denial.
    for token in re.findall(r"[a-z0-9']+", flat):
        if token in _YES:
            return "yes"
        if token in _NO:
            return "no"
    return None


def extract_number(text: str) -> float | None:
    """Extract the first number in a reply, tolerating thousands separators."""
    if not text:
        return None
    flat = normalize_text(text).replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", flat)
    return float(match.group(0)) if match else None


def extract_short_answer(text: str) -> str:
    """Reduce a reply to a short answer string for exact-match scoring.

    Takes the first non-empty line, strips common lead-ins and drops a leading
    option label such as "A) 12" when the label is not part of the answer.
    """
    if not text:
        return ""
    flat = unicodedata.normalize("NFKC", text).strip()
    first_line = next((ln.strip() for ln in flat.splitlines() if ln.strip()), "")
    first_line = re.sub(
        r"^(?:the\s+)?(?:final\s+)?(?:answer|response)\s*(?:is|:)\s*",
        "",
        first_line,
        flags=re.IGNORECASE,
    )
    first_line = re.sub(r"^\(?[A-Za-z]\)?[.)]\s+", "", first_line.strip())
    return first_line.strip().strip(".,;:")


def answers_match(pred: str, gold: str) -> bool:
    """Lenient-but-principled equality for short answers.

    Numeric answers compare by value, so "3.0" matches "3". Otherwise both sides
    are normalised, articles and filler words are dropped, and a token match is
    required. This is deliberately *not* fuzzy matching: a benchmark that rewards
    near-misses stops measuring the model and starts measuring the normaliser.
    """
    p, g = normalize_text(pred), normalize_text(gold)
    if not p or not g:
        return False
    if p == g:
        return True

    p_num, g_num = extract_number(p), extract_number(g)
    if p_num is not None and g_num is not None:
        return abs(p_num - g_num) < 1e-6

    p = _CONTRACTIONS.get(p, p)
    p_tokens = [t for t in re.findall(r"[a-z0-9.]+", p) if t not in _ARTICLES]
    g_tokens = [t for t in re.findall(r"[a-z0-9.]+", g) if t not in _ARTICLES]
    if p_tokens == g_tokens:
        return True
    # A single-token gold answer may appear inside a longer prediction, e.g.
    # gold "red" vs pred "a bright red circle".
    if len(g_tokens) == 1 and g_tokens[0] in p_tokens:
        return True
    return False
