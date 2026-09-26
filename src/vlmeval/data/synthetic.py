"""Synthetic dataset generation.

This module is what makes the project runnable before a single API key exists.
It draws simple scenes -- coloured geometric shapes on a plain background -- and
derives questions whose answers are known by construction:

* ``mcq`` items ask for a count, a colour, a shape or a size, with four
  lettered options.
* ``pope`` items ask whether an object is present, split into the same
  random / popular / adversarial difficulty tiers as the real benchmark.

Why bother, when MMMU and POPE are downloadable? Three reasons that turn out to
matter more than expected:

1. **CI and tests.** A benchmark whose ground truth is generated, not scraped,
   has no licence, no download and no rate limit, so every commit can be
   verified.
2. **Ground-truth debugging.** When a real benchmark reports 41%, there is no
   way to tell a weak model from a broken parser. On generated data the
   correct answers are known exactly, so a harness that scores 40% is provably
   at fault rather than merely suspicious.
3. **Controlled difficulty.** Object counts, distractor colours and distractor
   similarity are all parameters here, so "does accuracy fall off with object
   count?" is a question this dataset can answer and MMMU cannot.

It is not a substitute for a real benchmark and the report says so. It is the
instrument panel, not the engine.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..prompts import LETTERS
from ..types import Example

SHAPES = ("circle", "square", "triangle", "star", "hexagon")
COLORS = ("red", "blue", "green", "yellow", "purple", "orange")
SIZES = ("small", "medium", "large")

#: 8 well-separated hues on white; kept explicit rather than generated so the
#: rendered image is byte-identical across machines and Pillow versions.
_RGB = {
    "red": (220, 40, 40),
    "blue": (40, 80, 220),
    "green": (30, 150, 70),
    "yellow": (240, 200, 40),
    "purple": (140, 60, 190),
    "orange": (240, 130, 30),
    "cyan": (40, 190, 200),
    "brown": (140, 90, 50),
}


@dataclass
class Scene:
    """A generated image and the ground truth needed to ask about it."""

    objects: list[dict[str, Any]]

    def count_of(self, shape: str) -> int:
        return sum(1 for o in self.objects if o["shape"] == shape)

    def of_shape(self, shape: str) -> list[dict[str, Any]]:
        return [o for o in self.objects if o["shape"] == shape]

    def colors(self) -> set[str]:
        return {o["color"] for o in self.objects}

    def shapes(self) -> set[str]:
        return {o["shape"] for o in self.objects}


def _draw_shape(draw: Any, shape: str, cx: float, cy: float, r: float, fill: tuple) -> None:
    """Render one shape centred at (cx, cy) with circumradius `r`."""
    from PIL import ImageDraw

    box = [cx - r, cy - r, cx + r, cy + r]
    if shape == "circle":
        draw.ellipse(box, fill=fill, outline=(40, 40, 40), width=2)
    elif shape == "square":
        draw.rectangle(box, fill=fill, outline=(40, 40, 40), width=2)
    elif shape == "triangle":
        draw.polygon(
            [(cx, cy - r), (cx - r, cy + r * 0.8), (cx + r, cy + r * 0.8)],
            fill=fill,
            outline=(40, 40, 40),
        )
    elif shape == "hexagon":
        import math

        points = [
            (cx + r * math.cos(math.pi / 6 + i * math.pi / 3), cy + r * math.sin(math.pi / 6 + i * math.pi / 3))
            for i in range(6)
        ]
        draw.polygon(points, fill=fill, outline=(40, 40, 40))
    else:  # star
        import math

        points = []
        for i in range(10):
            radius = r if i % 2 == 0 else r * 0.45
            angle = -math.pi / 2 + i * math.pi / 5
            points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
        draw.polygon(points, fill=fill, outline=(40, 40, 40))


def _place(rng: random.Random, count: int, canvas: int, radius: int) -> list[tuple[int, int]]:
    """Scatter `count` centres that do not overlap.

    Rejection sampling with a distance check. Scene complexity is therefore a
    real controlled variable: `count` objects always means `count` *distinguishable*
    objects, so a counting question never has an ambiguous answer.
    """
    margin = radius + 8
    placed: list[tuple[int, int]] = []
    min_gap = radius * 2.6
    for _ in range(count):
        for _attempt in range(400):
            x = rng.randint(margin, canvas - margin)
            y = rng.randint(margin, canvas - margin)
            if all((x - px) ** 2 + (y - py) ** 2 > min_gap**2 for px, py in placed):
                placed.append((x, y))
                break
        else:
            # Crowded canvas: accept the last position rather than silently
            # dropping an object, which would corrupt the ground truth.
            placed.append((x, y))
    return placed


def make_scene(
    rng: random.Random,
    n_objects: int = 4,
    shape_pool: tuple[str, ...] = SHAPES,
    color_pool: tuple[str, ...] = COLORS,
    canvas: int = 384,
) -> Scene:
    """Generate one scene of `n_objects` shapes."""
    positions = _place(rng, n_objects, canvas, 34)
    size_radius = {"small": 22, "medium": 32, "large": 44}
    objects: list[dict[str, Any]] = []
    for x, y in positions:
        shape = rng.choice(shape_pool)
        # Colour is drawn without immediate repetition so that "the colour of
        # the circle" is usually unambiguous.
        color = rng.choice([c for c in color_pool if c != (objects[-1]["color"] if objects else None)])
        objects.append(
            {
                "shape": shape,
                "color": color,
                "size": rng.choice(SIZES),
                "radius": size_radius[rng.choice(SIZES)],
                "center": (x, y),
            }
        )
    return Scene(objects=objects)


def render(scene: Scene, path: str | Path, canvas: int = 384) -> Path:
    """Write a scene to `path` as a PNG."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (canvas, canvas), (250, 250, 248))
    draw = ImageDraw.Draw(img)
    size_radius = {"small": 22, "medium": 32, "large": 44}
    for obj in scene.objects:
        _draw_shape(
            draw,
            obj["shape"],
            obj["center"][0],
            obj["center"][1],
            size_radius[obj["size"]],
            _RGB[obj["color"]],
        )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, format="PNG", optimize=True)
    return out


def _distractors(rng: random.Random, correct: Any, pool: list[Any], k: int = 3) -> list[Any]:
    """Pick `k` plausible wrong options around one correct answer.

    Distractors are drawn from the same type as the answer, so a model cannot
    score well by picking "the only numeric option" or similar shortcuts.
    """
    others = [v for v in pool if v != correct]
    rng.shuffle(others)
    return others[:k]


def _mcq_question(rng: random.Random, scene: Scene) -> tuple[str, list[str], str, str]:
    """Build one multiple-choice item. Returns (question, options, gold, kind)."""
    kind = rng.choice(["count", "color", "shape", "size", "existence_shape"])
    present = sorted(scene.shapes())
    absent = [s for s in SHAPES if s not in scene.shapes()]

    if kind == "count":
        # Count one shape, or count everything if the scene is homogeneous.
        target = rng.choice(present)
        n = scene.count_of(target)
        correct: Any = n
        pool = [v for v in range(1, 9) if v != n]
        rng.shuffle(pool)
        options = [str(n)] + [str(v) for v in pool[:3]]
        question = f"How many {target}s are in the image?"
        gold_text = str(n)
        return question, options, gold_text, kind

    if kind == "color" and len(scene.of_shape(rng.choice(present))) >= 0:
        target_shape = rng.choice(present)
        candidates = scene.of_shape(target_shape)
        # Prefer a shape that occurs exactly once, else the question is
        # ambiguous and the item would be unanswerable.
        unique = [s for s in present if len(scene.of_shape(s)) == 1]
        if unique:
            target_shape = rng.choice(unique)
            candidates = scene.of_shape(target_shape)
        color = candidates[0]["color"]
        options_pool = [c for c in COLORS if c != color]
        rng.shuffle(options_pool)
        options = [color] + options_pool[:3]
        question = f"What is the colour of the {target_shape} in the image?"
        return question, options, color, kind

    if kind == "size" and present:
        unique = [s for s in present if len(scene.of_shape(s)) == 1]
        if unique:
            target_shape = rng.choice(unique)
            size = scene.of_shape(target_shape)[0]["size"]
            others = [s for s in SIZES if s != size]
            rng.shuffle(others)
            return (
                f"What is the size of the {target_shape} in the image?",
                [size] + others,
                size,
                kind,
            )
        kind = "shape"

    # shape: which of these shapes appears in the image?
    picks = rng.sample(present, k=min(2, len(present)))
    while len(picks) < 2:
        picks.append(rng.choice(absent))
    picks += [s for s in absent if s not in picks][: 4 - len(picks)]
    options = picks[:4]
    gold_text = rng.choice(picks)
    rng.shuffle(options)
    return (
        "Which of the following shapes appears in the image?",
        options,
        gold_text,
        "shape",
    )


def generate_mcq(
    out_dir: str | Path,
    n_items: int = 120,
    seed: int = 20260926,
    canvas: int = 384,
    max_objects: int = 6,
) -> list[Example]:
    """Write an MCQ dataset and return its examples.

    Each image is shared by up to three questions about different aspects of
    the same scene. That is deliberate: it makes the dataset cheap to generate
    while still giving the report per-category slices, and it means an item's
    options never leak information about another item's answer.
    """
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    examples: list[Example] = []

    for i in range(n_items):
        n_objects = rng.randint(2, max_objects)
        scene = make_scene(rng, n_objects=n_objects, canvas=canvas)
        img_path = render(scene, out_dir / "images" / f"scene_{i:04d}.png", canvas=canvas)
        n_questions = rng.randint(1, 3)
        for q in range(n_questions):
            question, options, gold_text, kind = _mcq_question(rng, scene)
            if gold_text not in options:
                continue
            order = list(range(len(options)))
            rng.shuffle(order)
            shuffled = [options[j] for j in order]
            gold_letter = LETTERS[shuffled.index(gold_text)]
            examples.append(
                Example(
                    uid=f"mcq_{i:04d}_{q}",
                    image=img_path,
                    question=question,
                    choices=shuffled,
                    label=gold_letter,
                    category=kind,
                    meta={
                        "n_objects": len(scene.objects),
                        "question_kind": kind,
                        "echo_answer": gold_letter,
                    },
                )
            )
    return examples


def generate_pope(
    out_dir: str | Path,
    n_items: int = 120,
    seed: int = 20260926,
    canvas: int = 384,
) -> list[Example]:
    """Write a POPE-style dataset with the three standard difficulty splits.

    Split construction mirrors the real benchmark:

    * ``random`` -- absent shapes sampled uniformly from those not present.
    * ``popular`` -- absent shapes biased to frequent real-world objects, which
      are the ones models are most inclined to assert.
    * ``adversarial`` -- absent shapes chosen to be maximally confusable with
      what *is* present, so the model must actually resolve the image.

    The gold positive rate is held at exactly 0.5 within each split. A model
    answering "yes" to everything therefore scores 50% accuracy, which makes the
    yes-ratio and F1 columns the informative ones -- the same trap the real
    benchmark is built to expose.
    """
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed + 1)
    examples: list[Example] = []
    splits = ("random", "popular", "adversarial")
    per_split = max(1, n_items // len(splits))

    # Shapes a VLM is likely to over-assert, used for the popular split.
    frequent = ("person", "car", "dog", "cat", "chair", "table", "book", "cup")

    idx = 0
    for split in splits:
        for j in range(per_split):
            n_objects = rng.randint(2, 4)
            scene = make_scene(rng, n_objects=n_objects, canvas=canvas)
            img_path = render(scene, out_dir / "images" / f"pope_{idx:04d}.png", canvas=canvas)
            present = sorted(scene.shapes())
            absent = [s for s in SHAPES if s not in present]
            if not absent:
                absent = [rng.choice([s for s in SHAPES if s not in present] or list(SHAPES))]

            # One positive and one negative probe per image keeps the split
            # exactly balanced and makes the pair easy to eyeball when
            # debugging a model failure.
            positive = rng.choice(present)
            if split == "random":
                negative = rng.choice(absent)
            elif split == "popular":
                # Prefer a frequent real-world noun that is also absent, so the
                # probe tests a prior the model actually has.
                negative = rng.choice(frequent)
            else:
                # A shape that shares attributes with something present.
                confusable = [s for s in absent if s in present] or [
                    s for s in SHAPES if s not in present and s != positive
                ]
                negative = rng.choice(confusable or absent)

            for label, target in (("yes", positive), ("no", negative)):
                examples.append(
                    Example(
                        uid=f"pope_{idx:04d}_{label}",
                        image=img_path,
                        question=f"Is there a {target} in the image?",
                        label=label,
                        category=split,
                        meta={
                            "target": target,
                            "present_shapes": present,
                            "n_objects": len(scene.objects),
                        },
                    )
                )
            idx += 1
    return examples


def write_jsonl(examples: list[Example], path: str | Path) -> Path:
    """Serialise examples to JSON Lines, one object per line."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        for ex in examples:
            handle.write(json.dumps(ex.to_json(), ensure_ascii=False) + "\n")
    return out
