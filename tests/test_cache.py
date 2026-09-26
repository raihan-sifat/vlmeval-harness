"""Response cache: correctness under threads, and key hygiene.

The bug these guard against was not hypothetical. A single SQLite connection
shared across the runner's worker threads, with ``check_same_thread=False``,
raised ``bad parameter or other API misuse`` and a bare ``tuple index out of
range`` from garbled result rows -- intermittently, and only under load.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tests.helpers import *  # noqa: F401,F403 - path bootstrap

from vlmeval.cache import ResponseCache
from vlmeval.types import ModelResponse


class CacheCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.path = Path(self.tmp) / "responses.sqlite3"

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestCacheBasics(CacheCase):
    def test_miss_then_hit(self) -> None:
        cache = ResponseCache(self.path)
        cache.put("k", "m", ModelResponse(text="hello", completion_tokens=2))
        hit = cache.get("k")
        self.assertIsNotNone(hit)
        self.assertEqual(hit.text, "hello")
        self.assertEqual(hit.completion_tokens, 2)
        cache.close()

    def test_unknown_key_is_a_miss(self) -> None:
        cache = ResponseCache(self.path)
        self.assertIsNone(cache.get("nope"))
        cache.close()

    def test_disabled_cache_stores_nothing(self) -> None:
        cache = ResponseCache(self.path, enabled=False)
        cache.put("k", "m", ModelResponse(text="hello"))
        self.assertIsNone(cache.get("k"))
        self.assertEqual(cache.stats(), {"entries": 0})
        cache.close()

    def test_survives_reopening(self) -> None:
        cache = ResponseCache(self.path)
        cache.put("k", "m", ModelResponse(text="persisted"))
        cache.close()
        again = ResponseCache(self.path)
        self.assertEqual(again.get("k").text, "persisted")
        self.assertEqual(again.stats()["entries"], 1)
        again.close()

    def test_overwrite_is_last_write_wins(self) -> None:
        cache = ResponseCache(self.path)
        cache.put("k", "m", ModelResponse(text="first"))
        cache.put("k", "m", ModelResponse(text="second"))
        self.assertEqual(cache.get("k").text, "second")
        self.assertEqual(cache.stats()["entries"], 1)
        cache.close()

    def test_corrupt_row_is_a_miss_not_a_crash(self) -> None:
        cache = ResponseCache(self.path)
        cache.put("k", "m", ModelResponse(text="fine"))
        cache.close()
        import sqlite3

        conn = sqlite3.connect(str(self.path))
        conn.execute("UPDATE responses SET payload = 'not json' WHERE key = 'k'")
        conn.commit()
        conn.close()
        again = ResponseCache(self.path)
        self.assertIsNone(again.get("k"))
        again.close()

    def test_use_after_close_is_safe(self) -> None:
        cache = ResponseCache(self.path)
        cache.put("k", "m", ModelResponse(text="x"))
        cache.close()
        self.assertIsNone(cache.get("k"))
        cache.put("k", "m", ModelResponse(text="y"))
        self.assertEqual(cache.stats(), {"entries": 0})


class TestCacheConcurrency(CacheCase):
    def test_concurrent_writes_all_land(self) -> None:
        cache = ResponseCache(self.path)
        keys = [f"k{i}" for i in range(600)]
        errors: list[str] = []

        def write(key: str) -> None:
            try:
                cache.put(key, "m", ModelResponse(text=f"v-{key}"))
            except Exception as exc:  # noqa: BLE001
                errors.append(repr(exc))

        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(write, keys))

        self.assertEqual(errors, [])
        self.assertEqual(cache.stats()["entries"], len(keys))
        cache.close()

    def test_concurrent_reads_return_the_right_rows(self) -> None:
        """The regression: concurrent reads returned garbled or wrong rows."""
        cache = ResponseCache(self.path)
        keys = [f"k{i}" for i in range(400)]
        for key in keys:
            cache.put(key, "m", ModelResponse(text=f"v-{key}"))

        wrong: list[str] = []
        errors: list[str] = []

        def read(key: str) -> None:
            try:
                got = cache.get(key)
                if got is None or got.text != f"v-{key}":
                    wrong.append(f"{key} -> {got!r}")
            except Exception as exc:  # noqa: BLE001
                errors.append(repr(exc))

        for _ in range(3):
            with ThreadPoolExecutor(max_workers=16) as pool:
                list(pool.map(read, keys))

        self.assertEqual(errors, [])
        self.assertEqual(wrong, [])
        cache.close()

    def test_mixed_read_write_storm(self) -> None:
        cache = ResponseCache(self.path)
        errors: list[str] = []

        def churn(i: int) -> None:
            try:
                key = f"mixed-{i % 200}"
                cache.get(key)
                cache.put(key, "m", ModelResponse(text=f"t{i}"))
                cache.get(key)
            except Exception as exc:  # noqa: BLE001
                errors.append(repr(exc))

        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(churn, range(1200)))

        self.assertEqual(errors, [])
        self.assertEqual(cache.stats()["entries"], 200)
        cache.close()


class TestCacheKeys(unittest.TestCase):
    def _key(self, **overrides):
        base = dict(
            model_name="m", model_info={}, prompt="p", image="i.png", params={}
        )
        base.update(overrides)
        return ResponseCache.make_key(**base)

    def test_deterministic(self) -> None:
        self.assertEqual(self._key(), self._key())

    def test_varies_with_prompt(self) -> None:
        self.assertNotEqual(self._key(), self._key(prompt="q"))

    def test_varies_with_model_info(self) -> None:
        """Two different models sharing a name must not share answers."""
        self.assertNotEqual(self._key(), self._key(model_info={"checkpoint": "a"}))
        self.assertNotEqual(
            self._key(model_info={"checkpoint": "a"}),
            self._key(model_info={"checkpoint": "b"}),
        )

    def test_varies_with_params(self) -> None:
        self.assertNotEqual(
            self._key(params={"temperature": 0.0}),
            self._key(params={"temperature": 1.0}),
        )

    def test_missing_image_file_is_still_keyed(self) -> None:
        self.assertIsInstance(self._key(image="does/not/exist.png"), str)

    def test_same_path_same_content_same_key(self) -> None:
        tmp = tempfile.mkdtemp()
        try:
            a, b = Path(tmp) / "a.png", Path(tmp) / "b.png"
            a.write_bytes(b"same-bytes")
            b.write_bytes(b"same-bytes")
            self.assertEqual(self._key(image=str(a)), self._key(image=str(b)))
            b.write_bytes(b"different")
            self.assertNotEqual(self._key(image=str(a)), self._key(image=str(b)))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
