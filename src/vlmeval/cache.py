"""On-disk response cache.

Re-running an evaluation should not re-pay for answers you already have, and a
half-finished run must survive a Ctrl-C. Both needs reduce to one idea: key
every request by *what was asked* and store the answer, so a repeat request is a
lookup rather than a bill.

The key covers the model identity, the generation parameters, the image content
hash and the prompt. Two consequences worth stating:

* Editing an image on disk invalidates its cache entries automatically, so a
  regenerated dataset can never be scored against stale answers.
* Entries are content-addressed, so the cache is safely shared across runs,
  models and machines. Deleting it costs time and money, never correctness.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from .types import ModelResponse, file_digest, stable_hash

_SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    key         TEXT PRIMARY KEY,
    model       TEXT NOT NULL,
    created_at  REAL NOT NULL,
    payload     TEXT NOT NULL
)
"""


class ResponseCache:
    """A thread-safe SQLite cache of model responses.

    SQLite rather than a directory of JSON files because a run issues thousands
    of concurrent lookups and needs a single transactional writer.

    Each thread gets its own connection. A single connection shared between
    threads looks tempting -- ``check_same_thread=False`` makes it legal -- but
    the sqlite3 statement cache behind that connection is not itself
    thread-safe, and concurrent use fails in ways that are miserable to
    diagnose: ``OperationalError: bad parameter or other API misuse``, a bare
    ``IndexError: tuple index out of range`` from a garbled result row, or worse,
    a silently wrong row. Under WAL, one connection per thread means readers
    never block each other and only writers serialise, which is the contention
    profile this workload actually has.

    Args:
        path: Database file. Created on first write.
        enabled: When False, every lookup misses and nothing is stored, which
            is what ``--no-cache`` does.
    """

    def __init__(self, path: str | Path = "data/cache/responses.sqlite3", enabled: bool = True) -> None:
        self.path = Path(path)
        self.enabled = enabled
        self._local = threading.local()
        self._write_lock = threading.Lock()
        self._all_lock = threading.Lock()
        self._connections: list[sqlite3.Connection] = []
        self._closed = False
        if enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._execute_schema(self._new_connection())

    def _new_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.path),
            check_same_thread=False,
            timeout=30.0,
            isolation_level=None,
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA synchronous=NORMAL")
        with self._all_lock:
            self._connections.append(conn)
        return conn

    def _conn(self) -> sqlite3.Connection:
        """This thread's connection, opened on first use."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._new_connection()
            self._local.conn = conn
        return conn

    def _execute_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute(_SCHEMA)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_model ON responses(model)")

    @staticmethod
    def make_key(
        *,
        model_name: str,
        model_info: dict[str, Any],
        prompt: str,
        image: Any,
        params: dict[str, Any],
        gold: str | None = None,
    ) -> str:
        """Build the cache key for one request.

        `model_info` participates because two adapters with the same label can
        still be different models -- a local checkpoint and a hosted endpoint
        both called `qwen`, say. Including it prevents one silently answering
        for the other.

        Images contribute a *content* hash and nothing else. Folding in the
        filename would break the content-addressing this module promises: the
        same picture re-downloaded under a new name, or served from a different
        directory, would miss the cache and re-pay for an answer already bought.

        `gold` is included only for adapters that declare ``sees_gold`` in their
        info -- currently just the offline echo model. No real model is asked a
        question whose answer varies, so for them the gold is deliberately not
        part of what identifies the request.
        """
        image_key = None
        if isinstance(image, (str, Path)):
            p = Path(image)
            image_key = f"file:{file_digest(p)}" if p.exists() else f"file:{image}"
        elif image is not None:
            image_key = f"obj:{type(image).__name__}:{stable_hash(str(image))}"
        return stable_hash(
            {
                "model": model_name,
                "info": model_info,
                "prompt": prompt,
                "image": image_key,
                "params": params,
                "gold": gold if model_info.get("sees_gold") else None,
            },
            length=32,
        )

    def get(self, key: str) -> ModelResponse | None:
        """Return a cached response, or ``None`` on a miss."""
        if not self.enabled or self._closed:
            return None
        row = self._conn().execute("SELECT payload FROM responses WHERE key = ?", (key,)).fetchone()
        if row is None or not row[0]:
            return None
        try:
            data = json.loads(row[0])
        except (TypeError, ValueError):
            # A corrupt row is a cache miss, never a failed run.
            return None
        return ModelResponse(
            text=data.get("text", ""),
            prompt_tokens=int(data.get("prompt_tokens", 0)),
            completion_tokens=int(data.get("completion_tokens", 0)),
            latency_s=float(data.get("latency_s", 0.0)),
            finish_reason=data.get("finish_reason"),
        )

    def put(self, key: str, model_name: str, response: ModelResponse) -> None:
        """Store a response.

        Failures are ignored on purpose: a cache write must never take down a
        run that has already paid for the answer.
        """
        if not self.enabled or self._closed:
            return
        payload = json.dumps(
            {
                "text": response.text,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "latency_s": response.latency_s,
                "finish_reason": response.finish_reason,
            },
            ensure_ascii=False,
        )
        try:
            with self._write_lock:
                self._conn().execute(
                    "INSERT OR REPLACE INTO responses (key, model, created_at, payload) "
                    "VALUES (?, ?, ?, ?)",
                    (key, model_name, time.time(), payload),
                )
        except sqlite3.Error:
            pass

    def stats(self) -> dict[str, int]:
        """Entry count, for the run manifest."""
        if not self.enabled or self._closed:
            return {"entries": 0}
        try:
            row = self._conn().execute("SELECT COUNT(*) FROM responses").fetchone()
        except sqlite3.Error:
            return {"entries": 0}
        return {"entries": int(row[0]) if row else 0}

    def close(self) -> None:
        """Close every connection handed out so far."""
        with self._all_lock:
            connections, self._connections = self._connections, []
            self._closed = True
        for conn in connections:
            try:
                conn.close()
            except sqlite3.Error:
                pass
