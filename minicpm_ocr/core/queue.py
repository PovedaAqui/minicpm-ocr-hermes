"""Durable review queue.

The point of the plugin is not that it reads a page; it is that it knows which
regions it could not read, and keeps them until a person decides. That state
outlives the turn, the context window and the process, so it lives in SQLite
under the plugin's data directory.

``plugin_data_dir`` comes from Hermes when the plugin runs inside Hermes. Standalone
(CLI, tests) the module falls back to ``$HERMES_HOME/plugin-data/minicpm-ocr``,
and ``MINICPM_OCR_DB`` overrides both.
"""

from __future__ import annotations

import csv
import json
import os
import sqlite3
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

PLUGIN_ID = "minicpm-ocr"
_SCHEMA = """
create table if not exists items (
    id          integer primary key autoincrement,
    created_at  text    not null,
    source      text    not null,
    job_id      text,
    page        integer,
    label       text,
    bbox        text,
    content     text,
    score       real,
    reason      text    not null,
    state       text    not null default 'pending',
    note        text,
    decided_at  text
);
create index if not exists items_state_idx on items(state);
create index if not exists items_source_idx on items(source);
"""


def db_path(explicit: str | Path | None = None) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    env = os.environ.get("MINICPM_OCR_DB")
    if env:
        return Path(env).expanduser()
    try:  # inside Hermes
        from plugins.plugin_storage import plugin_data_dir

        return Path(plugin_data_dir(PLUGIN_ID)) / "queue.sqlite3"
    except Exception:  # standalone: CLI, tests
        home = Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))
        return home / "plugin-data" / PLUGIN_ID / "queue.sqlite3"


class Queue:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = db_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # -- writes -------------------------------------------------------------

    def add(
        self,
        *,
        source: str,
        reason: str,
        content: str = "",
        page: int | None = None,
        label: str = "",
        bbox: Sequence[int] | None = None,
        score: float | None = None,
        job_id: str | None = None,
    ) -> int:
        now = _now()
        with self._conn() as conn:
            cur = conn.execute(
                "insert into items (created_at, source, job_id, page, label, bbox, content, score, reason)"
                " values (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    now,
                    source,
                    job_id,
                    page,
                    label,
                    json.dumps(list(bbox)) if bbox else None,
                    content,
                    score,
                    reason,
                ),
            )
            return int(cur.lastrowid)

    def add_failed_blocks(
        self,
        *,
        source: str,
        pages: Sequence[Any],
        job_id: str | None = None,
        min_score: float | None = None,
    ) -> list[int]:
        """Enqueue every region a page reported as not-ok (and any scoring below ``min_score``)."""
        ids: list[int] = []
        for page in pages:
            for block in page.blocks:
                below = min_score is not None and block.score is not None and block.score < min_score
                if block.ok and not below:
                    continue
                ids.append(
                    self.add(
                        source=source,
                        reason="block_status_failed" if not block.ok else "low_score",
                        content=block.content,
                        page=page.index,
                        label=block.label,
                        bbox=block.bbox,
                        score=getattr(block, "score", None),
                        job_id=job_id,
                    )
                )
        return ids

    def decide(self, item_id: int, decision: str, note: str = "") -> dict[str, Any] | None:
        if decision not in ("accepted", "rejected"):
            raise ValueError("decision must be accepted or rejected")
        with self._conn() as conn:
            cur = conn.execute(
                "update items set state = ?, note = ?, decided_at = ? where id = ? and state = 'pending'",
                (decision, note or None, _now(), item_id),
            )
            if cur.rowcount == 0:
                return None
            row = conn.execute("select * from items where id = ?", (item_id,)).fetchone()
        return _row(row) if row else None

    # -- reads --------------------------------------------------------------

    def pending(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "select * from items where state = 'pending' order by id limit ?", (max(1, limit),)
            ).fetchall()
        return [_row(r) for r in rows]

    def get(self, item_id: int) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute("select * from items where id = ?", (item_id,)).fetchone()
        return _row(row) if row else None

    def stats(self) -> dict[str, Any]:
        with self._conn() as conn:
            rows = conn.execute("select state, count(*) as n from items group by state").fetchall()
            sources = conn.execute("select count(distinct source) as n from items").fetchone()
        counts = {r["state"]: r["n"] for r in rows}
        return {
            "pending": counts.get("pending", 0),
            "accepted": counts.get("accepted", 0),
            "rejected": counts.get("rejected", 0),
            "documents": sources["n"] if sources else 0,
        }

    def export_csv(self, directory: str | Path | None = None) -> tuple[str, int]:
        target = Path(directory) if directory else self.path.parent / "exports"
        target.mkdir(parents=True, exist_ok=True)
        out = target / f"review-queue-{time.strftime('%Y%m%d-%H%M%S')}.csv"
        with self._conn() as conn:
            rows = conn.execute("select * from items order by id").fetchall()
        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["id", "created_at", "source", "job_id", "page", "label", "bbox", "content", "score",
                 "reason", "state", "note", "decided_at"]
            )
            for row in rows:
                writer.writerow([row[k] for k in row.keys()])
        return str(out), len(rows)


def _row(row: sqlite3.Row) -> dict[str, Any]:
    out = dict(row)
    if out.get("bbox"):
        try:
            out["bbox"] = json.loads(out["bbox"])
        except ValueError:
            pass
    return out


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
