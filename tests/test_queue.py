"""Review queue: state, decisions, export, and the tool surface over it."""

from __future__ import annotations

import json

from minicpm_ocr.core.client import Block, Page
from minicpm_ocr.core.queue import Queue


def test_add_and_list(tmp_path):
    queue = Queue(tmp_path / "q.sqlite3")
    item_id = queue.add(source="a.pdf", reason="agent_flagged", content="1,2x4", page=0, label="text")

    items = queue.pending()
    assert len(items) == 1
    assert items[0]["id"] == item_id
    assert items[0]["content"] == "1,2x4"
    assert items[0]["state"] == "pending"
    assert queue.stats()["pending"] == 1


def test_decide_only_pending(tmp_path):
    queue = Queue(tmp_path / "q.sqlite3")
    item_id = queue.add(source="a.pdf", reason="agent_flagged")

    decided = queue.decide(item_id, "accepted", "checked by hand")
    assert decided is not None and decided["state"] == "accepted"
    assert decided["note"] == "checked by hand"
    assert queue.decide(item_id, "rejected") is None
    assert queue.pending() == []
    assert queue.stats()["accepted"] == 1


def test_add_failed_blocks_keeps_bbox_and_job(tmp_path):
    queue = Queue(tmp_path / "q.sqlite3")
    page = Page(
        index=2,
        markdown="",
        blocks=(
            Block(0, "text", "reads fine", "ok", (1, 2, 3, 4)),
            Block(1, "table", "", "failed", (5, 6, 7, 8)),
        ),
    )

    ids = queue.add_failed_blocks(source="doc.pdf", pages=[page], job_id="j1")
    assert len(ids) == 1
    item = queue.get(ids[0])
    assert item["reason"] == "block_status_failed"
    assert item["label"] == "table"
    assert item["bbox"] == [5, 6, 7, 8]
    assert item["page"] == 2
    assert item["job_id"] == "j1"


def test_export_csv(tmp_path):
    queue = Queue(tmp_path / "q.sqlite3")
    queue.add(source="a.pdf", reason="agent_flagged")
    queue.add(source="b.pdf", reason="agent_flagged")

    path, rows = queue.export_csv(tmp_path / "out")
    assert rows == 2
    text = open(path, encoding="utf-8").read()
    assert "review-queue" not in text  # header, not filename
    assert text.count("\n") >= 3
    assert "a.pdf" in text and "b.pdf" in text


def test_tool_actions_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("MINICPM_OCR_DB", str(tmp_path / "tool.sqlite3"))
    from minicpm_ocr import tools

    added = json.loads(tools.ocr_queue({"action": "add", "source": "doc.pdf", "content": "x", "score": 0.2}))
    assert added["ok"] is True
    item_id = added["item"]["id"]
    assert added["item"]["score"] == 0.2

    accepted = json.loads(tools.ocr_queue({"action": "accept", "item_id": item_id, "note": "checked"}))
    assert accepted["ok"] is True and accepted["item"]["state"] == "accepted"

    again = json.loads(tools.ocr_queue({"action": "accept", "item_id": item_id}))
    assert again["ok"] is False
    assert "no pending" in again["error"]["message"]


def test_tool_rejects_bad_action(tmp_path, monkeypatch):
    monkeypatch.setenv("MINICPM_OCR_DB", str(tmp_path / "tool.sqlite3"))
    from minicpm_ocr import tools

    result = json.loads(tools.ocr_queue({"action": "nonsense"}))
    assert result["ok"] is False
    assert result["error"]["code"] == "request_rejected"


def test_skipped_regions_are_not_review_items(tmp_path):
    queue = Queue(tmp_path / "q.sqlite3")
    page = Page(
        index=0,
        markdown="",
        blocks=(
            Block(0, "image", "", "skipped_no_vlm", (1, 2, 3, 4)),
            Block(1, "seal", "", "skipped_seal", (5, 6, 7, 8)),
            Block(2, "table", "", "failed", (9, 10, 11, 12)),
        ),
    )
    ids = queue.add_failed_blocks(source="doc.pdf", pages=[page])
    assert len(ids) == 1
    assert queue.get(ids[0])["label"] == "table"


def test_db_path_explicit_and_env(tmp_path, monkeypatch):
    from minicpm_ocr.core import queue as queue_mod

    explicit = tmp_path / "explicit.sqlite3"
    assert queue_mod.db_path(explicit) == explicit
    monkeypatch.setenv("MINICPM_OCR_DB", str(tmp_path / "env.sqlite3"))
    assert queue_mod.db_path() == tmp_path / "env.sqlite3"


def test_add_records_bbox_and_job(tmp_path):
    queue = Queue(tmp_path / "q.sqlite3")
    item_id = queue.add(
        source="doc.pdf",
        reason="low_score",
        content="x",
        page=3,
        label="formula",
        bbox=[1, 2, 3, 4],
        score=0.2,
        job_id="j9",
    )
    item = queue.get(item_id)
    assert item["bbox"] == [1, 2, 3, 4]
    assert item["job_id"] == "j9"
    assert item["score"] == 0.2
    assert item["page"] == 3


def test_get_missing_and_invalid_decision(tmp_path):
    queue = Queue(tmp_path / "q.sqlite3")
    assert queue.get(999) is None
    item_id = queue.add(source="doc.pdf", reason="agent_flagged")
    try:
        queue.decide(item_id, "maybe")
    except ValueError as exc:
        assert "accepted or rejected" in str(exc)
    else:
        raise AssertionError("invalid decision was accepted")


def test_min_score_queues_an_ok_block(tmp_path):
    queue = Queue(tmp_path / "q.sqlite3")
    page = Page(
        index=0,
        markdown="",
        blocks=(Block(0, "text", "plausible", "ok", (1, 2, 3, 4), score=0.2),),
    )
    ids = queue.add_failed_blocks(source="doc.pdf", pages=[page], min_score=0.6)
    assert len(ids) == 1
    item = queue.get(ids[0])
    assert item["reason"] == "low_score"
    assert item["score"] == 0.2


def test_export_empty_queue(tmp_path):
    queue = Queue(tmp_path / "q.sqlite3")
    path, rows = queue.export_csv(tmp_path / "exports")
    assert rows == 0
    text = open(path, encoding="utf-8").read()
    assert text.startswith("id,created_at,source")
