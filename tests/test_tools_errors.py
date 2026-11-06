"""Tool-surface validation, error mapping, registration, and helper edges."""

from __future__ import annotations

import json
import os
import sys
import types

import pytest

import minicpm_ocr
from minicpm_ocr import tools
from minicpm_ocr.core.client import Block, Page, ParseResult
from minicpm_ocr.core.errors import BackendUnavailable, RequestRejected
from minicpm_ocr.core.probe import Backend

PNG = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR"
    b"\x00\x00\x00\x14\x00\x00\x00\x0a"
    b"\x08\x02\x00\x00\x00"
    b"\x00" * 8
)


@pytest.fixture(autouse=True)
def reset_context(monkeypatch):
    monkeypatch.delenv("MODELBEST_API_KEY", raising=False)
    tools.set_context(None)
    yield
    tools.set_context(None)


class RecordingCtx:
    def __init__(self):
        self.registered = []

    def register_tool(self, **kwargs):
        self.registered.append(kwargs)

    def get_config(self, key, default=None):
        return default


def test_register_wires_four_tools():
    ctx = RecordingCtx()
    minicpm_ocr.register(ctx)
    assert [entry["name"] for entry in ctx.registered] == [
        "ocr_read",
        "ocr_job",
        "ocr_queue",
        "ocr_verify",
    ]
    assert all(entry["toolset"] == "minicpm-ocr" for entry in ctx.registered)
    assert all(entry["schema"]["name"] == entry["name"] for entry in ctx.registered)


def test_ocr_read_rejects_bad_mode_before_touching_the_file():
    result = json.loads(tools.ocr_read({"path": "/does/not/exist.png", "mode": "word"}))
    assert result["ok"] is False
    assert result["error"]["code"] == "request_rejected"
    assert "summary" in result["error"]["message"]


def test_ocr_read_requires_a_path():
    result = json.loads(tools.ocr_read({"path": "   "}))
    assert result["ok"] is False
    assert result["error"]["code"] == "request_rejected"


def test_ocr_read_rejects_unsupported_bytes(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("not a document")
    result = json.loads(tools.ocr_read({"path": str(path)}))
    assert result["ok"] is False
    assert result["error"]["code"] == "payload_too_large"
    assert "not a supported input" in result["error"]["message"]


def test_ocr_verify_requires_text():
    result = json.loads(tools.ocr_verify({"path": "x.png", "text": "  "}))
    assert result["ok"] is False
    assert result["error"]["code"] == "request_rejected"
    assert "text is required" in result["error"]["message"]


def test_ocr_job_requires_an_id():
    result = json.loads(tools.ocr_job({"action": "status"}))
    assert result["ok"] is False
    assert result["error"]["code"] == "request_rejected"


def test_queue_action_validation(tmp_path, monkeypatch):
    monkeypatch.setenv("MINICPM_OCR_DB", str(tmp_path / "queue.sqlite3"))

    assert json.loads(tools.ocr_queue({"action": "add"}))["error"]["code"] == "request_rejected"
    bad_page = json.loads(
        tools.ocr_queue({"action": "add", "source": "doc.pdf", "page": "not-an-int"})
    )
    assert bad_page["error"]["code"] == "request_rejected"

    missing = json.loads(tools.ocr_queue({"action": "show", "item_id": 999}))
    assert missing["error"]["code"] == "request_rejected"
    assert "no queue item" in missing["error"]["message"]

    no_id = json.loads(tools.ocr_queue({"action": "show"}))
    assert no_id["error"]["code"] == "request_rejected"
    assert "item_id" in no_id["error"]["message"]

    invalid = json.loads(tools.ocr_queue({"action": "dance"}))
    assert invalid["error"]["code"] == "request_rejected"


def test_queue_reject_and_show_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("MINICPM_OCR_DB", str(tmp_path / "queue.sqlite3"))
    added = json.loads(tools.ocr_queue({"action": "add", "source": "a.pdf", "content": "1,2"}))
    item_id = added["item"]["id"]
    shown = json.loads(tools.ocr_queue({"action": "show", "item_id": item_id}))
    assert shown["item"]["content"] == "1,2"
    rejected = json.loads(tools.ocr_queue({"action": "reject", "item_id": item_id, "note": "nope"}))
    assert rejected["item"]["state"] == "rejected"
    assert rejected["item"]["note"] == "nope"


def test_job_timeout_env(monkeypatch):
    monkeypatch.setenv("HERMES_MINICPM_OCR_JOB_TIMEOUT", "7.5")
    assert tools._job_timeout() == 7.5
    monkeypatch.setenv("HERMES_MINICPM_OCR_JOB_TIMEOUT", "bad")
    assert tools._job_timeout() == 180.0


def test_helpers_validate_and_clip():
    with pytest.raises(RequestRejected):
        tools._opt_int("x")
    with pytest.raises(RequestRejected):
        tools._opt_float("x")
    assert tools._opt_int(None) is None
    assert tools._opt_float("1.5") == 1.5
    assert tools._clip("abcdef", 4) == ("abc…", True)
    assert tools._clip("abc", 4) == ("abc", False)
    assert tools._excerpt("  hi  ") == "hi"
    assert tools._advice(False, 0.1, 0.6, "forward_error").startswith("Jev did not produce")
    assert tools._advice(True, 0.9, 0.6).startswith("Accepted")
    assert "Below threshold" in tools._advice(False, 0.1, 0.6)


def test_legacy_context_mapping_is_used_when_keyed_call_fails():
    class Legacy:
        def get_config(self):
            return {"base_url": "http://127.0.0.1:8966", "threshold": 0.7}

    tools.set_context(Legacy())
    settings = tools._settings()
    assert settings["base_url"] == "http://127.0.0.1:8966"
    assert settings["threshold"] == 0.7


def test_settings_context_exception_is_swallowed():
    class Broken:
        def get_config(self, key, default=None):
            raise RuntimeError("config broken")

    tools.set_context(Broken())
    assert tools._settings() == {}


def test_secret_route_has_priority(monkeypatch):
    fake = types.ModuleType("agent.secret_scope")
    fake.get_secret = lambda name: "from-secret"
    monkeypatch.setitem(sys.modules, "agent.secret_scope", fake)
    assert tools._api_key({"api_key": "from-config"}) == "from-secret"


def test_env_key_is_standalone_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, "agent.secret_scope", None)
    monkeypatch.setenv("MODELBEST_API_KEY", "env-key")
    assert tools._api_key({}) == "env-key"


def test_client_refuses_when_no_candidate_is_usable(monkeypatch):
    monkeypatch.setattr(tools, "candidates", lambda settings: [])
    with pytest.raises(BackendUnavailable):
        tools._client()


def test_client_skips_cloud_without_a_key(monkeypatch):
    monkeypatch.setattr(
        tools,
        "candidates",
        lambda settings: [Backend(name="cloud", base_url="https://api.modelbest.ai", api="modelbest")],
    )
    with pytest.raises(BackendUnavailable):
        tools._client()


def test_client_skips_unreachable_loopback(monkeypatch):
    monkeypatch.setattr(
        tools,
        "candidates",
        lambda settings: [
            Backend(name="local", base_url="http://127.0.0.1:18499", api="chat", source="loopback")
        ],
    )
    monkeypatch.setattr(tools, "reachable", lambda backend: False)
    with pytest.raises(BackendUnavailable):
        tools._client()


def test_client_returns_reachable_local(monkeypatch):
    backend = Backend(name="local", base_url="http://127.0.0.1:18499", api="chat", source="loopback")
    monkeypatch.setattr(tools, "candidates", lambda settings: [backend])
    monkeypatch.setattr(tools, "reachable", lambda candidate: True)
    client, chosen = tools._client()
    assert chosen is backend
    assert client.surface == "chat"


def test_unexpected_exception_never_escapes(monkeypatch):
    monkeypatch.setattr(tools, "read_source", lambda source: (_ for _ in ()).throw(RuntimeError("boom")))
    result = json.loads(tools.ocr_read({"path": "x.png"}))
    assert result["ok"] is False
    assert result["error"]["code"] == "internal"
    assert "boom" in result["error"]["message"]


class FakeClient:
    def parse_page(self, data, *, name):
        return ParseResult(pages=(Page(index=0, markdown="text", blocks=()),))


def test_queue_failure_does_not_lose_the_read(tmp_path, monkeypatch):
    path = tmp_path / "page.png"
    path.write_bytes(PNG)
    backend = Backend(name="local", base_url="http://127.0.0.1:18499", api="chat")
    monkeypatch.setattr(tools, "_client", lambda: (FakeClient(), backend))

    class BrokenQueue:
        def __init__(self):
            raise RuntimeError("disk full")

    monkeypatch.setattr(tools, "Queue", BrokenQueue)
    result = json.loads(tools.ocr_read({"path": str(path), "queue": True}))
    assert result["ok"] is True
    assert result["preview"] == "text"
    assert "disk full" in result["queue_error"]


def test_shape_modes_truncate_and_separate_skips():
    page = Page(
        index=0,
        markdown="x" * 1000,
        blocks=(
            Block(0, "text", "ok", "ok", (1, 2, 3, 4)),
            Block(1, "table", "", "failed", (5, 6, 7, 8)),
            Block(2, "image", "", "skipped_no_vlm", (9, 10, 11, 12)),
        ),
    )
    document = Backend(name="doc", base_url="http://127.0.0.1:8966", api="document")
    shaped = tools._shape([page], page.markdown, mode="markdown", max_chars=10, backend=document)
    assert shaped["regions"] == 3
    assert shaped["failed_regions"] == 1
    assert shaped["skipped_regions"] == 1
    assert shaped["truncated"] is True
    assert shaped["markdown"].endswith("…")

    blocks = tools._shape([page], page.markdown, mode="blocks", max_chars=10, backend=document)
    assert [item["status"] for item in blocks["blocks"]] == ["ok", "failed", "skipped_no_vlm"]

    chat = Backend(name="chat", base_url="http://127.0.0.1:18499", api="chat")
    summary = tools._shape([page], "short", mode="summary", max_chars=10, backend=chat)
    assert summary["note"].startswith("This backend has no layout stage")
    assert summary["preview"] == "short"


class CloudClient:
    def parse_image(self, data, *, name):
        return ParseResult(
            pages=(
                Page(
                    index=0,
                    markdown="Total 1,284.50",
                    blocks=(
                        Block(0, "text", "Total 1,284.50", "ok", (1, 2, 3, 4)),
                        Block(1, "table", "", "failed", (5, 6, 7, 8)),
                    ),
                ),
            )
        )


def test_ocr_read_cloud_image_shapes_and_queues_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("MINICPM_OCR_DB", str(tmp_path / "queue.sqlite3"))
    path = tmp_path / "page.png"
    path.write_bytes(PNG)
    backend = Backend(name="modelbest", base_url="https://api.modelbest.ai", api="modelbest")
    monkeypatch.setattr(tools, "_client", lambda: (CloudClient(), backend))

    result = json.loads(tools.ocr_read({"path": str(path), "queue": True}))
    assert result["ok"] is True
    assert result["backend"] == "modelbest"
    assert result["failed_regions"] == 1
    assert result["queued"]
    assert "1,284.50" in result["preview"]


class VerifyClient:
    def jev_score(self, data, text, *, category, threshold):
        from minicpm_ocr.core.client import JevResult

        return JevResult(
            score=0.93,
            prob_no=0.07,
            verdict="scored",
            passed=True,
            category=category,
            threshold=threshold,
            scored=True,
        )


def test_ocr_verify_success_payload(tmp_path, monkeypatch):
    path = tmp_path / "crop.png"
    path.write_bytes(PNG)
    backend = Backend(
        name="modelbest", base_url="https://api.modelbest.ai", api="modelbest"
    )
    monkeypatch.setattr(tools, "_client", lambda: (VerifyClient(), backend))
    result = json.loads(tools.ocr_verify({"path": str(path), "text": "1,284.50"}))
    assert result["ok"] is True
    assert result["score"] == 0.93
    assert result["scored"] is True
    assert result["passed"] is True
    assert result["advice"].startswith("Accepted")


class JobClient:
    def job(self, job_id):
        from minicpm_ocr.core.client import JobStatus, Page

        return JobStatus(
            id=job_id,
            status="Running",
            pages_done=2,
            pages_total=5,
            pages=(Page(index=0, markdown="one"),),
            markdown="one",
        )


def test_ocr_job_success_payload(monkeypatch):
    backend = Backend(
        name="modelbest", base_url="https://api.modelbest.ai", api="modelbest"
    )
    monkeypatch.setattr(tools, "_client", lambda: (JobClient(), backend))
    result = json.loads(tools.ocr_job({"job_id": "j1"}))
    assert result["ok"] is True
    assert result["job_id"] == "j1"
    assert result["pages_done"] == 2
    assert result["fraction"] == 0.4
    assert result["pages"] == 1


def test_client_skips_unreachable_configured_local_base(monkeypatch):
    backend = Backend(
        name="local-chat",
        base_url="http://127.0.0.1:8999",
        api="chat",
        source="settings",
    )
    monkeypatch.setattr(tools, "candidates", lambda settings: [backend])
    monkeypatch.setattr(tools, "reachable", lambda candidate: False)
    with pytest.raises(BackendUnavailable):
        tools._client()


def test_verify_rejects_unreachable_jev_fast(tmp_path, monkeypatch):
    path = tmp_path / "crop.png"
    path.write_bytes(PNG)
    backend = Backend(
        name="local-document",
        base_url="http://127.0.0.1:8966",
        api="document",
        jev_base_url="http://127.0.0.1:18599",
        jev_api="rm",
    )
    monkeypatch.setattr(tools, "_client", lambda: (VerifyClient(), backend))
    monkeypatch.setattr(tools, "reachable", lambda candidate: False)
    result = json.loads(tools.ocr_verify({"path": str(path), "text": "x"}))
    assert result["ok"] is False
    assert result["error"]["code"] == "backend_unavailable"
    assert "not listening" in result["error"]["message"]


class UnscoredClient:
    def jev_score(self, data, text, *, category, threshold):
        from minicpm_ocr.core.client import JevResult

        return JevResult(
            score=-1.0,
            prob_no=2.0,
            verdict="forward_error",
            passed=False,
            category=category,
            threshold=threshold,
            detail="backend not ready",
            scored=False,
        )


def test_ocr_verify_unscored_verdict(tmp_path, monkeypatch):
    path = tmp_path / "crop.png"
    path.write_bytes(PNG)
    backend = Backend(name="modelbest", base_url="https://api.modelbest.ai", api="modelbest")
    monkeypatch.setattr(tools, "_client", lambda: (UnscoredClient(), backend))
    result = json.loads(tools.ocr_verify({"path": str(path), "text": "x"}))
    assert result["ok"] is True
    assert result["scored"] is False
    assert result["passed"] is False
    assert result["advice"].startswith("Jev did not produce")


def test_ocr_job_cancel_action(monkeypatch):
    from minicpm_ocr.core.client import JobStatus

    class CancelClient:
        def cancel_job(self, job_id):
            return JobStatus(id=job_id, status="Cancelled")

    backend = Backend(name="modelbest", base_url="https://api.modelbest.ai", api="modelbest")
    monkeypatch.setattr(tools, "_client", lambda: (CancelClient(), backend))
    result = json.loads(tools.ocr_job({"job_id": "j1", "action": "cancel"}))
    assert result["ok"] is True
    assert result["status"] == "Cancelled"
    assert result["done"] is True
    assert result["completed"] is False


def test_ocr_queue_list_and_export_actions(tmp_path, monkeypatch):
    monkeypatch.setenv("MINICPM_OCR_DB", str(tmp_path / "queue.sqlite3"))
    added = json.loads(tools.ocr_queue({"action": "add", "source": "a.pdf", "content": "x"}))
    assert added["ok"] is True
    listed = json.loads(tools.ocr_queue({"action": "list", "limit": 5}))
    assert len(listed["items"]) == 1
    exported = json.loads(tools.ocr_queue({"action": "export"}))
    assert exported["ok"] is True
    assert exported["rows"] == 1
    assert os.path.exists(exported["path"])


def test_blocks_mode_caps_at_200():
    blocks = tuple(Block(i, "text", f"line {i}", "ok", (i, i, i + 1, i + 1)) for i in range(250))
    page = Page(index=0, markdown="", blocks=blocks)
    backend = Backend(name="doc", base_url="http://127.0.0.1:8966", api="document")
    payload = tools._shape([page], "", mode="blocks", max_chars=100, backend=backend)
    assert len(payload["blocks"]) == 200


def test_ocr_verify_accepts_zero_threshold(tmp_path, monkeypatch):
    path = tmp_path / "crop.png"
    path.write_bytes(PNG)
    backend = Backend(name="modelbest", base_url="https://api.modelbest.ai", api="modelbest")
    monkeypatch.setattr(tools, "_client", lambda: (VerifyClient(), backend))
    result = json.loads(tools.ocr_verify({"path": str(path), "text": "x", "threshold": 0.0}))
    assert result["ok"] is True
    assert result["threshold"] == 0.0
    assert result["passed"] is True


def test_ocr_verify_reads_zero_threshold_from_settings(tmp_path, monkeypatch):
    path = tmp_path / "crop.png"
    path.write_bytes(PNG)
    backend = Backend(name="modelbest", base_url="https://api.modelbest.ai", api="modelbest")
    monkeypatch.setattr(tools, "_client", lambda: (VerifyClient(), backend))
    class Ctx:
        def get_config(self, key, default=None):
            return 0.0 if key == "threshold" else default

    tools.set_context(Ctx())
    result = json.loads(tools.ocr_verify({"path": str(path), "text": "x"}))
    assert result["ok"] is True
    assert result["threshold"] == 0.0
