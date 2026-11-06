"""Local document service parity: jobs, blocks, and the local Jev render+score API."""

from __future__ import annotations

import json

from minicpm_ocr import tools
from minicpm_ocr.core.client import OcrClient
from minicpm_ocr.core.probe import Backend, candidates

PNG = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR"
    b"\x00\x00\x00\x14\x00\x00\x00\x0a"
    b"\x08\x02\x00\x00\x00"
    b"\x00" * 8
)

PAGE = {
    "metadata": {"page_index": 0},
    "choices": [{"message": {"content": "Total due: 1,284.50 EUR"}}],
    "minicpm_ocr": {
        "image_width": 900,
        "image_height": 1200,
        "blocks": [
            {
                "order": 0,
                "label": "text",
                "bbox": [60, 160, 500, 180],
                "content": "Total due: 1,284.50",
                "status": "ok",
            },
            {
                "order": 1,
                "label": "table",
                "bbox": [60, 300, 800, 600],
                "content": "",
                "status": "failed",
            },
        ],
    },
}


class Fake:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, method, url, headers, body, timeout):
        self.calls.append({"method": method, "url": url, "body": body})
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, dict):
            return 200, {}, json.dumps(item).encode()
        return item


def document_client(transport, **kwargs):
    kwargs.setdefault("base_url", "http://127.0.0.1:8966")
    kwargs.setdefault("surface", "document")
    kwargs.setdefault("max_attempts", 1)
    return OcrClient(transport=transport, **kwargs)


def test_document_surface_has_blocks_and_jobs():
    backend = Backend(name="local-document", base_url="http://127.0.0.1:8966", api="document")
    assert backend.surface == "document"
    assert backend.blocks is True and backend.jobs is True
    assert backend.client().jobs_path == "/api/document-parse/jobs"

    chat = Backend(name="local-vllm", base_url="http://127.0.0.1:8000", api="chat")
    assert chat.surface == "chat"
    assert chat.blocks is False and chat.jobs is False


def test_local_api_document_selects_document_candidates():
    backends = candidates({"backend": "local", "local_api": "document"})
    assert [b.name for b in backends] == ["local-document", "local-document-docker"]
    assert all(b.surface == "document" for b in backends)


def test_document_job_uses_api_path_and_parses_minicpm_ocr_envelope():
    fake = Fake([{"id": "job_1", "status": "Success", "pages_done": 1, "pages_total": 1, "pages": [PAGE]}])
    status = document_client(fake).job("job_1")
    assert fake.calls[0]["url"].endswith("/api/document-parse/jobs/job_1")
    assert status.pages[0].blocks[0].content == "Total due: 1,284.50"
    assert status.pages[0].blocks[0].score is None
    assert status.markdown == "Total due: 1,284.50 EUR"
    assert [b.status for b in status.pages[0].blocks] == ["ok", "failed"]


def test_document_submit_sends_upright_not_model():
    fake = Fake([{"id": "job_9", "status": "Running", "pages_done": 0, "pages_total": 1}])
    job_id = document_client(fake).submit_job(PNG, name="page.png", upright=False)
    assert job_id == "job_9"
    assert fake.calls[0]["url"].endswith("/api/document-parse/jobs")
    assert b'name="upright"' in fake.calls[0]["body"]
    assert b"false" in fake.calls[0]["body"]
    assert b'name="model"' not in fake.calls[0]["body"]


def test_ocr_read_document_image_goes_through_job(tmp_path, monkeypatch):
    page = tmp_path / "page.png"
    page.write_bytes(PNG)
    fake = Fake(
        [
            {"id": "job_1", "status": "Running", "pages_done": 0, "pages_total": 1, "pages": []},
            {"id": "job_1", "status": "Success", "pages_done": 1, "pages_total": 1, "pages": [PAGE]},
        ]
    )
    local = Backend(name="local-document", base_url="http://127.0.0.1:8966", api="document")
    client = document_client(fake)
    monkeypatch.setattr(tools, "_client", lambda: (client, local))
    monkeypatch.setattr(client, "wait", lambda job_id, timeout: client.job(job_id))

    result = json.loads(tools.ocr_read({"path": str(page), "queue": False, "upright": False}))
    assert result["ok"] is True
    assert result["backend"] == "local-document"
    assert result["blocks_available"] is True
    assert result["regions"] == 2
    assert result["failed_regions"] == 1
    assert fake.calls[0]["method"] == "POST"
    assert fake.calls[1]["method"] == "GET"


def test_jev_rm_uses_local_render_score_endpoint():
    payload = {
        "results": [
            {
                "case_id": "0",
                "category": "text",
                "verdict": "scored",
                "prob_yes": 0.97,
                "prob_no": 0.03,
                "detail": "",
            }
        ],
        "summary": {"threshold_used": 0.6},
    }
    fake = Fake([payload])
    client = document_client(fake, jev_base_url="http://127.0.0.1:18500")
    result = client.jev_score(PNG, "1,284.50", category="text", threshold=0.6)
    assert fake.calls[0]["url"] == "http://127.0.0.1:18500/v1/rm/score"
    sent = json.loads(fake.calls[0]["body"])
    assert sent["input_form"] == "ternary"
    assert sent["items"][0]["ocr_text"] == "1,284.50"
    assert not sent["items"][0]["image"].startswith("data:")
    assert result.scored is True
    assert result.passed is True
    assert result.score == 0.97


def test_jev_rm_failure_verdict_is_not_a_pass():
    payload = {
        "results": [
            {
                "case_id": "0",
                "category": "text",
                "verdict": "forward_error",
                "prob_yes": -1.0,
                "prob_no": 2.0,
                "detail": "backend not ready",
            }
        ],
        "summary": {"threshold_used": 0.6},
    }
    result = document_client(Fake([payload]), jev_base_url="http://127.0.0.1:18500").jev_score(
        PNG, "x"
    )
    assert result.scored is False
    assert result.passed is False
    assert result.detail == "backend not ready"


def test_settings_use_keyed_get_config():
    calls = []

    class Ctx:
        def get_config(self, key, default=None):
            calls.append(key)
            return {"base_url": "http://127.0.0.1:8966", "local_api": "document"}.get(key, default)

    tools.set_context(Ctx())
    try:
        settings = tools._settings()
    finally:
        tools.set_context(None)
    assert settings["base_url"] == "http://127.0.0.1:8966"
    assert settings["local_api"] == "document"
    assert "threshold" in calls and "api_key" not in calls


def test_skipped_regions_are_reported_separately(tmp_path, monkeypatch):
    page = tmp_path / "page.png"
    page.write_bytes(PNG)
    response = {
        "id": "job_1",
        "status": "Success",
        "pages_done": 1,
        "pages_total": 1,
        "pages": [
            {
                **PAGE,
                "minicpm_ocr": {
                    **PAGE["minicpm_ocr"],
                    "blocks": [
                        {
                            "order": 0,
                            "label": "image",
                            "bbox": [0, 0, 10, 10],
                            "content": "",
                            "status": "skipped_no_vlm",
                        },
                        *PAGE["minicpm_ocr"]["blocks"],
                    ],
                },
            }
        ],
    }
    fake = Fake([response])
    local = Backend(name="local-document", base_url="http://127.0.0.1:8966", api="document")
    client = document_client(fake)
    monkeypatch.setattr(tools, "_client", lambda: (client, local))
    monkeypatch.setattr(client, "wait", lambda job_id, timeout: client.job(job_id))

    result = json.loads(tools.ocr_read({"path": str(page), "queue": False}))
    assert result["regions"] == 3
    assert result["failed_regions"] == 1
    assert result["skipped_regions"] == 1
    assert result["failed"][0]["label"] == "table"


def test_web_demo_port_infers_document_surface():
    backend = candidates({"base_url": "http://127.0.0.1:8966"})[0]
    assert backend.surface == "document"
    assert backend.blocks is True
