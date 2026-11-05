"""Client behaviour against a fake transport — no network, no key."""

from __future__ import annotations

import json

import pytest

from minicpm_ocr.core.client import OcrClient
from minicpm_ocr.core.errors import AuthError, BackendUnavailable, PayloadTooLarge, RateLimited


def png(width: int = 20, height: int = 10) -> bytes:
    import struct

    return (
        b"\x89PNG\r\n\x1a\n"
        + b"\x00\x00\x00\rIHDR"
        + struct.pack(">II", width, height)
        + b"\x08\x02\x00\x00\x00"
        + b"\x00" * 8
    )


PAGE = {
    "metadata": {"page_index": 0},
    "choices": [{"message": {"content": "# Invoice\n\nTotal due: 1,284.50 EUR"}}],
    "ocr": {
        "image_width": 900,
        "image_height": 1200,
        "blocks": [
            {
                "order": 0,
                "label": "doc_title",
                "bbox": [60, 80, 400, 120],
                "content": "Invoice",
                "status": "ok",
            },
            {
                "order": 1,
                "label": "text",
                "bbox": [60, 160, 500, 180],
                "content": "Total due: 1,284.50",
                "status": "ok",
            },
            {
                "order": 2,
                "label": "table",
                "bbox": [60, 300, 800, 600],
                "content": "",
                "status": "failed",
            },
        ],
        "rotation": {"applied": False, "enabled": True, "total_angle": 0},
    },
    "request_id": "req_1",
    "result_status": "ok",
}


class Fake:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, method, url, headers, body, timeout):
        self.calls.append(
            {"method": method, "url": url, "headers": dict(headers), "body": body, "timeout": timeout}
        )
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, Exception):
            raise item
        return item


def json_response(payload, status=200):
    return status, {}, json.dumps(payload).encode()


def client(transport, **kwargs):
    kwargs.setdefault("api_key", "test-key")
    kwargs.setdefault("sleep", lambda _s: None)
    return OcrClient(transport=transport, **kwargs)


def test_parse_image_shapes_pages_and_blocks():
    fake = Fake(json_response(PAGE))
    result = client(fake).parse_image(png(), name="page.png")

    assert fake.calls[0]["url"].endswith("/v1/document-parse")
    assert fake.calls[0]["headers"]["Authorization"] == "Bearer test-key"
    assert len(result.pages) == 1
    page = result.pages[0]
    assert page.markdown.startswith("# Invoice")
    assert [b.label for b in page.blocks] == ["doc_title", "text", "table"]
    assert page.blocks[1].bbox == (60, 160, 500, 180)
    assert len(page.failed) == 1 and page.failed[0].status == "failed"
    assert result.request_id == "req_1"


def test_parse_image_refuses_before_network_when_too_large(monkeypatch):
    from minicpm_ocr.core import limits

    monkeypatch.setattr(limits, "MAX_IMAGE_BYTES", 10)
    fake = Fake(json_response(PAGE))
    with pytest.raises(PayloadTooLarge):
        client(fake).parse_image(b"x" * 11, name="big.png")
    assert fake.calls == []


def test_auth_error_is_not_retried():
    fake = Fake((401, {}, b'{"error": {"message": "invalid api key"}}'))
    with pytest.raises(AuthError) as exc:
        client(fake).parse_image(png(), name="page.png")
    assert "invalid api key" in str(exc.value)
    assert len(fake.calls) == 1


def test_retry_after_503_then_success():
    slept = []
    fake = Fake((503, {}, b"upstream busy"), json_response(PAGE))
    result = client(fake, max_attempts=3, sleep=slept.append).parse_image(png(), name="page.png")
    assert len(fake.calls) == 2
    assert len(slept) == 1 and slept[0] > 0
    assert result.pages[0].markdown


def test_rate_limit_carries_retry_after():
    fake = Fake((429, {"Retry-After": "7"}, b'{"error": {"message": "slow down"}}'))
    with pytest.raises(RateLimited) as exc:
        client(fake, max_attempts=1).parse_image(png(), name="page.png")
    assert exc.value.retry_after == 7.0


def test_connection_error_becomes_backend_unavailable():
    import urllib.error

    fake = Fake(urllib.error.URLError("connection refused"))
    with pytest.raises(BackendUnavailable):
        client(fake, max_attempts=1).parse_image(png(), name="page.png")


def test_jev_score_passes_above_threshold():
    payload = {
        "results": [
            {
                "case_id": "0",
                "category": "text",
                "verdict": "scored",
                "prob_yes": 0.93,
                "prob_no": 0.07,
                "detail": "",
            }
        ],
        "summary": {"threshold_used": 0.6, "total": 1, "scored": 1, "pass_rate": 1.0},
    }
    fake = Fake(json_response(payload))
    result = client(fake).jev_score(png(), "Total due: 1,284.50", category="text", threshold=0.6)
    assert result.score == pytest.approx(0.93)
    assert result.passed is True
    sent = json.loads(fake.calls[0]["body"])
    assert sent["model"] == "MiniCPM-OCR-Jev"
    assert sent["minicpm_ocr_jev"]["category"] == "text"
    assert sent["stream"] is False


def test_jev_uses_server_threshold():
    payload = {
        "results": [{"prob_yes": 0.55, "prob_no": 0.45, "verdict": "scored", "category": "text"}],
        "summary": {"threshold_used": 0.75},
    }
    result = client(Fake(json_response(payload))).jev_score(png(), "x", threshold=0.6)
    assert result.threshold == pytest.approx(0.75)
    assert result.passed is False


def test_submit_job_uploads_multipart_and_returns_id():
    fake = Fake((202, {}, b'{"id": "ocr_123", "status": "Running", "pages_done": 0, "pages_total": 0}'))
    job_id = client(fake).submit_job(png(), name="page.png")
    assert job_id == "ocr_123"
    body = fake.calls[0]["body"]
    assert b'filename="page.png"' in body
    assert b"MiniCPM-OCR-1.0" in body
    assert fake.calls[0]["headers"]["Content-Type"].startswith("multipart/form-data; boundary=")


def test_job_status_terminal_fraction_and_pages():
    payload = {
        "id": "ocr_123",
        "status": "Success",
        "pages_done": 2,
        "pages_total": 2,
        "pages": [PAGE, {**PAGE, "metadata": {"page_index": 1}}],
        "markdown": "line one\n\nline two",
    }
    status = client(Fake(json_response(payload))).job("ocr_123")
    assert status.terminal and status.ok
    assert status.fraction == 1.0
    assert len(status.pages) == 2
    assert status.pages[1].index == 1
    assert status.markdown.startswith("line one")


def test_job_running_is_not_terminal():
    payload = {"id": "j", "status": "Running", "pages_done": 1, "pages_total": 4}
    status = client(Fake(json_response(payload))).job("j")
    assert status.terminal is False
    assert status.fraction == 0.25


def test_wait_polls_until_terminal():
    running = {"id": "j", "status": "Running", "pages_done": 1, "pages_total": 2}
    done = {"id": "j", "status": "Success", "pages_done": 2, "pages_total": 2, "pages": [PAGE]}
    fake = Fake(json_response(running), json_response(done))
    ticks = iter([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    status = client(fake).wait("j", timeout=10, sleep=lambda _s: None, clock=lambda: next(ticks))
    assert status.ok
    assert len(fake.calls) == 2
