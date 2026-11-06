"""Client edges: discovery responses, guards, malformed payloads, retries, Jev failures."""

from __future__ import annotations

import json
import urllib.error

import pytest

from minicpm_ocr.core import client as client_mod
from minicpm_ocr.core.client import OcrClient
from minicpm_ocr.core.errors import BackendUnavailable, ParseFailed

PNG = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR"
    b"\x00\x00\x00\x14\x00\x00\x00\x0a"
    b"\x08\x02\x00\x00\x00"
    b"\x00" * 8
)


class Fake:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, method, url, headers, body, timeout):
        self.calls.append((method, url, body))
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, Exception):
            raise item
        if isinstance(item, dict):
            return 200, {}, json.dumps(item).encode()
        return item


def make(transport, **kwargs):
    kwargs.setdefault("api_key", "k")
    kwargs.setdefault("max_attempts", 1)
    kwargs.setdefault("sleep", lambda _s: None)
    return OcrClient(transport=transport, **kwargs)


def test_models_parses_strings_dicts_and_mapping():
    payload = {"data": ["one", {"id": "two"}, {"nope": 1}, 3]}
    assert make(Fake(payload)).models() == ["one", "two"]
    assert make(Fake({"models": [{"id": "three"}]})).models() == ["three"]
    assert make(Fake({"data": {"data": [{"id": "four"}]}})).models() == ["four"]


def test_health_returns_json():
    assert make(Fake({"status": "ok"})).health() == {"status": "ok"}


def test_parse_guards_match_surface():
    with pytest.raises(ParseFailed, match="no synchronous"):
        make(Fake({"choices": []}), surface="document").parse_image(PNG)
    with pytest.raises(ParseFailed, match="not an OpenAI"):
        make(Fake({"choices": []}), surface="modelbest").parse_page(PNG)


def test_parse_page_without_choices_fails():
    with pytest.raises(ParseFailed, match="no choices"):
        make(Fake({"choices": []}), surface="chat").parse_page(PNG)


def test_parse_image_reads_nested_request_id_and_issues():
    payload = {
        "choices": [{"message": {"content": "page"}}],
        "ocr": {"request_id": "req-nested", "blocks": []},
        "issues": ["warn"],
        "result_status": "ok",
    }
    result = make(Fake(payload)).parse_image(PNG)
    assert result.request_id == "req-nested"
    assert result.issues == ("warn",)
    assert result.pages[0].markdown == "page"


def test_submit_job_without_id_fails():
    with pytest.raises(ParseFailed, match="no job id"):
        make(Fake({"status": "Running"})).submit_job(PNG, name="page.png")


def test_cancel_empty_204_is_cancelled():
    status = make(Fake((204, {}, b""))).cancel_job("job-1")
    assert status.status == "cancelled"
    assert status.terminal is True
    assert status.fraction is None


def test_non_json_and_non_object_bodies_fail():
    with pytest.raises(ParseFailed, match="non-JSON"):
        make(Fake((200, {}, b"<html>"))).models()
    with pytest.raises(ParseFailed, match="not an object"):
        make(Fake((200, {}, b"[1, 2]"))).models()


def test_page_from_tolerates_malformed_blocks():
    payload = {
        "choices": [{"message": {"content": "md"}}],
        "ocr": {
            "blocks": [
                "junk",
                {"order": 1, "label": "text", "content": "ok", "status": "ok", "bbox": ["x", 2, 3, 4]},
                {"order": "2", "label": "table", "content": "", "status": "failed", "score": 0.25},
            ]
        },
    }
    page = client_mod._page_from(payload)
    assert len(page.blocks) == 2
    assert page.blocks[0].bbox is None
    assert page.blocks[1].score == 0.25
    assert len(page.failed) == 1


def test_job_status_falls_back_to_page_markdown():
    payload = {
        "id": "j",
        "status": "Success",
        "pages": [
            {"choices": [{"message": {"content": "one"}}], "ocr": {"blocks": []}},
            {"choices": [{"message": {"content": "two"}}], "ocr": {"blocks": []}},
        ],
    }
    status = client_mod._job_status(payload, "fallback")
    assert status.markdown == "one\n\ntwo"
    assert status.fraction is None
    assert status.pages_done == 0


def test_job_id_nested_and_retry_after_variants():
    assert client_mod._job_id({"data": {"job_id": "nested"}}) == "nested"
    assert client_mod._job_id({"nothing": 1}) is None
    assert client_mod._retry_after({"retry-after": "2"}) == 2.0
    assert client_mod._retry_after({"Retry-After": "soon"}) is None
    assert client_mod._retry_after({"Retry-After": "-3"}) == 0.0


def test_retry_exhaustion_raises_backend_unavailable():
    fake = Fake((503, {}, b"busy"))
    with pytest.raises(BackendUnavailable):
        OcrClient(transport=fake, max_attempts=2, sleep=lambda _s: None).models()
    assert len(fake.calls) == 2


def test_timeout_then_success():
    fake = Fake(TimeoutError("slow"), {"data": []})
    assert OcrClient(transport=fake, max_attempts=2, sleep=lambda _s: None).models() == []
    assert len(fake.calls) == 2


def test_wait_returns_running_after_deadline():
    running = {"id": "j", "status": "Running", "pages_done": 1, "pages_total": 4}
    fake = Fake(running)
    ticks = iter([0.0, 0.1, 2.0])
    status = make(fake).wait("j", timeout=1, sleep=lambda _s: None, clock=lambda: next(ticks))
    assert status.ok is False
    assert status.terminal is False
    assert status.fraction == 0.25


def test_jev_rejects_unknown_category():
    with pytest.raises(ParseFailed, match="category"):
        make(Fake({"results": []})).jev_score(PNG, "text", category="drawing")


def test_jev_no_results_fails_in_both_modes():
    with pytest.raises(ParseFailed, match="no results"):
        make(Fake({"results": []})).jev_score(PNG, "text")
    with pytest.raises(ParseFailed, match="no results"):
        make(Fake({"results": []}), jev_api="rm", base_url="http://127.0.0.1:1").jev_score(
            PNG, "text"
        )


def test_urllib_transport_maps_http_and_url_errors(monkeypatch):
    class Resp:
        status = 200
        headers = {"X-Test": "1"}

        def read(self):
            return b"body"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(client_mod.urllib.request, "urlopen", lambda req, timeout: Resp())
    assert client_mod.urllib_transport("GET", "https://example.com/x", {}, None, 1) == (
        200,
        {"X-Test": "1"},
        b"body",
    )

    def http_error(req, timeout):
        raise urllib.error.HTTPError("https://example.com/x", 404, "missing", {"A": "b"}, None)

    monkeypatch.setattr(client_mod.urllib.request, "urlopen", http_error)
    status, headers, body = client_mod.urllib_transport("GET", "https://example.com/x", {}, None, 1)
    assert status == 404 and headers == {"A": "b"}

    def url_error(req, timeout):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(client_mod.urllib.request, "urlopen", url_error)
    with pytest.raises(BackendUnavailable, match="example.com"):
        client_mod.urllib_transport("GET", "https://example.com/x", {}, None, 1)
