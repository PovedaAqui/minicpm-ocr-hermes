"""Local / cloud parity: the same tools, two very different backends."""

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

CHAT = {"id": "chatcmpl-1", "choices": [{"message": {"content": "Total due: 1,284.50 EUR  \n"}}]}


class Fake:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def __call__(self, method, url, headers, body, timeout):
        self.calls.append({"url": url, "body": body})
        return 200, {}, json.dumps(self.payload).encode()


def test_configured_local_backend_has_no_blocks():
    local = candidates({"base_url": "http://127.0.0.1:8000"})[0]
    assert local.resolved_kind == "openai"
    assert local.blocks is False

    cloud = candidates({"base_url": "https://api.modelbest.ai"})[0]
    assert cloud.resolved_kind == "modelbest"
    assert cloud.blocks is True


def test_mode_local_excludes_the_cloud_key():
    names = [b.name for b in candidates({"backend": "local", "_api_key": "k"})]
    assert "modelbest" not in names
    assert names == ["local-vllm", "local-ollama"]


def test_mode_cloud_excludes_loopback():
    names = [b.name for b in candidates({"backend": "cloud", "_api_key": "k"})]
    assert names == ["modelbest"]


def test_parse_page_uses_chat_completions_and_returns_text():
    fake = Fake(CHAT)
    client = OcrClient(base_url="http://127.0.0.1:8000", api_key="", transport=fake)
    result = client.parse_page(PNG, name="page.png")

    assert fake.calls[0]["url"].endswith("/v1/chat/completions")
    sent = json.loads(fake.calls[0]["body"])
    assert sent["model"] == "MiniCPM-OCR-1.0"
    assert sent["messages"][0]["content"][1]["text"] == "Text Recognition:"
    assert sent["stream"] is False
    assert result.pages[0].markdown == "Total due: 1,284.50 EUR"
    assert result.pages[0].blocks == ()


def test_ocr_read_on_local_backend_reports_no_blocks(tmp_path, monkeypatch):
    page = tmp_path / "page.png"
    page.write_bytes(PNG)

    fake = Fake(CHAT)
    local = Backend(name="local-vllm", base_url="http://127.0.0.1:8000", kind="openai")
    monkeypatch.setattr(tools, "_client", lambda: (OcrClient(transport=fake), local))

    result = json.loads(tools.ocr_read({"path": str(page), "queue": False}))
    assert result["ok"] is True
    assert result["backend"] == "local-vllm"
    assert result["blocks_available"] is False
    assert "Total due" in result["preview"]
    assert "no layout stage" in result["note"]


def test_ocr_read_refuses_pdf_on_local_backend(tmp_path, monkeypatch):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n1 0 obj <</Type /Page>> endobj\ntrailer")

    local = Backend(name="local-vllm", base_url="http://127.0.0.1:8000", kind="openai")
    monkeypatch.setattr(tools, "_client", lambda: (OcrClient(transport=Fake(CHAT)), local))

    result = json.loads(tools.ocr_read({"path": str(pdf), "queue": False}))
    assert result["ok"] is False
    assert result["error"]["code"] == "request_rejected"
    assert "images only" in result["error"]["message"]
