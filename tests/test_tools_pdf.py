"""PDF job ergonomics: a slow job returns a handle, it does not fail the call."""

from __future__ import annotations

import json

from minicpm_ocr import tools
from minicpm_ocr.core.client import JobStatus, Page
from minicpm_ocr.core.probe import Backend

PDF = b"%PDF-1.4\n1 0 obj <</Type /Page>> endobj\ntrailer"


class FakeClient:
    def __init__(self, status: JobStatus | None = None) -> None:
        self.status = status
        self.waited = False

    def submit_job(self, data: bytes, *, name: str) -> str:
        return "job_1"

    def wait(self, job_id: str, *, timeout: float) -> JobStatus:
        self.waited = True
        assert self.status is not None
        return self.status


def _backend() -> Backend:
    return Backend(name="modelbest", base_url="https://api.modelbest.ai", kind="modelbest")


def test_pdf_with_wait_false_returns_the_handle(tmp_path, monkeypatch):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(PDF)
    client = FakeClient()
    monkeypatch.setattr(tools, "_client", lambda: (client, _backend()))

    result = json.loads(tools.ocr_read({"path": str(pdf), "wait": False, "queue": False}))
    assert result["ok"] is True
    assert result["job_id"] == "job_1"
    assert result["status"] == "submitted"
    assert client.waited is False


def test_slow_pdf_is_not_an_error(tmp_path, monkeypatch):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(PDF)
    client = FakeClient(JobStatus(id="job_1", status="Running", pages_done=3, pages_total=40))
    monkeypatch.setattr(tools, "_client", lambda: (client, _backend()))

    result = json.loads(tools.ocr_read({"path": str(pdf), "queue": False}))
    assert result["ok"] is True
    assert result["job_id"] == "job_1"
    assert result["status"] == "Running"
    assert result["pages_done"] == 3
    assert "ocr_job" in result["next"]


def test_finished_pdf_returns_pages(tmp_path, monkeypatch):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(PDF)
    page = Page(index=0, markdown="Total due: 1,284.50 EUR", blocks=())
    client = FakeClient(
        JobStatus(
            id="job_1",
            status="Success",
            pages_done=2,
            pages_total=2,
            pages=(page,),
            markdown=page.markdown,
        )
    )
    monkeypatch.setattr(tools, "_client", lambda: (client, _backend()))

    result = json.loads(tools.ocr_read({"path": str(pdf), "mode": "markdown", "queue": False}))
    assert result["ok"] is True
    assert result["pages"] == 1
    assert "1,284.50" in result["markdown"]
    assert result["blocks_available"] is True


def test_failed_job_is_an_error(tmp_path, monkeypatch):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(PDF)
    client = FakeClient(JobStatus(id="job_1", status="Failed", pages_done=0, pages_total=2))
    monkeypatch.setattr(tools, "_client", lambda: (client, _backend()))

    result = json.loads(tools.ocr_read({"path": str(pdf), "queue": False}))
    assert result["ok"] is False
    assert result["error"]["code"] == "parse_failed"
