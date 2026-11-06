"""Backend discovery, probing, and the surface each URL implies."""

from __future__ import annotations

from minicpm_ocr.core import probe as probe_mod
from minicpm_ocr.core.errors import BackendUnavailable
from minicpm_ocr.core.probe import Backend, candidates, doctor, probe, reachable


def test_surface_inference():
    assert Backend(name="cloud", base_url="https://api.modelbest.ai").surface == "modelbest"
    assert Backend(name="webdemo", base_url="http://127.0.0.1:8966").surface == "document"
    assert Backend(name="vllm", base_url="http://127.0.0.1:8000").surface == "chat"
    assert Backend(name="explicit", base_url="http://127.0.0.1:8000", api="document").surface == "document"


def test_cloud_and_local_flags():
    cloud = Backend(name="cloud", base_url="https://api.modelbest.ai", api="modelbest")
    assert cloud.blocks and cloud.jobs and cloud.resolved_kind == "modelbest"

    document = Backend(name="doc", base_url="http://127.0.0.1:8966", api="document")
    assert document.blocks and document.jobs

    chat = Backend(name="chat", base_url="http://127.0.0.1:18499", api="chat")
    assert not chat.blocks and not chat.jobs


def test_candidates_modes_and_settings_are_passed_through():
    cloud = candidates({"backend": "cloud", "_api_key": "k", "parser_model": "m", "jev_api": "rm"})
    assert [b.name for b in cloud] == ["modelbest"]
    assert cloud[0].model == "m" and cloud[0].jev_api == "rm"

    local = candidates({"backend": "local", "local_api": "document"})
    assert [b.name for b in local] == ["local-document", "local-document-docker"]
    assert all(b.surface == "document" for b in local)

    configured = candidates(
        {
            "base_url": "http://127.0.0.1:18499/",
            "local_api": "chat",
            "parser_model": "minicpm-ocr",
            "jev_base_url": "http://127.0.0.1:18500/",
        }
    )[0]
    assert configured.name == "local-chat"
    assert configured.base_url == "http://127.0.0.1:18499"
    assert configured.jev_base_url == "http://127.0.0.1:18500"
    assert configured.model == "minicpm-ocr"


def test_auto_mode_prefers_explicit_base_then_cloud_then_loopback():
    names = [
        b.name
        for b in candidates(
            {"base_url": "http://127.0.0.1:18499", "_api_key": "k", "backend": "auto"}
        )
    ]
    assert names[0] == "configured"
    assert "modelbest" in names
    assert "local-vllm" in names


def test_reachable_only_checks_loopback():
    assert reachable(Backend(name="public", base_url="https://example.com")) is True
    assert reachable(Backend(name="closed", base_url="http://127.0.0.1:1"), timeout=0.2) is False


def test_probe_document_uses_health(monkeypatch):
    monkeypatch.setattr(probe_mod, "reachable", lambda backend: True)
    monkeypatch.setattr(
        probe_mod.OcrClient, "health", lambda self, timeout=8.0: {"status": "ok", "backend": "x"}
    )
    report = probe(Backend(name="doc", base_url="http://127.0.0.1:8966", api="document"))
    assert report["ok"] is True
    assert report["surface"] == "document"
    assert report["blocks"] is True and report["jobs"] is True
    assert report["health"]["backend"] == "x"


def test_probe_document_reports_unhealthy(monkeypatch):
    monkeypatch.setattr(probe_mod, "reachable", lambda backend: True)
    monkeypatch.setattr(probe_mod.OcrClient, "health", lambda self, timeout=8.0: {"status": "loading"})
    report = probe(Backend(name="doc", base_url="http://127.0.0.1:8966", api="document"))
    assert report["ok"] is False
    assert "loading" in report["error"]


def test_probe_models_detects_parser_and_jev(monkeypatch):
    monkeypatch.setattr(probe_mod, "reachable", lambda backend: True)
    monkeypatch.setattr(
        probe_mod.OcrClient,
        "models",
        lambda self, timeout=8.0: ["minicpm-ocr", "minicpm-ocr-jev"],
    )
    report = probe(
        Backend(
            name="chat",
            base_url="http://127.0.0.1:18499",
            api="chat",
            model="minicpm-ocr",
            jev_model="minicpm-ocr-jev",
        )
    )
    assert report["ok"] is True
    assert report["has_parser"] is True and report["has_jev"] is True
    assert report["models"] == ["minicpm-ocr", "minicpm-ocr-jev"]


def test_probe_reports_connection_error(monkeypatch):
    monkeypatch.setattr(probe_mod, "reachable", lambda backend: True)

    def fail(self, timeout=8.0):
        raise BackendUnavailable("connection refused")

    monkeypatch.setattr(probe_mod.OcrClient, "models", fail)
    report = probe(Backend(name="chat", base_url="http://127.0.0.1:18499", api="chat"))
    assert report["ok"] is False
    assert report["code"] == "backend_unavailable"


def test_doctor_summarises_usable_and_missing(monkeypatch):
    monkeypatch.setattr(
        probe_mod,
        "probe",
        lambda backend, timeout=8.0: {"name": backend.name, "base_url": backend.base_url, "ok": False},
    )
    result = doctor({"backend": "local", "local_api": "chat"})
    assert result["ok"] is False
    assert result["usable"] is None
    assert "ModelBest API key" in result["hint"]

    monkeypatch.setattr(
        probe_mod,
        "probe",
        lambda backend, timeout=8.0: {"name": backend.name, "base_url": backend.base_url, "ok": True},
    )
    result = doctor({"backend": "local", "local_api": "chat"})
    assert result["ok"] is True
    assert result["usable"] == "local-vllm"


def test_client_receives_surface_and_jev_api():
    backend = Backend(
        name="doc",
        base_url="http://127.0.0.1:8966",
        api="document",
        jev_base_url="http://127.0.0.1:18500",
        jev_api="rm",
    )
    client = backend.client()
    assert client.surface == "document"
    assert client.jev_api == "rm"
    assert client.jobs_path == "/api/document-parse/jobs"


def test_unknown_settings_do_not_break_candidates():
    assert candidates({})[0].name == "local-vllm"
