"""Source resolution: local files only, or a public https URL — never an internal one."""

from __future__ import annotations

import pytest

from minicpm_ocr.core import sources
from minicpm_ocr.core.errors import RequestRejected


def test_private_and_loopback_ranges_are_detected():
    assert sources._is_private_address("127.0.0.1")
    assert sources._is_private_address("10.0.0.5")
    assert sources._is_private_address("192.168.1.10")
    assert sources._is_private_address("169.254.169.254")  # cloud metadata
    assert sources._is_private_address("::1")
    assert not sources._is_private_address("8.8.8.8")


def test_ensure_public_refuses_loopback_and_private():
    with pytest.raises(RequestRejected):
        sources.ensure_public("https://127.0.0.1/secret.png")
    with pytest.raises(RequestRejected):
        sources.ensure_public("https://10.0.0.5/secret.png")
    with pytest.raises(RequestRejected):
        sources.ensure_public("https://localhost/secret.png")


def test_plain_http_is_refused_before_any_request():
    with pytest.raises(RequestRejected) as exc:
        sources.read_source("http://example.com/invoice.png")
    assert "https" in str(exc.value)


def test_local_file_roundtrip(tmp_path):
    path = tmp_path / "page.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)

    data, name = sources.read_source(str(path))
    assert name == "page.png"
    assert data.startswith(b"\x89PNG")


def test_missing_file_is_a_clear_error(tmp_path):
    with pytest.raises(RequestRejected) as exc:
        sources.read_source(str(tmp_path / "nope.pdf"))
    assert "no such file" in str(exc.value)


class _Response:
    def __init__(self, data: bytes, headers=None):
        self._data = data
        self.headers = headers or {}

    def read(self, n: int = -1):
        return self._data[:n] if n >= 0 else self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Opener:
    def __init__(self, response):
        self.response = response

    def open(self, req, timeout=None):
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_fetch_returns_bytes_and_name(monkeypatch):
    monkeypatch.setattr(sources, "ensure_public", lambda url: None)
    monkeypatch.setattr(sources, "_OPENER", _Opener(_Response(b"png", {"Content-Length": "3"})))
    data, name = sources._fetch("https://example.com/docs/page.png", timeout=1)
    assert data == b"png"
    assert name == "page.png"


def test_fetch_rejects_declared_and_actual_oversize(monkeypatch):
    monkeypatch.setattr(sources, "ensure_public", lambda url: None)
    monkeypatch.setattr(sources, "MAX_ORIGINAL_BYTES", 4)
    monkeypatch.setattr(sources, "_OPENER", _Opener(_Response(b"x", {"Content-Length": "5"})))
    with pytest.raises(RequestRejected):
        sources._fetch("https://example.com/big.png", timeout=1)

    monkeypatch.setattr(sources, "_OPENER", _Opener(_Response(b"12345", {})))
    with pytest.raises(RequestRejected):
        sources._fetch("https://example.com/big.png", timeout=1)


def test_fetch_maps_http_and_transport_errors(monkeypatch):
    import urllib.error

    monkeypatch.setattr(sources, "ensure_public", lambda url: None)
    monkeypatch.setattr(sources, "_OPENER", _Opener(urllib.error.HTTPError(
        "https://example.com/x", 404, "not found", {}, None
    )))
    with pytest.raises(RequestRejected):
        sources._fetch("https://example.com/x.png", timeout=1)

    monkeypatch.setattr(sources, "_OPENER", _Opener(urllib.error.URLError("offline")))
    with pytest.raises(Exception) as exc:
        sources._fetch("https://example.com/x.png", timeout=1)
    assert "offline" in str(exc.value)


def test_ensure_public_rejects_bad_urls(monkeypatch):
    with pytest.raises(RequestRejected):
        sources.ensure_public("https://")
    def no_dns(*args, **kwargs):
        raise OSError("dns")

    monkeypatch.setattr(sources.socket, "getaddrinfo", no_dns)
    with pytest.raises(RequestRejected):
        sources.ensure_public("https://unresolvable.invalid/x.png")
