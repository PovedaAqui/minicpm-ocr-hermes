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
