"""Limit and sniffing behaviour."""

from __future__ import annotations

import struct

import pytest

from minicpm_ocr.core import limits
from minicpm_ocr.core.errors import PageLimitExceeded, PayloadTooLarge


def png(width: int, height: int) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        + b"\x00\x00\x00\rIHDR"
        + struct.pack(">II", width, height)
        + b"\x08\x02\x00\x00\x00"
        + b"\x00" * 8
    )


def test_sniff_common_formats():
    assert limits.sniff_image(png(4, 4)) == "png"
    assert limits.sniff_image(b"\xff\xd8\xff\xe0" + b"0" * 20) == "jpeg"
    assert limits.sniff_image(b"GIF89a" + b"0" * 20) == "gif"
    assert limits.sniff_image(b"BM" + b"0" * 30) == "bmp"
    assert limits.sniff_image(b"RIFF0000WEBP" + b"0" * 20) == "webp"
    assert limits.sniff_image(b"nope") is None


def test_png_dimensions_and_mime():
    data = png(900, 1200)
    assert limits.image_size(data) == (900, 1200)
    assert limits.mime_for(data) == "image/png"


def test_oversized_image_is_refused():
    with pytest.raises(PayloadTooLarge):
        limits.check_image(b"\x00" * (limits.MAX_IMAGE_BYTES + 1), "huge.png")


def test_huge_canvas_is_refused():
    with pytest.raises(PayloadTooLarge):
        limits.check_image(png(9000, 9000), "poster.png")


def test_pdf_page_count_and_limit(monkeypatch):
    pdf = b"%PDF-1.4\n" + b"1 0 obj <</Type /Page>> endobj\n" * 3 + b"trailer"
    assert limits.count_pdf_pages(pdf) == 3
    monkeypatch.setattr(limits, "MAX_PAGES", 2)
    with pytest.raises(PageLimitExceeded):
        limits.check_pdf(pdf, "big.pdf")


def test_page_type_does_not_count_pages_tree():
    pdf = b"%PDF-1.4\n1 0 obj <</Type /Pages /Count 3>> endobj\n2 0 obj <</Type /Page>> endobj"
    assert limits.count_pdf_pages(pdf) == 1


def test_check_document_routes_by_magic():
    assert limits.check_document(png(10, 10), "a.png") == "image"
    assert limits.check_document(b"%PDF-1.7\n1 0 obj <</Type /Page>> endobj", "a.pdf") == "pdf"
    with pytest.raises(PayloadTooLarge):
        limits.check_document(b"just text", "notes.txt")


def test_image_size_across_formats():
    jpeg = (
        b"\xff\xd8"
        + b"\xff\xc0"
        + (17).to_bytes(2, "big")
        + b"\x08"
        + (10).to_bytes(2, "big")
        + (20).to_bytes(2, "big")
        + b"\x03" + b"\x00" * 9
    )
    assert limits.image_size(jpeg) == (20, 10)

    gif = b"GIF89a" + (30).to_bytes(2, "little") + (40).to_bytes(2, "little") + b"\x00" * 20
    assert limits.image_size(gif) == (30, 40)

    bmp = (
        b"BM"
        + b"\x00" * 16
        + (50).to_bytes(4, "little", signed=True)
        + (60).to_bytes(4, "little", signed=True)
    )
    assert limits.image_size(bmp) == (50, 60)

    webp = b"RIFF" + b"\x00" * 4 + b"WEBP" + b"VP8X" + b"\x00" * 8
    webp += (99).to_bytes(3, "little") + (101).to_bytes(3, "little") + b"\x00" * 2
    assert limits.image_size(webp) == (100, 102)


def test_image_size_returns_none_for_unknown_or_truncated():
    assert limits.image_size(b"not-an-image") is None
    assert limits.image_size(b"\x89PNG\r\n\x1a\n" + b"\x00") is None
    assert limits.mime_for(b"not-an-image") == "application/octet-stream"


def test_image_dimension_side_limit():
    with pytest.raises(PayloadTooLarge):
        limits.check_image(png(10, 9000), "tall.png")


def test_pdf_size_limit(monkeypatch):
    monkeypatch.setattr(limits, "MAX_ORIGINAL_BYTES", 10)
    with pytest.raises(PayloadTooLarge):
        limits.check_pdf(b"%PDF-1.4" + b"x" * 20, "big.pdf")


def test_file_uri_roundtrip(tmp_path):
    from minicpm_ocr.core import sources

    path = tmp_path / "page.png"
    path.write_bytes(png(4, 4))
    data, name = sources.read_source(path.as_uri())
    assert data.startswith(b"\x89PNG") and name == "page.png"


def test_source_size_limit(monkeypatch, tmp_path):
    from minicpm_ocr.core import sources
    from minicpm_ocr.core.errors import RequestRejected

    monkeypatch.setattr(sources, "MAX_ORIGINAL_BYTES", 4)
    path = tmp_path / "page.png"
    path.write_bytes(b"12345")
    with pytest.raises(RequestRejected):
        sources.read_source(str(path))
