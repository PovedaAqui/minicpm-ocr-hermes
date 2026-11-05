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
