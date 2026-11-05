"""Pre-flight checks against the documented service limits.

The API rejects an oversized file with a 413 and a PDF over 500 pages with a
4xx that says nothing useful, so the plugin checks first and returns a message
the model can act on. Limits come from the public Document Parsing API docs.
"""

from __future__ import annotations

import re
import struct

from .errors import PageLimitExceeded, PayloadTooLarge

MAX_IMAGE_BYTES = 24 * 1024 * 1024
MAX_ORIGINAL_BYTES = 64 * 1024 * 1024
MAX_PAGES = 500
MAX_PIXELS = 40_000_000
MAX_DIMENSION = 8192

_MIME = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "bmp": "image/bmp",
    "webp": "image/webp",
}
_PDF_PAGE_RE = re.compile(rb"/Type\s*/Page(?![s\w])")


def sniff_image(data: bytes) -> str | None:
    """Return ``png`` / ``jpeg`` / ``gif`` / ``bmp`` / ``webp`` from magic bytes."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if data[:2] == b"BM":
        return "bmp"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def mime_for(data: bytes) -> str:
    kind = sniff_image(data)
    return _MIME.get(kind or "", "application/octet-stream")


def image_size(data: bytes) -> tuple[int, int] | None:
    """``(width, height)`` for PNG / GIF / BMP / WEBP-VP8X, else ``None``."""
    kind = sniff_image(data)
    try:
        if kind == "png" and len(data) >= 24:
            return struct.unpack(">II", data[16:24])
        if kind == "gif" and len(data) >= 10:
            return struct.unpack("<HH", data[6:10])
        if kind == "bmp" and len(data) >= 26:
            w, h = struct.unpack("<ii", data[18:26])
            return abs(w), abs(h)
        if kind == "webp" and data[12:16] == b"VP8X" and len(data) >= 30:
            w = int.from_bytes(data[24:27], "little") + 1
            h = int.from_bytes(data[27:30], "little") + 1
            return w, h
        if kind == "jpeg":
            return _jpeg_size(data)
    except (struct.error, IndexError):
        return None
    return None


def _jpeg_size(data: bytes) -> tuple[int, int] | None:
    i = 2
    n = len(data)
    while i + 9 < n:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        seg_len = int.from_bytes(data[i + 2 : i + 4], "big")
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            h = int.from_bytes(data[i + 5 : i + 7], "big")
            w = int.from_bytes(data[i + 7 : i + 9], "big")
            return w, h
        i += 2 + max(seg_len, 2)
    return None


def check_image(data: bytes, name: str = "") -> None:
    label = name or "image"
    if len(data) > MAX_IMAGE_BYTES:
        raise PayloadTooLarge(
            f"{label} is {len(data) / 1e6:.1f} MB; the image endpoint accepts up to 24 MB. "
            "Downscale it or send it as a PDF job."
        )
    size = image_size(data)
    if size:
        w, h = size
        if w * h > MAX_PIXELS or w > MAX_DIMENSION or h > MAX_DIMENSION:
            raise PayloadTooLarge(
                f"{label} is {w}x{h}px; the service accepts up to 40 MP and 8192 px per side. "
                "Downscale it before sending."
            )


def looks_like_pdf(data: bytes) -> bool:
    return data[:5] == b"%PDF-"


def count_pdf_pages(data: bytes) -> int:
    """Rough page count from ``/Type /Page`` markers; 0 when it cannot be told."""
    return len(_PDF_PAGE_RE.findall(data))


def check_pdf(data: bytes, name: str = "") -> int:
    label = name or "PDF"
    if len(data) > MAX_ORIGINAL_BYTES:
        raise PayloadTooLarge(
            f"{label} is {len(data) / 1e6:.1f} MB; the service accepts originals up to 64 MB."
        )
    pages = count_pdf_pages(data)
    if pages > MAX_PAGES:
        raise PageLimitExceeded(
            f"{label} has ~{pages} pages; the job endpoint accepts up to {MAX_PAGES}. Split it first."
        )
    return pages


def check_document(data: bytes, name: str = "") -> str:
    """Validate a source and return ``"image"`` or ``"pdf"``."""
    if looks_like_pdf(data):
        check_pdf(data, name)
        return "pdf"
    kind = sniff_image(data)
    if kind is None:
        raise PayloadTooLarge(
            f"{name or 'file'} is not a supported input (PDF, PNG, JPEG, GIF, BMP, WEBP)."
        )
    check_image(data, name)
    return "image"
