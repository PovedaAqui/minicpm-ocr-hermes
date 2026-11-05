"""Resolve a tool argument (local path or https URL) into bytes."""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

from .errors import BackendUnavailable, RequestRejected
from .limits import MAX_ORIGINAL_BYTES

_MAX_REDIRECTS = 3


def read_source(source: str, *, timeout: float = 60.0) -> tuple[bytes, str]:
    """Return ``(bytes, name)`` for a path, ``file://`` URI or public https URL."""
    if not source or not isinstance(source, str):
        raise RequestRejected("source is required: an absolute path or a public https URL")
    low = source.lower()
    if low.startswith(("http://", "https://")):
        if low.startswith("http://") and not _is_loopback(source):
            raise RequestRejected(
                "only https URLs are fetched; plain http is allowed for localhost services only"
            )
        return _fetch(source, timeout=timeout)
    if low.startswith("file://"):
        source = source[7:]
    path = Path(source).expanduser()
    if not path.is_file():
        raise RequestRejected(f"no such file: {source}")
    data = path.read_bytes()
    if len(data) > MAX_ORIGINAL_BYTES:
        raise RequestRejected(f"{path.name} is larger than the 64 MB source limit")
    return data, path.name


def _is_loopback(url: str) -> bool:
    host = urllib.parse.urlsplit(url).hostname or ""
    return host in ("127.0.0.1", "localhost", "::1")


def _fetch(url: str, *, timeout: float) -> tuple[bytes, str]:
    req = urllib.request.Request(url, headers={"User-Agent": "hermes-minicpm-ocr/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            length = resp.headers.get("Content-Length")
            if length and int(length) > MAX_ORIGINAL_BYTES:
                raise RequestRejected(f"{url} is larger than the 64 MB source limit")
            data = resp.read(MAX_ORIGINAL_BYTES + 1)
            if len(data) > MAX_ORIGINAL_BYTES:
                raise RequestRejected(f"{url} is larger than the 64 MB source limit")
    except urllib.error.HTTPError as exc:
        raise RequestRejected(f"download failed ({exc.code}) for {url}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise BackendUnavailable(f"could not download {url}: {exc}") from exc
    name = Path(urllib.parse.urlsplit(url).path).name or "download"
    return data, name
