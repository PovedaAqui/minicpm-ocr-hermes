"""Resolve a tool argument (local path or https URL) into bytes."""

from __future__ import annotations

import ipaddress
import socket
import urllib.error
import urllib.parse
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
        if low.startswith("http://"):
            raise RequestRejected(
                "only https URLs are fetched; download the file and pass a local path instead"
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


def _is_private_address(ip: str) -> bool:
    """True for anything a public URL must never resolve to (SSRF guard)."""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return True
    return (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified
    )


def ensure_public(url: str) -> None:
    """Reject URLs whose host resolves to a private, loopback or link-local address."""
    parts = urllib.parse.urlsplit(url)
    host = parts.hostname or ""
    port = parts.port or (443 if parts.scheme == "https" else 80)
    if not host:
        raise RequestRejected(f"bad URL: {url}")
    if host == "localhost" or _is_private_address(host):
        raise RequestRejected("refusing to fetch a URL that points at a private or loopback address")
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except OSError as exc:
        raise RequestRejected(f"cannot resolve {host}: {exc}") from exc
    for info in infos:
        if _is_private_address(info[4][0]):
            raise RequestRejected(
                f"{host} resolves to a private address ({info[4][0]}); refusing to fetch it"
            )


class _GuardedRedirects(urllib.request.HTTPRedirectHandler):
    """Follow redirects only while every hop stays public."""

    max_redirections = _MAX_REDIRECTS

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        ensure_public(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_GuardedRedirects)


def _fetch(url: str, *, timeout: float) -> tuple[bytes, str]:
    ensure_public(url)
    req = urllib.request.Request(url, headers={"User-Agent": "hermes-minicpm-ocr/0.6"})
    try:
        with _OPENER.open(req, timeout=timeout) as resp:
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
