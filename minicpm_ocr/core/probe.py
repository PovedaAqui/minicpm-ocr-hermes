"""Backend discovery for ``hermes ocr doctor`` and first-run setup.

Nothing here runs at registration: ``hermes plugins doctor`` blocks sockets
while ``register(ctx)`` executes, so probing is always an explicit call.
"""

from __future__ import annotations

import socket
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from .client import DEFAULT_BASE_URL, DEFAULT_JEV_MODEL, DEFAULT_MODEL, OcrClient
from .errors import OcrError


@dataclass(frozen=True)
class Backend:
    name: str
    base_url: str
    api_key: str = ""
    blocks: bool = True
    model: str = DEFAULT_MODEL
    jev_model: str = DEFAULT_JEV_MODEL
    source: str = ""

    @property
    def kind(self) -> str:
        host = urlsplit(self.base_url).hostname or ""
        return "modelbest" if host.endswith("modelbest.ai") else "openai"

    def client(self, **kwargs: Any) -> OcrClient:
        return OcrClient(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            jev_model=self.jev_model,
            **kwargs,
        )


def candidates(settings: Mapping[str, Any], *, env_key: str = "") -> list[Backend]:
    """Ordered backends to try: explicit setting, then cloud key, then local servers."""
    out: list[Backend] = []
    base = str(settings.get("base_url") or "").strip().rstrip("/")
    key = str(settings.get("_api_key") or env_key or "").strip()
    if base:
        out.append(Backend(name="configured", base_url=base, api_key=key, blocks=True, source="settings"))
    if key:
        out.append(
            Backend(
                name="modelbest",
                base_url=DEFAULT_BASE_URL,
                api_key=key,
                blocks=True,
                source="api key",
            )
        )
    for name, url in (
        ("local-vllm", "http://127.0.0.1:8000"),
        ("local-ollama", "http://127.0.0.1:11434"),
    ):
        # A local server has no layout stage, so it cannot produce per-region bboxes.
        out.append(Backend(name=name, base_url=url, blocks=False, source="loopback"))
    return out


def reachable(backend: Backend, *, timeout: float = 2.5) -> bool:
    host = urlsplit(backend.base_url).hostname or ""
    if host not in ("127.0.0.1", "localhost", "::1"):
        return True
    port = urlsplit(backend.base_url).port or (443 if backend.base_url.startswith("https") else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def probe(backend: Backend, *, timeout: float = 8.0) -> dict[str, Any]:
    """Return a JSON-safe report for one backend."""
    report: dict[str, Any] = {
        "name": backend.name,
        "base_url": backend.base_url,
        "kind": backend.kind,
        "blocks": backend.blocks,
        "source": backend.source,
    }
    if not reachable(backend):
        report.update(ok=False, error="not listening")
        return report
    try:
        models = backend.client(timeout=timeout, max_attempts=1).models(timeout=timeout)
    except OcrError as exc:
        report.update(ok=False, error=exc.message, code=exc.code)
        return report
    report.update(ok=True, models=models[:20])
    found = {m.lower() for m in models}
    report["has_parser"] = backend.model.lower() in found or not models
    report["has_jev"] = backend.jev_model.lower() in found or not models
    return report


def doctor(settings: Mapping[str, Any], *, env_key: str = "", timeout: float = 8.0) -> dict[str, Any]:
    reports = [probe(b, timeout=timeout) for b in candidates(settings, env_key=env_key)]
    usable = [r for r in reports if r.get("ok")]
    return {
        "ok": bool(usable),
        "usable": usable[0]["name"] if usable else None,
        "backends": reports,
        "hint": None
        if usable
        else "Set a ModelBest API key, or run vLLM/Ollama with MiniCPM-OCR on 127.0.0.1:8000 / :11434.",
    }
