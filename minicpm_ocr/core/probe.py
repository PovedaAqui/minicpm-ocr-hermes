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

# A local document service speaks the same job contract as the hosted API and
# owns layout, so it has blocks and PDFs. The official Docker image serves it on
# :8000 under /api; the SDK Web Demo backend defaults to :8966.
LOCAL_DOCUMENT_SERVERS = (
    ("local-document", "http://127.0.0.1:8966"),
    ("local-document-docker", "http://127.0.0.1:8000"),
)

# Default ports: vLLM / SGLang, llama.cpp's llama-server, Ollama.
LOCAL_CHAT_SERVERS = (
    ("local-vllm", "http://127.0.0.1:8000"),
    ("local-llamacpp", "http://127.0.0.1:8080"),
    ("local-ollama", "http://127.0.0.1:11434"),
)

# Kept for callers that imported the old name.
LOCAL_SERVERS = LOCAL_CHAT_SERVERS


@dataclass(frozen=True)
class Backend:
    name: str
    base_url: str
    api_key: str = ""
    kind: str = ""  # "modelbest" | "openai"; empty = infer from the host
    model: str = DEFAULT_MODEL
    jev_model: str = DEFAULT_JEV_MODEL
    jev_base_url: str = ""
    source: str = ""
    api: str = "auto"  # auto | modelbest | document | chat
    jev_api: str = "auto"  # auto | rm | chat

    @property
    def surface(self) -> str:
        """What the endpoint speaks: ``modelbest``, ``document`` or ``chat``."""
        if self.api and self.api != "auto":
            return self.api
        parts = urlsplit(self.base_url)
        host = parts.hostname or ""
        if host.endswith("modelbest.ai"):
            return "modelbest"
        # The SDK Web Demo's document service defaults to :8966; inferring it
        # saves a setting without guessing about :8000, which both vLLM and the
        # Docker document image use.
        return "document" if parts.port == 8966 else "chat"

    @property
    def resolved_kind(self) -> str:
        """Kept for older callers; the surface is the useful distinction now."""
        if self.kind:
            return self.kind
        return "modelbest" if self.surface == "modelbest" else "openai"

    @property
    def blocks(self) -> bool:
        """Per-region bboxes exist on the hosted API and on a local document service.

        A bare vLLM / SGLang / llama.cpp / Ollama chat server returns page text;
        the layout stage that produces blocks lives in OpenBMB's SDK, which the
        document service runs for you.
        """
        return self.surface in ("modelbest", "document")

    @property
    def jobs(self) -> bool:
        """Whether the endpoint has the asynchronous document-parse job contract."""
        return self.blocks

    def client(self, **kwargs: Any) -> OcrClient:
        return OcrClient(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            jev_model=self.jev_model,
            jev_base_url=self.jev_base_url,
            surface=self.surface,
            jev_api=self.jev_api,
            **kwargs,
        )


def _surface(local_api: str) -> str:
    return "document" if local_api == "document" else "chat"


def candidates(settings: Mapping[str, Any], *, env_key: str = "") -> list[Backend]:
    """Ordered backends to try, honouring an explicit mode: auto | cloud | local."""
    out: list[Backend] = []
    base = str(settings.get("base_url") or "").strip().rstrip("/")
    key = str(settings.get("_api_key") or env_key or "").strip()
    mode = str(settings.get("backend") or "auto").strip().lower()
    local_api = str(settings.get("local_api") or "auto").strip().lower()
    jev_api = str(settings.get("jev_api") or "auto").strip().lower()
    model = str(settings.get("parser_model") or settings.get("model") or DEFAULT_MODEL)
    jev_model = str(settings.get("jev_model") or DEFAULT_JEV_MODEL)
    jev_base = str(settings.get("jev_base_url") or "").strip().rstrip("/")
    configured_name = {"document": "local-document", "chat": "local-chat"}.get(local_api, "configured")

    if base:
        out.append(
            Backend(
                name=configured_name,
                base_url=base,
                api_key=key,
                model=model,
                jev_model=jev_model,
                jev_base_url=jev_base,
                source="settings",
                api=_surface(local_api) if local_api in ("document", "chat") else "auto",
                jev_api=jev_api,
            )
        )
    if key and mode in ("auto", "cloud"):
        out.append(
            Backend(
                name="modelbest",
                base_url=DEFAULT_BASE_URL,
                api_key=key,
                kind="modelbest",
                model=model,
                jev_model=jev_model,
                jev_base_url=jev_base,
                source="api key",
                api="modelbest",
                jev_api=jev_api,
            )
        )
    if mode in ("auto", "local"):
        servers = LOCAL_DOCUMENT_SERVERS if local_api == "document" else LOCAL_CHAT_SERVERS
        for name, url in servers:
            out.append(
                Backend(
                    name=name,
                    base_url=url,
                    kind="openai",
                    model=model,
                    jev_model=jev_model,
                    jev_base_url=jev_base,
                    source="loopback",
                    api=_surface(local_api) if local_api in ("document", "chat") else "auto",
                    jev_api=jev_api,
                )
            )
    return out


def reachable(backend: Backend, *, timeout: float = 0.5) -> bool:
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
        "surface": backend.surface,
        "kind": backend.resolved_kind,
        "blocks": backend.blocks,
        "jobs": backend.jobs,
        "jev_base_url": backend.jev_base_url or backend.base_url,
        "jev_api": backend.jev_api,
        "source": backend.source,
    }
    if not reachable(backend):
        report.update(ok=False, error="not listening")
        return report
    client = backend.client(timeout=timeout, max_attempts=1)
    if backend.surface == "document":
        try:
            health = client.health(timeout=timeout)
        except OcrError as exc:
            report.update(ok=False, error=exc.message, code=exc.code)
            return report
        report.update(ok=str(health.get("status") or "").lower() == "ok", health=health)
        if not report["ok"]:
            report["error"] = f"health returned {health.get('status')!r}"
        return report
    try:
        models = client.models(timeout=timeout)
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
    jev_base = str(settings.get("jev_base_url") or "").strip().rstrip("/")
    if jev_base and jev_base not in {r["base_url"] for r in reports}:
        reports.append(
            probe(
                Backend(name="jev-only", base_url=jev_base, kind="openai", api="chat"),
                timeout=timeout,
            )
        )
    usable = [r for r in reports if r.get("ok")]
    return {
        "ok": bool(usable),
        "usable": usable[0]["name"] if usable else None,
        "backends": reports,
        "hint": None
        if usable
        else "Set a ModelBest API key, point base_url at a local document service, or run "
        "llama-server / vLLM / Ollama with MiniCPM-OCR.",
    }
