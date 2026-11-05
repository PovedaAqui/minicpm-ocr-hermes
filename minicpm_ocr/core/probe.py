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

# Default ports: vLLM / SGLang, llama.cpp's llama-server, Ollama.
LOCAL_SERVERS = (
    ("local-vllm", "http://127.0.0.1:8000"),
    ("local-llamacpp", "http://127.0.0.1:8080"),
    ("local-ollama", "http://127.0.0.1:11434"),
)


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

    @property
    def resolved_kind(self) -> str:
        if self.kind:
            return self.kind
        host = urlsplit(self.base_url).hostname or ""
        return "modelbest" if host.endswith("modelbest.ai") else "openai"

    @property
    def blocks(self) -> bool:
        """Per-region bboxes exist only on the hosted document-parse API.

        A local vLLM / SGLang / llama.cpp / Ollama server speaks Chat Completions
        and returns page text; the layout stage that produces blocks lives in
        OpenBMB's SDK, not in the model.
        """
        return self.resolved_kind == "modelbest"

    def client(self, **kwargs: Any) -> OcrClient:
        return OcrClient(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            jev_model=self.jev_model,
            jev_base_url=self.jev_base_url,
            **kwargs,
        )


def candidates(settings: Mapping[str, Any], *, env_key: str = "") -> list[Backend]:
    """Ordered backends to try, honouring an explicit mode: auto | cloud | local."""
    out: list[Backend] = []
    base = str(settings.get("base_url") or "").strip().rstrip("/")
    key = str(settings.get("_api_key") or env_key or "").strip()
    mode = str(settings.get("backend") or "auto").strip().lower()
    model = str(settings.get("model") or DEFAULT_MODEL)
    jev_model = str(settings.get("jev_model") or DEFAULT_JEV_MODEL)
    jev_base = str(settings.get("jev_base_url") or "").strip().rstrip("/")

    if base:
        out.append(
            Backend(
                name="configured",
                base_url=base,
                api_key=key,
                model=model,
                jev_model=jev_model,
                jev_base_url=jev_base,
                source="settings",
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
            )
        )
    if mode in ("auto", "local"):
        for name, url in LOCAL_SERVERS:
            out.append(
                Backend(
                    name=name,
                    base_url=url,
                    kind="openai",
                    model=model,
                    jev_model=jev_model,
                    jev_base_url=jev_base,
                    source="loopback",
                )
            )
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
        "kind": backend.resolved_kind,
        "blocks": backend.blocks,
        "jev_base_url": backend.jev_base_url or backend.base_url,
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
    jev_base = str(settings.get("jev_base_url") or "").strip().rstrip("/")
    if jev_base and jev_base not in {r["base_url"] for r in reports}:
        reports.append(probe(Backend(name="jev-only", base_url=jev_base, kind="openai"), timeout=timeout))
    usable = [r for r in reports if r.get("ok")]
    return {
        "ok": bool(usable),
        "usable": usable[0]["name"] if usable else None,
        "backends": reports,
        "hint": None
        if usable
        else "Set a ModelBest API key, or run llama-server / vLLM / Ollama with MiniCPM-OCR.",
    }
