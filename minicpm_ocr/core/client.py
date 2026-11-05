"""HTTP client for MiniCPM-OCR — ModelBest API and any OpenAI-compatible server.

Standard library only: the plugin declares no Python dependencies, which keeps
catalog review simple and the install side-effect-free.

Two backends, one surface:

* ``modelbest`` — ``POST /v1/document-parse`` (sync image) and
  ``POST /v1/document-parse/jobs`` (async image or PDF), returning Markdown plus
  per-block ``status`` and ``bbox``. Jev is a Chat Completions call.
* ``openai`` — a local vLLM / SGLang / llama.cpp / Ollama server. It answers the
  same Chat Completions shape, page-level only: the layout stage that produces
  blocks lives in OpenBMB's SDK, not in the model. ``Backend.blocks`` says which
  one you are talking to so the caller never assumes bboxes it does not have.
"""

from __future__ import annotations

import base64
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from . import limits
from .errors import (
    BackendUnavailable,
    OcrError,
    ParseFailed,
    from_status,
)

DEFAULT_BASE_URL = "https://api.modelbest.ai"
DEFAULT_MODEL = "MiniCPM-OCR-1.0"
DEFAULT_JEV_MODEL = "MiniCPM-OCR-Jev"

PARSE_PATH = "/v1/document-parse"
JOBS_PATH = "/v1/document-parse/jobs"
CHAT_PATH = "/v1/chat/completions"
MODELS_PATH = "/v1/models"

RETRY_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

Response = tuple[int, Mapping[str, str], bytes]
Transport = Callable[[str, str, Mapping[str, str], bytes | None, float], Response]


# --------------------------------------------------------------------------- models


@dataclass(frozen=True)
class Block:
    """One detected region: what the model read, and whether it trusts it."""

    order: int
    label: str
    content: str
    status: str
    bbox: tuple[int, int, int, int] | None = None
    score: float | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"


@dataclass(frozen=True)
class Page:
    index: int
    markdown: str
    blocks: tuple[Block, ...] = ()
    width: int | None = None
    height: int | None = None
    rotation: Mapping[str, Any] | None = None

    @property
    def failed(self) -> tuple[Block, ...]:
        return tuple(b for b in self.blocks if not b.ok)


@dataclass(frozen=True)
class ParseResult:
    pages: tuple[Page, ...]
    request_id: str | None = None
    result_status: str | None = None
    issues: tuple[Mapping[str, Any], ...] = ()
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class JobStatus:
    """State of an asynchronous parse job.

    The service reports ``status`` as ``Running`` / ``Success`` (capitalised),
    with ``pages_done`` / ``pages_total`` and no progress fraction, so both are
    derived here instead of guessed at the call site.
    """

    id: str
    status: str
    pages_done: int = 0
    pages_total: int = 0
    pages: tuple[Page, ...] = ()
    markdown: str = ""
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @property
    def fraction(self) -> float | None:
        if not self.pages_total:
            return None
        return min(1.0, self.pages_done / self.pages_total)

    @property
    def terminal(self) -> bool:
        return self.status.strip().lower() in _TERMINAL

    @property
    def ok(self) -> bool:
        return self.status.strip().lower() in _OK


_TERMINAL = frozenset(
    {"success", "succeeded", "completed", "done", "failed", "error", "cancelled", "canceled"}
)
_OK = frozenset({"success", "succeeded", "completed", "done"})


@dataclass(frozen=True)
class JevResult:
    score: float
    prob_no: float
    verdict: str
    passed: bool
    category: str
    threshold: float
    detail: str = ""
    case_id: str = ""


# --------------------------------------------------------------------------- transport


def urllib_transport(
    method: str, url: str, headers: Mapping[str, str], body: bytes | None, timeout: float
) -> Response:
    req = urllib.request.Request(url, data=body, headers=dict(headers), method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, dict(resp.headers.items()), resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers.items()) if exc.headers else {}, exc.read()
    except urllib.error.URLError as exc:
        raise BackendUnavailable(f"cannot reach {_origin(url)}: {exc.reason}") from exc


def _origin(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}"


# --------------------------------------------------------------------------- client


class OcrClient:
    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = "",
        model: str = DEFAULT_MODEL,
        jev_model: str = DEFAULT_JEV_MODEL,
        timeout: float = 300.0,
        max_attempts: int = 3,
        backoff: float = 1.5,
        sleep: Callable[[float], None] = time.sleep,
        transport: Transport | None = None,
        user_agent: str = "hermes-minicpm-ocr/0.1",
    ) -> None:
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key
        self.model = model
        self.jev_model = jev_model
        self.timeout = timeout
        self.max_attempts = max(1, max_attempts)
        self.backoff = backoff
        self._sleep = sleep
        self._transport = transport or urllib_transport
        self.user_agent = user_agent

    # -- public API ---------------------------------------------------------

    def models(self, *, timeout: float = 8.0) -> list[str]:
        """List model ids the key can see; also the cheapest reachability probe."""
        _, _, raw = self._request("GET", MODELS_PATH, expect=(200,), timeout=timeout)
        payload = _json(raw)
        entries = payload.get("data") or payload.get("models") or payload
        if isinstance(entries, dict):
            entries = entries.get("data") or []
        ids: list[str] = []
        if isinstance(entries, list):
            for item in entries:
                if isinstance(item, str):
                    ids.append(item)
                elif isinstance(item, dict) and item.get("id"):
                    ids.append(str(item["id"]))
        return ids

    def parse_image(
        self,
        data: bytes,
        *,
        name: str = "page",
        model: str | None = None,
        upright: bool = True,
        recognize_seal: bool = True,
    ) -> ParseResult:
        limits.check_image(data, name)
        payload = {
            "model": model or self.model,
            "image": base64.b64encode(data).decode("ascii"),
            "upright": upright,
            "recognize_seal": recognize_seal,
        }
        _, _, raw = self._request("POST", PARSE_PATH, payload=payload)
        body = _json(raw)
        page = _page_from(body)
        return ParseResult(
            pages=(page,),
            request_id=body.get("request_id"),
            result_status=body.get("result_status"),
            issues=tuple(body.get("issues") or ()),
            raw=body,
        )

    def parse_page(
        self,
        data: bytes,
        *,
        name: str = "page",
        prompt: str = "Text Recognition:",
        model: str | None = None,
        timeout: float | None = None,
    ) -> ParseResult:
        """Read a page through Chat Completions — the local vLLM / Ollama contract.

        Page-level text only: without the SDK's layout stage there are no blocks
        and no per-region status, which is why the tool payload reports
        ``blocks_available: false`` for this backend.
        """
        limits.check_image(data, name)
        data_uri = f"data:{limits.mime_for(data)};base64,{base64.b64encode(data).decode('ascii')}"
        payload = {
            "model": model or self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 4096,
            "stream": False,
        }
        _, _, raw = self._request("POST", CHAT_PATH, payload=payload, timeout=timeout or self.timeout)
        body = _json(raw)
        choices = body.get("choices") or []
        if not choices:
            raise ParseFailed("the backend returned no choices", detail=str(body)[:400])
        content = str((choices[0].get("message") or {}).get("content") or "")
        page = Page(index=0, markdown=content.strip(), blocks=())
        return ParseResult(
            pages=(page,),
            request_id=str(body.get("id") or "") or None,
            result_status="ok",
            raw=body,
        )

    def submit_job(
        self,
        data: bytes,
        *,
        name: str,
        model: str | None = None,
        timeout: float | None = None,
    ) -> str:
        """Upload an image or PDF and return the job id (HTTP 202)."""
        kind = limits.check_document(data, name)
        boundary = f"----hermes{uuid.uuid4().hex}"
        body = _multipart(
            boundary,
            fields={"model": model or self.model},
            file_field="file",
            filename=name,
            file_bytes=data,
            content_type=limits.mime_for(data) if kind == "image" else "application/pdf",
        )
        headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
        _, _, raw = self._request(
            "POST", JOBS_PATH, body=body, headers=headers, expect=(200, 201, 202), timeout=timeout
        )
        payload = _json(raw)
        job_id = _job_id(payload)
        if not job_id:
            raise ParseFailed("the job endpoint returned no job id", detail=str(payload)[:400])
        return job_id

    def job(self, job_id: str, *, timeout: float | None = None) -> JobStatus:
        _, _, raw = self._request("GET", f"{JOBS_PATH}/{job_id}", timeout=timeout)
        return _job_status(_json(raw), job_id)

    def cancel_job(self, job_id: str, *, timeout: float | None = None) -> JobStatus:
        _, _, raw = self._request(
            "DELETE", f"{JOBS_PATH}/{job_id}", expect=(200, 202, 204), timeout=timeout
        )
        payload = _json(raw) if raw else {}
        return _job_status(payload, job_id, default_status="cancelled")

    def wait(
        self,
        job_id: str,
        *,
        timeout: float = 600.0,
        interval: float = 1.5,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> JobStatus:
        """Poll until the job reaches a terminal state or ``timeout`` elapses."""
        deadline = clock() + timeout
        status = self.job(job_id)
        while not status.terminal and clock() < deadline:
            sleep(interval)
            status = self.job(job_id)
        return status

    def jev_score(
        self,
        image: bytes,
        text: str,
        *,
        category: str = "text",
        threshold: float = 0.6,
        model: str | None = None,
        timeout: float | None = None,
    ) -> JevResult:
        """Score whether ``text`` actually matches ``image`` (0-1, higher = better)."""
        limits.check_image(image, "region")
        if category not in ("text", "formula", "table"):
            raise ParseFailed(f"category must be text, formula or table (got {category!r})")
        data_uri = f"data:{limits.mime_for(image)};base64,{base64.b64encode(image).decode('ascii')}"
        payload = {
            "model": model or self.jev_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": text},
                    ],
                }
            ],
            "minicpm_ocr_jev": {"mode": "score", "category": category, "threshold": threshold},
            "stream": False,
        }
        _, _, raw = self._request("POST", CHAT_PATH, payload=payload, timeout=timeout or 240.0)
        body = _json(raw)
        results = body.get("results") or []
        if not results:
            raise ParseFailed("Jev returned no results", detail=str(body)[:400])
        first = results[0]
        score = float(first.get("prob_yes") or 0.0)
        used = float((body.get("summary") or {}).get("threshold_used", threshold))
        return JevResult(
            score=score,
            prob_no=float(first.get("prob_no") or 0.0),
            verdict=str(first.get("verdict") or ""),
            passed=score >= used,
            category=str(first.get("category") or category),
            threshold=used,
            detail=str(first.get("detail") or ""),
            case_id=str(first.get("case_id") or ""),
        )

    # -- internals ----------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: Mapping[str, Any] | None = None,
        body: bytes | None = None,
        headers: Mapping[str, str] | None = None,
        expect: tuple[int, ...] = (200,),
        timeout: float | None = None,
    ) -> Response:
        url = self.base_url + path
        hdrs: dict[str, str] = {
            "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
            "Accept": "application/json",
            "User-Agent": self.user_agent,
        }
        hdrs = {k: v for k, v in hdrs.items() if v}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            hdrs["Content-Type"] = "application/json"
        if headers:
            hdrs.update(headers)
        wait = timeout if timeout is not None else self.timeout

        last: OcrError | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                status, resp_headers, raw = self._transport(method, url, hdrs, body, wait)
            except (TimeoutError, OSError) as exc:
                last = BackendUnavailable(f"cannot reach {_origin(url)}: {exc}")
                if attempt < self.max_attempts:
                    self._sleep(self._delay(attempt))
                    continue
                raise last from exc
            if status in expect:
                return status, resp_headers, raw
            if status in RETRY_STATUS and attempt < self.max_attempts:
                delay = _retry_after(resp_headers) or self._delay(attempt)
                self._sleep(delay)
                last = from_status(status, raw, headers=resp_headers)
                continue
            raise from_status(status, raw, headers=resp_headers)
        raise last or BackendUnavailable(f"no response from {_origin(url)}")

    def _delay(self, attempt: int) -> float:
        base = min(self.backoff * (2 ** (attempt - 1)), 30.0)
        return base * (1.0 + random.random() * 0.1)


# --------------------------------------------------------------------------- helpers


def _json(raw: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ParseFailed(
            "service returned a non-JSON body", detail=raw[:300].decode("utf-8", "replace")
        ) from exc
    if not isinstance(payload, dict):
        raise ParseFailed("service returned a JSON value that is not an object")
    return payload


def _page_from(payload: Mapping[str, Any]) -> Page:
    markdown = ""
    choices = payload.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        markdown = str(message.get("content") or "")
    ocr = payload.get("ocr") or {}
    blocks: list[Block] = []
    for item in ocr.get("blocks") or []:
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox")
        box: tuple[int, int, int, int] | None = None
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                box = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
            except (TypeError, ValueError):
                box = None
        blocks.append(
            Block(
                order=int(item.get("order") or 0),
                label=str(item.get("label") or ""),
                content=str(item.get("content") or ""),
                status=str(item.get("status") or "ok"),
                bbox=box,
            )
        )
    metadata = payload.get("metadata") or {}
    width = ocr.get("image_width")
    height = ocr.get("image_height")
    return Page(
        index=int(metadata.get("page_index") or 0),
        markdown=markdown,
        blocks=tuple(blocks),
        width=int(width) if isinstance(width, (int, float)) else None,
        height=int(height) if isinstance(height, (int, float)) else None,
        rotation=ocr.get("rotation"),
    )


def _job_status(payload: Mapping[str, Any], job_id: str, *, default_status: str = "unknown") -> JobStatus:
    pages: list[Page] = []
    raw_pages = payload.get("pages")
    if isinstance(raw_pages, list):
        pages = [_page_from(item) for item in raw_pages if isinstance(item, dict)]
    done = payload.get("pages_done")
    total = payload.get("pages_total")
    return JobStatus(
        id=str(payload.get("id") or job_id),
        status=str(payload.get("status") or default_status),
        pages_done=int(done) if isinstance(done, (int, float)) else 0,
        pages_total=int(total) if isinstance(total, (int, float)) else 0,
        pages=tuple(pages),
        markdown=str(payload.get("markdown") or ""),
        raw=payload,
    )


def _job_id(payload: Mapping[str, Any]) -> str | None:
    for key in ("id", "job_id", "task_id"):
        if payload.get(key):
            return str(payload[key])
    for container in ("job", "data", "result"):
        inner = payload.get(container)
        if isinstance(inner, dict):
            for key in ("id", "job_id"):
                if inner.get(key):
                    return str(inner[key])
    return None


def _retry_after(headers: Mapping[str, str]) -> float | None:
    for key in ("Retry-After", "retry-after"):
        if key in headers:
            try:
                return max(0.0, float(headers[key]))
            except (TypeError, ValueError):
                return None
    return None


def _multipart(
    boundary: str,
    *,
    fields: Mapping[str, str],
    file_field: str,
    filename: str,
    file_bytes: bytes,
    content_type: str,
) -> bytes:
    lines: list[bytes] = []
    for key, value in fields.items():
        lines.append(f"--{boundary}\r\n".encode())
        lines.append(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode())
        lines.append(str(value).encode())
        lines.append(b"\r\n")
    lines.append(f"--{boundary}\r\n".encode())
    lines.append(
        f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'.encode()
    )
    lines.append(f"Content-Type: {content_type}\r\n\r\n".encode())
    lines.append(file_bytes)
    lines.append(b"\r\n")
    lines.append(f"--{boundary}--\r\n".encode())
    return b"".join(lines)
