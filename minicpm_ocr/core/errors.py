"""Error taxonomy for the MiniCPM-OCR client.

Every failure the tools can hit is one of these, so a caller (or the model) can
branch on ``code`` instead of parsing prose. Handlers never raise: they turn
these into ``{"ok": false, "error": {...}}``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class OcrError(Exception):
    """Base error. ``retryable`` says whether the same call may succeed later."""

    code = "ocr_error"
    retryable = False

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        retryable: bool | None = None,
        detail: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        if retryable is not None:
            self.retryable = retryable
        self.detail = detail

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
        }
        if self.status is not None:
            out["status"] = self.status
        if self.detail:
            out["detail"] = self.detail
        return out


class AuthError(OcrError):
    code = "auth"


class RateLimited(OcrError):
    code = "rate_limited"
    retryable = True

    def __init__(self, message: str, *, retry_after: float | None = None, **kw: Any) -> None:
        super().__init__(message, **kw)
        self.retry_after = retry_after


class PayloadTooLarge(OcrError):
    code = "payload_too_large"


class PageLimitExceeded(OcrError):
    code = "page_limit"


class BackendUnavailable(OcrError):
    code = "backend_unavailable"
    retryable = True


class RequestRejected(OcrError):
    code = "request_rejected"


class ParseFailed(OcrError):
    code = "parse_failed"


def from_status(status: int, body: bytes, *, headers: Mapping[str, str] | None = None) -> OcrError:
    """Map an HTTP status onto the taxonomy, keeping a short body excerpt."""
    text = body.decode("utf-8", "replace").strip()
    detail = text[:600] if text else None
    snippet = _message_from_body(text) or f"HTTP {status}"
    if status in (401, 403):
        return AuthError(snippet, status=status, detail=detail)
    if status == 413:
        return PayloadTooLarge(snippet, status=status, detail=detail)
    if status == 429:
        retry_after = None
        if headers:
            raw = headers.get("Retry-After") or headers.get("retry-after")
            try:
                retry_after = float(raw) if raw else None
            except (TypeError, ValueError):
                retry_after = None
        return RateLimited(snippet, status=status, retry_after=retry_after, detail=detail)
    if 400 <= status < 500:
        return RequestRejected(snippet, status=status, detail=detail)
    return BackendUnavailable(snippet, status=status, detail=detail)


def _message_from_body(text: str) -> str | None:
    """Pull ``error.message`` / ``message`` out of a JSON error body, if present."""
    if not text:
        return None
    import json

    try:
        payload = json.loads(text)
    except ValueError:
        return text.splitlines()[0][:200]
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])[:300]
        if isinstance(err, str) and err:
            return err[:300]
        for key in ("message", "detail", "error_description"):
            if payload.get(key):
                return str(payload[key])[:300]
    return None
