"""Tool handlers.

Hermes contract: ``handler(args: dict, **kwargs) -> str``, always a JSON string,
never an exception. Everything the model needs to decide what to do next is in
the payload — including the reason a call failed.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

from .core import limits
from .core.client import JobStatus, OcrClient, Page
from .core.errors import BackendUnavailable, OcrError, ParseFailed, RequestRejected
from .core.probe import Backend, candidates, reachable
from .core.queue import Queue
from .core.sources import read_source

_CTX: Any = None
_API_KEY_NAME = "MODELBEST_API_KEY"


def set_context(ctx: Any) -> None:
    global _CTX
    _CTX = ctx


# --------------------------------------------------------------------------- handlers


def ocr_read(args: dict, **kwargs: Any) -> str:
    """Parse an image or PDF and report what came back, region by region."""
    try:
        source = _require(args, "path")
        mode = str(args.get("mode") or "summary").lower()
        if mode not in ("summary", "markdown", "blocks"):
            raise RequestRejected("mode must be summary, markdown or blocks")
        max_chars = int(args.get("max_chars") or 20000)
        data, name = read_source(source)
        kind = limits.check_document(data, name)
        client, backend = _client()

        job_id = None
        if kind == "pdf" or backend.surface == "document":
            if not backend.jobs:
                raise RequestRejected(
                    "this backend accepts images only: a bare vLLM/Ollama server has no document "
                    "job endpoint and no layout stage. Set local_api=document and point base_url at "
                    "the local document service (Docker or Web Demo), or use the hosted API."
                )
            job_id = client.submit_job(data, name=name, upright=bool(args.get("upright", True)))
            if not bool(args.get("wait", True)):
                return _ok(
                    {
                        "job_id": job_id,
                        "status": "submitted",
                        "source": {"name": name, "kind": kind, "bytes": len(data)},
                        "backend": backend.name,
                        "next": "Poll with ocr_job(job_id) — this document is still being parsed.",
                    }
                )
            status = client.wait(job_id, timeout=_job_timeout())
            if not status.ok and not status.terminal:
                # Slow is not broken: hand back the handle instead of failing the call.
                return _ok(
                    {
                        "job_id": job_id,
                        "status": status.status,
                        "pages_done": status.pages_done,
                        "pages_total": status.pages_total,
                        "backend": backend.name,
                        "next": "Still running: poll with ocr_job(job_id, action=status).",
                    }
                )
            if not status.ok:
                raise ParseFailed(
                    f"parse job {job_id} ended in state {status.status!r} "
                    f"({status.pages_done}/{status.pages_total} pages)",
                    detail=_excerpt(json.dumps(status.raw))[:400],
                )
            pages = list(status.pages)
            markdown = status.markdown or "\n\n".join(p.markdown for p in pages)
        elif backend.blocks:
            result = client.parse_image(data, name=name)
            pages = list(result.pages)
            markdown = "\n\n".join(p.markdown for p in pages)
        else:
            result = client.parse_page(data, name=name)
            pages = list(result.pages)
            markdown = "\n\n".join(p.markdown for p in pages)

        payload = _shape(pages, markdown, mode=mode, max_chars=max_chars, backend=backend)
        payload["source"] = {"name": name, "kind": kind, "bytes": len(data)}
        if job_id:
            payload["job_id"] = job_id
        if bool(args.get("queue", True)):
            try:
                queued = Queue().add_failed_blocks(source=name, pages=pages, job_id=job_id)
                if queued:
                    payload["queued"] = queued
            except Exception as exc:  # noqa: BLE001 — a full queue must not lose the read
                payload["queue_error"] = f"{type(exc).__name__}: {exc}"
        return _ok(payload)
    except OcrError as exc:
        return _err(exc.to_dict())
    except Exception as exc:  # noqa: BLE001 — the contract forbids raising
        return _err({"code": "internal", "message": f"{type(exc).__name__}: {exc}", "retryable": False})


def ocr_verify(args: dict, **kwargs: Any) -> str:
    """Score whether candidate text matches an image region, 0-1."""
    try:
        source = _require(args, "path")
        text = str(args.get("text") or "")
        if not text.strip():
            raise RequestRejected("text is required: the candidate reading you want scored")
        category = str(args.get("category") or "text").lower()
        settings = _settings()
        threshold = float(args.get("threshold") or settings.get("threshold") or 0.6)
        data, name = read_source(source)
        limits.check_image(data, name)
        client, backend = _client()
        if backend.surface != "modelbest" and not backend.jev_base_url:
            raise BackendUnavailable(
                "ocr_verify needs a Jev service. The local parser backend does not score text: "
                "start a local Jev render+score service and set jev_base_url (see README → Local Jev), "
                "or use the ModelBest API."
            )
        result = client.jev_score(data, text, category=category, threshold=threshold)
        return _ok(
            {
                "score": round(result.score, 4),
                "scored": result.scored,
                "passed": result.passed,
                "threshold": result.threshold,
                "category": result.category,
                "verdict": result.verdict,
                "backend": backend.name,
                "advice": _advice(result.passed, result.score, result.threshold, result.verdict),
            }
        )
    except OcrError as exc:
        return _err(exc.to_dict())
    except Exception as exc:  # noqa: BLE001
        return _err({"code": "internal", "message": f"{type(exc).__name__}: {exc}", "retryable": False})


def ocr_job(args: dict, **kwargs: Any) -> str:
    """Status of an asynchronous parse job; cancel it with action=cancel."""
    try:
        job_id = _require(args, "job_id")
        action = str(args.get("action") or "status").lower()
        client, backend = _client()
        status = client.cancel_job(job_id) if action == "cancel" else client.job(job_id)
        return _ok(_job_payload(status, backend))
    except OcrError as exc:
        return _err(exc.to_dict())
    except Exception as exc:  # noqa: BLE001
        return _err({"code": "internal", "message": f"{type(exc).__name__}: {exc}", "retryable": False})


def ocr_queue(args: dict, **kwargs: Any) -> str:
    """List, inspect, decide or export the human review queue."""
    try:
        queue = Queue()
        action = str(args.get("action") or "list").lower()
        if action == "stats":
            return _ok({"stats": queue.stats()})
        if action == "list":
            limit = int(args.get("limit") or 20)
            return _ok({"items": queue.pending(limit), "stats": queue.stats()})
        if action == "export":
            path, rows = queue.export_csv()
            return _ok({"path": path, "rows": rows})
        if action == "add":
            item_id = queue.add(
                source=_require(args, "source"),
                reason="agent_flagged",
                content=str(args.get("content") or ""),
                page=_opt_int(args.get("page")),
                label=str(args.get("label") or ""),
                score=_opt_float(args.get("score")),
            )
            return _ok({"item": queue.get(item_id), "stats": queue.stats()})
        item_id = _opt_int(args.get("item_id"))
        if item_id is None:
            raise RequestRejected("item_id is required for show / accept / reject")
        if action == "show":
            item = queue.get(item_id)
            if item is None:
                raise RequestRejected(f"no queue item {item_id}")
            return _ok({"item": item})
        if action in ("accept", "reject"):
            decided = queue.decide(
                item_id,
                "accepted" if action == "accept" else "rejected",
                str(args.get("note") or ""),
            )
            if decided is None:
                raise RequestRejected(f"no pending item {item_id}")
            return _ok({"item": decided, "stats": queue.stats()})
        raise RequestRejected("action must be list, show, accept, reject, add, stats or export")
    except OcrError as exc:
        return _err(exc.to_dict())
    except Exception as exc:  # noqa: BLE001
        return _err({"code": "internal", "message": f"{type(exc).__name__}: {exc}", "retryable": False})


# --------------------------------------------------------------------------- shaping


def _shape(
    pages: list[Page], markdown: str, *, mode: str, max_chars: int, backend: Backend
) -> dict[str, Any]:
    failed = [(p.index, b) for p in pages for b in p.failed]
    labels: dict[str, int] = {}
    for page in pages:
        for block in page.blocks:
            labels[block.label] = labels.get(block.label, 0) + 1
    payload: dict[str, Any] = {
        "backend": backend.name,
        "blocks_available": backend.blocks,
        "pages": len(pages),
        "regions": sum(len(p.blocks) for p in pages),
        "failed_regions": len(failed),
        "skipped_regions": sum(len(p.skipped) for p in pages),
        "block_labels": labels,
        "page_indexes": [p.index for p in pages],
    }
    if failed:
        payload["failed"] = [
            {
                "page": index,
                "label": block.label,
                "status": block.status,
                "bbox": list(block.bbox) if block.bbox else None,
                "preview": _excerpt(block.content, 120),
            }
            for index, block in failed[:20]
        ]
        payload["next"] = "Re-read these regions with a tighter crop, then check the text with ocr_verify."
    if not backend.blocks:
        payload["note"] = (
            "This backend has no layout stage, so regions and bboxes are unavailable; "
            "treat the page as one region and verify with ocr_verify."
        )
    if mode == "markdown":
        payload["markdown"], payload["truncated"] = _clip(markdown, max_chars)
    elif mode == "blocks":
        payload["blocks"] = [
            {
                "page": page.index,
                "order": block.order,
                "label": block.label,
                "status": block.status,
                "bbox": list(block.bbox) if block.bbox else None,
                "content": _excerpt(block.content, 400),
            }
            for page in pages
            for block in page.blocks
        ][:200]
    else:
        payload["preview"] = _excerpt(markdown, 600)
    return payload


def _job_payload(status: JobStatus, backend: Backend) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "job_id": status.id,
        "status": status.status,
        "done": status.terminal,
        "completed": status.ok,
        "pages_done": status.pages_done,
        "pages_total": status.pages_total,
        "fraction": round(status.fraction, 3) if status.fraction is not None else None,
        "backend": backend.name,
        "surface": backend.surface,
    }
    if status.pages:
        payload["pages"] = len(status.pages)
    if status.markdown:
        payload["markdown_chars"] = len(status.markdown)
    return payload


def _advice(passed: bool, score: float, threshold: float, verdict: str = "scored") -> str:
    if verdict and verdict != "scored":
        return (
            f"Jev did not produce a score (verdict {verdict!r}). Treat the text as unverified, "
            "re-crop the region, and try again; do not quote it."
        )
    if passed:
        return "Accepted: the text matches the region. Quote it if the surrounding context also checks out."
    return (
        f"Below threshold ({score:.2f} < {threshold:.2f}). Re-crop the region tighter and read it again, "
        "or ask the user instead of quoting it."
    )


# --------------------------------------------------------------------------- config


_DEFAULTS: dict[str, Any] = {
    "backend": "auto",
    "local_api": "auto",
    "base_url": "",
    "parser_model": "MiniCPM-OCR-1.0",
    "jev_model": "MiniCPM-OCR-Jev",
    "jev_base_url": "",
    "jev_api": "auto",
    "threshold": 0.6,
}


def _settings() -> dict[str, Any]:
    """Read plugin settings through the keyed Hermes route.

    ``PluginContext.get_config(key, default)`` is the public contract; calling it
    with no key raises and made earlier builds silently ignore every setting.
    """
    if _CTX is None:
        return {}
    out: dict[str, Any] = {}
    for key, default in _DEFAULTS.items():
        try:
            value = _CTX.get_config(key, default)
        except TypeError:
            # A minimal stub (or an old build) may only expose the no-argument
            # form; fall back to its mapping once, then keep the keyed contract.
            try:
                legacy = _CTX.get_config()
            except Exception:  # noqa: BLE001
                legacy = None
            if isinstance(legacy, Mapping):
                out.update({str(k): v for k, v in legacy.items()})
            continue
        except Exception:  # noqa: BLE001 — config must never fail a tool call
            continue
        out[key] = default if value is None else value
    return out


def _api_key(settings: Mapping[str, Any]) -> str:
    """Read the key the way Hermes expects: secret route first, never .env by hand."""
    try:
        from agent.secret_scope import get_secret

        value = get_secret(_API_KEY_NAME)
        if value:
            return str(value).strip()
    except Exception:  # noqa: BLE001 — outside Hermes, or no secret route in this build
        pass
    # Standalone/dev only: the same variable get_secret reads inside Hermes.
    return str(settings.get("api_key") or os.environ.get(_API_KEY_NAME) or "").strip()


def _client() -> tuple[OcrClient, Backend]:
    settings = _settings()
    key = _api_key(settings)
    if key:
        settings = {**settings, "_api_key": key}
    for candidate in candidates(settings):
        if candidate.resolved_kind == "modelbest" and not candidate.api_key:
            continue
        if candidate.source == "loopback" and not reachable(candidate):
            continue
        return candidate.client(), candidate
    raise BackendUnavailable(
        "No MiniCPM-OCR backend. Set a ModelBest API key, point base_url at a local vLLM/Ollama "
        "server, or start one on 127.0.0.1:8000 / :11434."
    )


def _job_timeout() -> float:
    try:
        return float(os.environ.get("HERMES_MINICPM_OCR_JOB_TIMEOUT") or 180)
    except ValueError:
        return 180.0


# --------------------------------------------------------------------------- helpers


def _require(args: Mapping[str, Any], key: str) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RequestRejected(f"{key} is required")
    return value.strip()


def _opt_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        raise RequestRejected(f"expected an integer, got {value!r}") from None


def _opt_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        raise RequestRejected(f"expected a number, got {value!r}") from None


def _excerpt(text: str, limit: int = 300) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _clip(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[: limit - 1] + "…", True


def _ok(payload: Mapping[str, Any]) -> str:
    return json.dumps({"ok": True, **payload}, ensure_ascii=False)


def _err(error: Mapping[str, Any]) -> str:
    return json.dumps({"ok": False, "error": dict(error)}, ensure_ascii=False)
