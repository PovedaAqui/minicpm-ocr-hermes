"""Model-facing tool schemas.

Flat OpenAI-style function schemas, the shape ``ctx.register_tool`` expects.
Descriptions are the whole game here: they are how the model decides when to
reach for OCR instead of guessing.
"""

from __future__ import annotations

OCR_READ = {
    "name": "ocr_read",
    "description": (
        "Read a document image with MiniCPM-OCR and get back Markdown plus per-region status. "
        "Use it whenever the answer depends on what a page or screenshot actually says — invoices, "
        "receipts, contracts, forms, scanned letters, tables — instead of asking a vision model to "
        "describe the file. Each region carries status ok/failed/skipped_no_vlm; anything not ok, "
        "or any region whose text matters, should be checked with ocr_verify before you quote it. "
        "Accepts one image or PDF; a PDF that outlives the wait returns a job_id for ocr_job."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute path to the image, or a public https URL.",
            },
            "mode": {
                "type": "string",
                "enum": ["summary", "markdown", "blocks"],
                "default": "summary",
                "description": (
                    "summary: counts, labels and failed regions only (cheap, use first). "
                    "markdown: the page text in reading order. blocks: every region with its bbox."
                ),
            },
            "max_chars": {
                "type": "integer",
                "default": 20000,
                "minimum": 500,
                "description": "Cap on returned text; longer output is truncated with a notice.",
            },
            "queue": {
                "type": "boolean",
                "default": True,
                "description": (
                    "Push every region with a non-ok status into the human review queue. "
                    "Leave it on unless you are parsing a throwaway document."
                ),
            },
            "wait": {
                "type": "boolean",
                "default": True,
                "description": (
                    "PDFs and local document services only: wait up to three minutes for the job. "
                    "With false, returns a job_id immediately and you poll it with ocr_job. A job "
                    "still running after the wait returns its handle either way."
                ),
            },
            "upright": {
                "type": "boolean",
                "default": True,
                "description": (
                    "Document backends only: let the service detect and correct page orientation "
                    "before recognition. Turn it off for already-upright scans."
                ),
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    },
}

OCR_JOB = {
    "name": "ocr_job",
    "description": (
        "Check or cancel an asynchronous MiniCPM-OCR parse job (PDFs and long documents run as jobs). "
        "Returns status, pages_done/pages_total and, once complete, how many pages came back. Use it when "
        "ocr_read handed you a job_id instead of waiting, or to cancel a job that is no longer needed."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "job_id": {
                "type": "string",
                "description": "The job id returned when the document was submitted.",
            },
            "action": {
                "type": "string",
                "enum": ["status", "cancel"],
                "default": "status",
                "description": "status: poll for progress and results. cancel: stop the job.",
            },
        },
        "required": ["job_id"],
        "additionalProperties": False,
    },
}

OCR_QUEUE = {
    "name": "ocr_queue",
    "description": (
        "The human review queue: regions the parser could not read, or that scored below threshold. "
        "Use it to list what is pending, show one item, accept or reject it with a note, add a region the "
        "agent could not resolve, or export the whole queue as CSV. Anything you cannot verify should end "
        "up here instead of in a confident answer."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "show", "accept", "reject", "add", "stats", "export"],
                "default": "list",
                "description": "list: pending items. show: one item by id. accept/reject: decide one item.",
            },
            "item_id": {
                "type": "integer",
                "description": "Queue item id, for show / accept / reject.",
            },
            "limit": {
                "type": "integer",
                "default": 20,
                "description": "Maximum items returned by list.",
            },
            "note": {
                "type": "string",
                "description": "Why it was accepted or rejected; stored with the decision.",
            },
            "source": {
                "type": "string",
                "description": "For add: the document the region came from.",
            },
            "page": {"type": "integer", "description": "For add: zero-based page index."},
            "label": {"type": "string", "description": "For add: region label, e.g. text or table."},
            "content": {"type": "string", "description": "For add: the text you are unsure about."},
            "score": {"type": "number", "description": "For add: the Jev score you measured, if any."},
        },
        "required": [],
        "additionalProperties": False,
    },
}

OCR_VERIFY = {
    "name": "ocr_verify",
    "description": (
        "Score whether a piece of text really matches an image region, 0 to 1, with MiniCPM-OCR Jev. "
        "Jev is model-agnostic: it grades text produced by any parser or by you. Use it before quoting a "
        "number, date, name or total you read from a document, before telling the user a transcription is "
        "correct, or to compare two candidate readings of the same region. A low score is a reason to "
        "re-crop and re-read, not a reason to guess."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Absolute path to the region image (a tight crop reads best), or an https URL."
                ),
            },
            "text": {
                "type": "string",
                "description": "The candidate text to check against that region.",
            },
            "category": {
                "type": "string",
                "enum": ["text", "formula", "table"],
                "default": "text",
                "description": "Region type: plain text, a formula in LaTeX, or a table in HTML.",
            },
            "threshold": {
                "type": "number",
                "default": 0.6,
                "description": "Score at or above which the text is accepted.",
            },
        },
        "required": ["path", "text"],
        "additionalProperties": False,
    },
}
