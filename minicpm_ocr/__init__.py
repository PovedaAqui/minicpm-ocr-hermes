"""MiniCPM-OCR for Hermes — registration.

Four tools carry the loop: read a document, poll its job, score whether a
region's text really matches the image, and keep what could not be read in a
durable human review queue.
"""

from __future__ import annotations

from . import schemas, tools


def register(ctx) -> None:
    """Wire schemas to handlers. Called once, at plugin load."""
    tools.set_context(ctx)
    ctx.register_tool(
        name="ocr_read",
        toolset="minicpm-ocr",
        schema=schemas.OCR_READ,
        handler=tools.ocr_read,
    )
    ctx.register_tool(
        name="ocr_job",
        toolset="minicpm-ocr",
        schema=schemas.OCR_JOB,
        handler=tools.ocr_job,
    )
    ctx.register_tool(
        name="ocr_queue",
        toolset="minicpm-ocr",
        schema=schemas.OCR_QUEUE,
        handler=tools.ocr_queue,
    )
    ctx.register_tool(
        name="ocr_verify",
        toolset="minicpm-ocr",
        schema=schemas.OCR_VERIFY,
        handler=tools.ocr_verify,
    )
