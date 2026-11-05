"""MiniCPM-OCR for Hermes — registration.

Five tools are planned; 0.1.0 ships the two that carry the idea: read a document
and score whether a piece of text really matches a region.
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
