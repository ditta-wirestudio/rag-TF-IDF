"""Turn an uploaded file's bytes into plain text for ingestion.

Supports PDF (text-based), Markdown, and plain text. Scanned/image-only PDFs
have no embedded text — those need OCR, which is out of scope here.
"""
from __future__ import annotations

import io


def extract_text(data: bytes, filename: str) -> str:
    name = (filename or "").lower()

    if name.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
        return "\n\n".join(p for p in pages if p)

    # md / txt / anything text-like
    return data.decode("utf-8", errors="ignore")
