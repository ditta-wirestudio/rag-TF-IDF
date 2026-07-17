"""Bulk-ingest a folder of .pdf/.md/.txt files. Filenames become the `source`.

    python ingest_folder.py ../sample_docs
    python ingest_folder.py ~/Documents/my_pdfs
"""
from __future__ import annotations

import sys
from pathlib import Path

import rag
from loaders import extract_text


def main(folder: str) -> None:
    paths = [p for ext in ("*.pdf", "*.md", "*.txt")
             for p in Path(folder).glob(ext)]
    if not paths:
        print(f"no .pdf/.md/.txt files in {folder}")
        return
    for p in sorted(paths):
        text = extract_text(p.read_bytes(), p.name)
        if not text.strip():
            print(f"skip {p.name}: no extractable text (scanned PDF?)")
            continue
        res = rag.ingest(source=p.name, text=text, title=p.stem)
        print(f"ingested {p.name}: {res['chunks']} chunks")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../sample_docs")
