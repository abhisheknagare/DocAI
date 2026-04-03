"""
PDF / HTM Text Extractor
Converts SEC filings (PDF or HTML) into clean plain text.
"""

import re
import os
import json
from pathlib import Path
from typing import Optional
import pdfplumber


# ── helpers ──────────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Remove noise common in SEC filings."""
    # Collapse excessive whitespace
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Collapse many blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove page-break artifacts
    text = re.sub(r"(Table of Contents\n?)+", "", text, flags=re.IGNORECASE)
    # Remove isolated single characters (table noise)
    text = re.sub(r"(?<!\w)\.(?!\w)", "", text)
    return text.strip()


def extract_pdf(path: str) -> str:
    """Extract text from a PDF file using pdfplumber."""
    pages_text = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text(x_tolerance=3, y_tolerance=3)
            if page_text:
                pages_text.append(page_text)
    return _clean_text("\n\n".join(pages_text))


def extract_htm(path: str) -> str:
    """Extract text from an HTM/HTML SEC filing."""
    try:
        from html.parser import HTMLParser

        class _Extractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self.parts = []
                self._skip = False

            def handle_starttag(self, tag, attrs):
                if tag in ("script", "style", "head"):
                    self._skip = True

            def handle_endtag(self, tag):
                if tag in ("script", "style", "head"):
                    self._skip = False
                if tag in ("p", "div", "br", "tr", "li"):
                    self.parts.append("\n")

            def handle_data(self, data):
                if not self._skip and data.strip():
                    self.parts.append(data)

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            html = f.read()

        parser = _Extractor()
        parser.feed(html)
        return _clean_text("".join(parser.parts))
    except Exception as e:
        print(f"  ⚠ HTM extraction error: {e}")
        return ""


def extract_file(path: str) -> Optional[str]:
    """
    Auto-detect file type and extract text.
    Returns None if extraction fails.
    """
    ext = Path(path).suffix.lower()
    try:
        if ext == ".pdf":
            text = extract_pdf(path)
        elif ext in (".htm", ".html"):
            text = extract_htm(path)
        else:
            print(f"  ⚠ Unsupported file type: {ext}")
            return None

        if len(text) < 100:
            print(f"  ⚠ Very short text extracted from {path} ({len(text)} chars)")
            return None

        return text
    except Exception as e:
        print(f"  ⚠ Extraction failed for {path}: {e}")
        return None


def extract_directory(
    input_dir: str,
    output_dir: str,
    overwrite: bool = False,
) -> list[dict]:
    """
    Extract text from all PDFs/HTMs in input_dir.
    Saves .txt files to output_dir.
    Returns list of metadata dicts.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []

    files = [
        f for f in Path(input_dir).iterdir()
        if f.suffix.lower() in (".pdf", ".htm", ".html")
    ]

    print(f"\n📄 Extracting {len(files)} files from {input_dir}")

    for fpath in files:
        out_name = fpath.stem + ".txt"
        out_path = Path(output_dir) / out_name

        if out_path.exists() and not overwrite:
            print(f"  Skip (exists): {out_name}")
            results.append(
                {
                    "source": str(fpath),
                    "output": str(out_path),
                    "status": "cached",
                }
            )
            continue

        print(f"  Extracting: {fpath.name}")
        text = extract_file(str(fpath))
        if text:
            out_path.write_text(text, encoding="utf-8")
            results.append(
                {
                    "source": str(fpath),
                    "output": str(out_path),
                    "chars": len(text),
                    "status": "ok",
                }
            )
            print(f"    ✓ {len(text):,} chars → {out_name}")
        else:
            results.append(
                {"source": str(fpath), "output": None, "status": "failed"}
            )

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract text from SEC filing PDFs")
    parser.add_argument("--input", default="data/raw", help="Input directory with PDFs")
    parser.add_argument("--output", default="data/processed", help="Output directory")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    results = extract_directory(args.input, args.output, overwrite=args.overwrite)
    ok = [r for r in results if r["status"] == "ok"]
    print(f"\n✅ Extracted {len(ok)}/{len(results)} files")
