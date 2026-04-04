import re
import json
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict


@dataclass
class Chunk:
    chunk_id: str          
    doc_id: str            
    filename: str          
    company: str          
    text: str              
    chunk_index: int       
    total_chunks: int     
    char_start: int        
    char_end: int         
    section: str           

    def to_dict(self) -> dict:
        return asdict(self)


SECTION_PATTERNS = {
    "Business Overview": r"(item\s*1[.\s]|business overview|about.*company)",
    "Risk Factors": r"(item\s*1a[.\s]|risk factor)",
    "MD&A": r"(item\s*7[.\s]|management.{0,10}discussion|MD&A)",
    "Financial Statements": r"(item\s*8[.\s]|financial statement|consolidated balance)",
    "Revenue & Earnings": r"(revenue|net sales|gross margin|earnings per share|EPS)",
    "Liquidity": r"(item\s*7a[.\s]|liquidity|capital resource)",
    "Executive Compensation": r"(item\s*11|executive compensation|named executive)",
    "Legal Proceedings": r"(item\s*3[.\s]|legal proceeding|litigation)",
    "Forward Looking": r"(forward.looking|safe harbor|cautionary)",
}


def _infer_section(text: str) -> str:
    
    lower = text[:500].lower()
    for section, pattern in SECTION_PATTERNS.items():
        if re.search(pattern, lower, re.IGNORECASE):
            return section
    return "General"


def _infer_company(filename: str) -> str:
    
    name = Path(filename).stem.lower()
    mapping = {
        "apple": "Apple Inc.",
        "aapl": "Apple Inc.",
        "tesla": "Tesla Inc.",
        "tsla": "Tesla Inc.",
        "microsoft": "Microsoft Corp.",
        "msft": "Microsoft Corp.",
        "amazon": "Amazon.com Inc.",
        "amzn": "Amazon.com Inc.",
        "alphabet": "Alphabet Inc.",
        "googl": "Alphabet Inc.",
        "nvidia": "NVIDIA Corp.",
        "nvda": "NVIDIA Corp.",
        "meta": "Meta Platforms Inc.",
    }
    for key, company in mapping.items():
        if key in name:
            return company
    return Path(filename).stem.replace("_", " ").title()


def chunk_text(
    text: str,
    doc_id: str,
    filename: str,
    chunk_size: int = 800,
    overlap: int = 150,
) -> list[Chunk]:
    
    company = _infer_company(filename)
    chunks = []
    start = 0
    idx = 0

    total_est = max(1, (len(text) - overlap) // (chunk_size - overlap))

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            boundary = text.rfind(". ", end - 200, end)
            if boundary > start + chunk_size // 2:
                end = boundary + 1

        chunk_text_str = text[start:end].strip()
        if len(chunk_text_str) < 50:
            break

        chunk = Chunk(
            chunk_id=f"{doc_id}__{idx:04d}",
            doc_id=doc_id,
            filename=filename,
            company=company,
            text=chunk_text_str,
            chunk_index=idx,
            total_chunks=total_est,
            char_start=start,
            char_end=end,
            section=_infer_section(chunk_text_str),
        )
        chunks.append(chunk)

        start = end - overlap
        idx += 1

    for c in chunks:
        c.total_chunks = len(chunks)

    return chunks


def chunk_directory(
    input_dir: str,
    output_dir: str,
    chunk_size: int = 800,
    overlap: int = 150,
    overwrite: bool = False,
) -> list[Chunk]:
    
    os.makedirs(output_dir, exist_ok=True)
    all_chunks = []

    txt_files = list(Path(input_dir).glob("*.txt"))
    print(f"\n✂️  Chunking {len(txt_files)} documents from {input_dir}")

    for fpath in txt_files:
        doc_id = fpath.stem
        out_path = Path(output_dir) / f"{doc_id}_chunks.jsonl"

        if out_path.exists() and not overwrite:
            print(f"  Skip (exists): {out_path.name}")
            with open(out_path) as f:
                for line in f:
                    d = json.loads(line)
                    all_chunks.append(Chunk(**d))
            continue

        text = fpath.read_text(encoding="utf-8")
        if not text.strip():
            print(f"  Skip (empty): {fpath.name}")
            continue

        chunks = chunk_text(
            text, doc_id=doc_id, filename=fpath.name,
            chunk_size=chunk_size, overlap=overlap,
        )
        all_chunks.extend(chunks)

        with open(out_path, "w") as f:
            for c in chunks:
                f.write(json.dumps(c.to_dict()) + "\n")

        print(f"  {fpath.name} → {len(chunks)} chunks")

    print(f"  Total: {len(all_chunks)} chunks across {len(txt_files)} documents")
    return all_chunks


def load_chunks(chunk_dir: str) -> list[Chunk]:
    chunks = []
    for fpath in Path(chunk_dir).glob("*_chunks.jsonl"):
        with open(fpath) as f:
            for line in f:
                d = json.loads(line)
                chunks.append(Chunk(**d))
    return chunks


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chunk extracted text files")
    parser.add_argument("--input", default="data/processed")
    parser.add_argument("--output", default="data/processed/chunks")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=150)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    chunks = chunk_directory(
        args.input, args.output,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        overwrite=args.overwrite,
    )
    print(f"\n✅ Created {len(chunks)} chunks")
