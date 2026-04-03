"""
Ingestion Pipeline
Runs the full ingestion: Download → Extract → Chunk → Embed → Index

Usage:
    python scripts/ingest.py                          # Download + ingest Apple, Tesla, Microsoft
    python scripts/ingest.py --companies apple tesla  # Specific companies
    python scripts/ingest.py --pdf-dir data/raw       # Use already-downloaded PDFs
    python scripts/ingest.py --skip-download          # Use existing PDFs
"""

import sys
import os
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser(description="Financial RAG ingestion pipeline")
    p.add_argument("--companies", nargs="+", default=["apple", "tesla", "microsoft"],
                   help="Companies to download from SEC EDGAR")
    p.add_argument("--form", default="10-K", help="SEC form type (10-K or 10-Q)")
    p.add_argument("--filings-per-company", type=int, default=1)
    p.add_argument("--pdf-dir", default="data/raw")
    p.add_argument("--text-dir", default="data/processed")
    p.add_argument("--chunk-dir", default="data/processed/chunks")
    p.add_argument("--index-dir", default="data/index")
    p.add_argument("--chunk-size", type=int, default=800)
    p.add_argument("--chunk-overlap", type=int, default=150)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--embedding-model", default="minilm",
                   help="Embedding model: minilm, mpnet, bge-small, e5-small")
    p.add_argument("--skip-download", action="store_true",
                   help="Skip download; use existing PDFs in --pdf-dir")
    p.add_argument("--skip-extract", action="store_true")
    p.add_argument("--skip-chunk", action="store_true")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-process existing files")
    return p.parse_args()


def main():
    args = parse_args()
    print("\n" + "="*60)
    print("  Financial RAG Ingestion Pipeline")
    print("="*60)

    #Step 1: Download
    if not args.skip_download:
        print("\n[1/4] DOWNLOADING SEC EDGAR FILINGS")
        from src.ingestion.downloader import download_company_filings
        for company in args.companies:
            try:
                download_company_filings(
                    company,
                    output_dir=args.pdf_dir,
                    form_type=args.form,
                    limit=args.filings_per_company,
                )
            except Exception as e:
                print(f"  ⚠ Could not download {company}: {e}")
    else:
        print("\n[1/4] SKIPPING DOWNLOAD (--skip-download)")

    #Step 2: Extract text 
    if not args.skip_extract:
        print("\n[2/4] EXTRACTING TEXT FROM PDFs")
        from src.ingestion.extractor import extract_directory
        results = extract_directory(
            args.pdf_dir, args.text_dir, overwrite=args.overwrite
        )
        ok = [r for r in results if r["status"] in ("ok", "cached")]
        print(f"  {len(ok)}/{len(results)} documents extracted")
    else:
        print("\n[2/4] SKIPPING EXTRACTION (--skip-extract)")

    #Step 3: Chunk 
    if not args.skip_chunk:
        print("\n[3/4] CHUNKING DOCUMENTS")
        from src.ingestion.chunker import chunk_directory
        chunks = chunk_directory(
            args.text_dir,
            args.chunk_dir,
            chunk_size=args.chunk_size,
            overlap=args.chunk_overlap,
            overwrite=args.overwrite,
        )
        if not chunks:
            print("  ⚠ No chunks created. Check that text files exist in data/processed/")
            sys.exit(1)
    else:
        print("\n[3/4] SKIPPING CHUNKING (--skip-chunk)")
        from src.ingestion.chunker import load_chunks
        chunks = load_chunks(args.chunk_dir)
        print(f"  Loaded {len(chunks)} chunks from {args.chunk_dir}")

    #Step 4: Embed + Index 
    print("\n[4/4] EMBEDDING + INDEXING")
    from src.embeddings.embedder import get_embedder
    from src.vectorstore.faiss_store import FAISSVectorStore

    embedder = get_embedder(args.embedding_model)

    #Fit TF-IDF fallback if needed
    texts = [c.text for c in chunks]
    embedder.fit(texts)

    print(f"  Embedding {len(chunks)} chunks (dim={embedder.dim})...")
    batch_size = 64
    all_vecs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i : i + batch_size]
        vecs = embedder.embed(batch, show_progress=False)
        all_vecs.append(vecs)

    embeddings = np.vstack(all_vecs).astype(np.float32)
    metadata = [c.to_dict() for c in chunks]

    store = FAISSVectorStore(dim=embedder.dim, index_dir=args.index_dir)
    store.build(embeddings, metadata)
    store.save()

    #Summary 
    stats = store.stats()
    print("\n" + "="*60)
    print("  ✅ Ingestion Complete")
    print("="*60)
    print(f"  Vectors indexed : {stats['total_vectors']:,}")
    print(f"  Documents       : {stats['documents']}")
    print(f"  Embedding dim   : {stats['dim']}")
    print(f"  Documents list  : {stats['document_list']}")
    print(f"\n  Run Streamlit UI: streamlit run app/streamlit_app.py")
    print("="*60)

    config = {
        "embedding_model": args.embedding_model,
        "embedding_dim": embedder.dim,
        "index_dir": args.index_dir,
        "chunk_dir": args.chunk_dir,
        "total_chunks": len(chunks),
        "documents": stats["document_list"],
    }
    Path("data/index/config.json").write_text(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
