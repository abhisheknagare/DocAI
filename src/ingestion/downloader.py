"""
SEC EDGAR Downloader
Downloads 10-K and 10-Q filings as PDFs from SEC EDGAR.
"""

import os
import time
import json
import requests
from pathlib import Path
from typing import Optional
from tqdm import tqdm


EDGAR_BASE = "https://data.sec.gov"
HEADERS = {
    "User-Agent": "FinancialRAG research@example.com",
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}

# CIK numbers for popular companies
COMPANY_CIKS = {
    "apple": "0000320193",
    "tesla": "0001318605",
    "amazon": "0001018724",
    "microsoft": "0000789019",
    "alphabet": "0001652044",
    "nvidia": "0001045810",
    "meta": "0001326801",
}


def get_company_filings(cik: str, form_type: str = "10-K", limit: int = 3) -> list[dict]:
    """Fetch recent filings metadata from SEC EDGAR."""
    url = f"{EDGAR_BASE}/submissions/CIK{cik}.json"
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    data = r.json()

    filings = data["filings"]["recent"]
    results = []
    for i, form in enumerate(filings["form"]):
        if form == form_type and len(results) < limit:
            acc_no = filings["accessionNumber"][i]
            results.append(
                {
                    "company": data["name"],
                    "cik": cik,
                    "form": form,
                    "date": filings["filingDate"][i],
                    "accessionNumber": acc_no,
                    "accessionPath": acc_no.replace("-", ""),
                }
            )
    return results


def get_filing_documents(cik: str, acc_path: str) -> list[dict]:
    """Get list of documents in a filing."""
    url = f"{EDGAR_BASE}/Archives/edgar/data/{cik.lstrip('0')}/{acc_path}/index.json"
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    data = r.json()
    return data.get("directory", {}).get("item", [])


def download_filing_pdf(
    cik: str, acc_path: str, documents: list[dict], output_dir: str
) -> Optional[str]:
    """Download the primary PDF or HTM document from a filing."""
    os.makedirs(output_dir, exist_ok=True)
    cik_bare = cik.lstrip("0")

    # Prefer PDF, then HTM
    for ext in [".pdf", ".htm", ".html"]:
        for doc in documents:
            name = doc.get("name", "")
            doc_type = doc.get("type", "")
            if name.endswith(ext) and doc_type in ("10-K", "10-Q", ""):
                url = f"{EDGAR_BASE}/Archives/edgar/data/{cik_bare}/{acc_path}/{name}"
                dest = os.path.join(output_dir, f"{cik_bare}_{acc_path}_{name}")
                if os.path.exists(dest):
                    print(f"  Already exists: {dest}")
                    return dest
                r = requests.get(url, headers=HEADERS, timeout=60, stream=True)
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"  Downloaded: {dest}")
                time.sleep(0.5)  # EDGAR rate limit courtesy
                return dest
    return None


def download_company_filings(
    company_name: str,
    output_dir: str = "data/raw",
    form_type: str = "10-K",
    limit: int = 2,
) -> list[str]:
    """
    Main entry point: download filings for a named company.

    Usage:
        paths = download_company_filings("apple", "data/raw", limit=2)
    """
    cik = COMPANY_CIKS.get(company_name.lower())
    if not cik:
        raise ValueError(
            f"Unknown company '{company_name}'. "
            f"Available: {list(COMPANY_CIKS.keys())}"
        )

    print(f"\n📥 Fetching {form_type} filings for {company_name.upper()} (CIK {cik})")
    filings = get_company_filings(cik, form_type=form_type, limit=limit)

    if not filings:
        print(f"  No {form_type} filings found.")
        return []

    downloaded = []
    for filing in tqdm(filings, desc=f"Downloading {company_name}"):
        print(f"\n  Filing: {filing['form']} dated {filing['date']}")
        try:
            docs = get_filing_documents(cik, filing["accessionPath"])
            path = download_filing_pdf(cik, filing["accessionPath"], docs, output_dir)
            if path:
                downloaded.append(path)
        except Exception as e:
            print(f"  ⚠ Error downloading filing: {e}")
        time.sleep(1)

    return downloaded


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download SEC EDGAR filings")
    parser.add_argument(
        "--companies",
        nargs="+",
        default=["apple", "tesla", "microsoft"],
        help="Company names to download",
    )
    parser.add_argument("--output", default="data/raw", help="Output directory")
    parser.add_argument("--form", default="10-K", help="Form type (10-K or 10-Q)")
    parser.add_argument("--limit", type=int, default=1, help="Filings per company")
    args = parser.parse_args()

    all_paths = []
    for company in args.companies:
        try:
            paths = download_company_filings(
                company, args.output, form_type=args.form, limit=args.limit
            )
            all_paths.extend(paths)
        except Exception as e:
            print(f"⚠ Skipping {company}: {e}")

    print(f"\n✅ Downloaded {len(all_paths)} filings to {args.output}")
    for p in all_paths:
        print(f"  {p}")
