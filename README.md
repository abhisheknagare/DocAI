# 📊 FinancialRAG — SEC Filing Intelligence Platform

A production-grade RAG (Retrieval-Augmented Generation) system for analyzing SEC 10-K filings.
Ask questions, generate summaries, extract financial metrics, and compare companies — all grounded in real documents.

---

## Architecture

```
SEC EDGAR PDFs
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE                          │
│  PDF → Text Extraction → Chunking (600-800 tok) → Embedding     │
│  (pdfplumber)           (overlap=150)             (TF-IDF+SVD   │
│                                                    or MiniLM)   │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
                     ┌─────────────────┐
                     │  FAISS Index    │  ← vector store (disk)
                     │  + metadata     │    + TF-IDF model pickle
                     └────────┬────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
         User Query    Guardrail          Cache
         → Embed       (input check)    (SHA-256 key)
              │               │               │
              └───────────────┴───────────────┘
                              │
                              ▼
                     Top-K Chunk Retrieval
                     (FAISS inner product)
                              │
                              ▼
                     Confidence Scoring
                     ┌──────────────────┐
                     │ Retrieval conf   │
                     │ Coverage score   │
                     │ Groundedness     │
                     │ Consistency      │
                     │ → Overall 0–1    │
                     └──────────────────┘
                              │
                              ▼
                     Claude API (Sonnet 4)
                     "Answer ONLY from context"
                              │
                              ▼
                     Output Guardrail
                     (hallucination / low-conf check)
                              │
                              ▼
                     Monitor Log (JSONL)
                     + Streamlit UI Response
```

---

## Features

| Feature | Description |
|---|---|
| **Q&A** | Ask any financial question, answered strictly from indexed documents |
| **Summarization** | Executive summary across 5 sections (Business, Financials, Risks, Strategy, Outlook) |
| **Metric Extraction** | Structured JSON: revenue, net income, EPS, margins, risks, segments |
| **Cross-doc Compare** | Side-by-side analysis of two companies on any dimension |
| **Confidence Scores** | 4-component score: retrieval, coverage, groundedness, consistency |
| **Guardrails** | Input: off-topic / injection detection. Output: hallucination flag, low-conf block |
| **Monitoring** | Real-time dashboard: latency, cache rate, guardrail triggers, query trends |
| **Caching** | SHA-256 keyed disk cache — identical queries return instantly |
| **Source Citation** | Every answer shows which chunks (company, section, score) it came from |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set API key

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Download & ingest SEC filings

```bash
# Downloads Apple, Tesla, Microsoft 10-Ks from SEC EDGAR, then indexes them
python scripts/ingest.py --companies apple tesla microsoft

# Options:
python scripts/ingest.py --companies apple --form 10-Q --filings-per-company 2
python scripts/ingest.py --skip-download   # use existing PDFs in data/raw/
```

### 4. Launch the UI

```bash
streamlit run app/streamlit_app.py
```

Open http://localhost:8501

---

## Sample Queries

**Q&A Tab:**
- *"What was Apple's total revenue and gross margin in fiscal 2024?"*
- *"What are Tesla's top 3 risk factors?"*
- *"How much did Microsoft's cloud segment grow year-over-year?"*
- *"What is Apple's diluted EPS and how does it compare to prior year?"*
- *"Describe Microsoft's AI strategy and OpenAI partnership"*

**Compare Tab:**
- Apple vs Tesla → "Revenue and profitability"
- Apple vs Microsoft → "Capital allocation and shareholder returns"
- Tesla vs Microsoft → "R&D investment and innovation"

---

## Confidence Score System

Every Q&A response includes a 4-component confidence breakdown:

```
Overall Confidence = 0.30 × Retrieval
                   + 0.20 × Coverage
                   + 0.35 × Groundedness
                   + 0.15 × Consistency
```

| Component | What it measures |
|---|---|
| **Retrieval Confidence** | Cosine similarity of top-k chunks to query |
| **Coverage Score** | Fraction of chunks above quality threshold (0.55) |
| **Groundedness** | Lexical + sentence overlap between answer and context |
| **Consistency** | Topical agreement across retrieved chunks |

Thresholds: ✅ ≥75% · ⚠️ 52–74% · 🔶 30–51% · ❌ <30%

---

## Guardrails

**Input guardrails** (block before generation):
- Off-topic queries (recipes, sports, politics) → blocked
- Prompt injection patterns → blocked
- Empty / malformed queries → blocked

**Output guardrails** (check after generation):
- Low confidence (<25%) → response blocked with explanation
- Numbers in answer not in context → hallucination warning
- Speculation language (≥2 hedging phrases) → warning
- Moderate confidence (25–48%) → pass-through with warning

---

## Monitoring Dashboard

The Monitor tab shows:
- **Session KPIs**: total queries, avg latency, cache hit rate, avg confidence
- **Recent query log**: per-query confidence, groundedness, latency, cache status
- **Guardrail events**: what triggered and why
- **Mode distribution**: Q&A vs Summarize vs Extract vs Compare
- **Top queries**: most frequently asked questions

All events are persisted to `data/monitoring/query_log.jsonl`.

---

## Project Structure

```
financial-rag/
├── app/
│   └── streamlit_app.py          ← Full UI (5 tabs)
├── src/
│   ├── ingestion/
│   │   ├── downloader.py         ← SEC EDGAR downloader
│   │   ├── extractor.py          ← PDF/HTM → text
│   │   └── chunker.py            ← Text → overlapping chunks
│   ├── embeddings/
│   │   └── embedder.py           ← TF-IDF+SVD or sentence-transformers
│   ├── vectorstore/
│   │   └── faiss_store.py        ← FAISS index + metadata
│   ├── rag/
│   │   └── pipeline.py           ← Full RAG orchestration
│   ├── evaluation/
│   │   └── confidence.py         ← 4-component confidence scorer
│   ├── guardrails/
│   │   └── guardrails.py         ← Input + output guardrails
│   └── monitoring/
│       └── monitor.py            ← Query logging + metrics
├── scripts/
│   └── ingest.py                 ← End-to-end ingestion CLI
├── data/
│   ├── raw/                      ← Downloaded PDFs
│   ├── processed/                ← Extracted .txt files
│   │   └── chunks/               ← Chunked .jsonl files
│   ├── index/                    ← FAISS index + metadata + config
│   ├── cache/                    ← Response cache (SHA-256 keyed JSON)
│   └── monitoring/               ← Query event log (JSONL)
└── requirements.txt
```

---

## Deploy to Streamlit Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set main file: `app/streamlit_app.py`
4. Add secret: `ANTHROPIC_API_KEY = sk-ant-...`
5. Deploy — users upload their own PDFs via sidebar

---

## Supported Companies (SEC EDGAR)

| Company | Ticker | CIK |
|---|---|---|
| Apple | AAPL | 0000320193 |
| Tesla | TSLA | 0001318605 |
| Microsoft | MSFT | 0000789019 |
| Amazon | AMZN | 0001018724 |
| Alphabet | GOOGL | 0001652044 |
| NVIDIA | NVDA | 0001045810 |
| Meta | META | 0001326801 |

Add more by editing `COMPANY_CIKS` in `src/ingestion/downloader.py`.
