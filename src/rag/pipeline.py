"""
Enhanced RAG Pipeline — Full Production Version
Integrates: Retrieval | Generation | Confidence | Guardrails | Monitoring | Cache
"""

import os, json, time, hashlib
import anthropic
from typing import Optional
from pathlib import Path

from src.vectorstore.faiss_store import FAISSVectorStore, SearchResult
from src.evaluation.confidence import ConfidenceScorer, ConfidenceResult
from src.guardrails.guardrails import Guardrails, GuardrailCheck
from src.monitoring.monitor import get_monitor

QA_SYSTEM_PROMPT = """You are a precise financial analyst assistant reviewing SEC filings.
RULES (follow strictly):
1. Answer using ONLY information present in the provided context
2. Cite sources using [Source N] references when stating specific figures
3. Quote specific figures exactly as they appear (e.g., "$391.0 billion", "46.2%")
4. If the answer is not in the context, say: "The provided documents don't contain this information."
5. Never speculate or add information not explicitly stated in the context
6. Always mention which company and filing year when citing numbers"""

SUMMARY_SYSTEM_PROMPT = """You are a financial analyst summarizing SEC 10-K annual reports.
Produce a structured executive summary with exactly these sections:
**1. Business Overview** — What the company does, key segments, markets
**2. Financial Highlights** — Revenue, net income, EPS, gross margin, YoY changes with exact figures
**3. Key Risks** — Top 3-5 material risks identified
**4. Strategic Priorities** — Growth initiatives, capital allocation, investments
**5. Outlook** — Forward guidance, management commentary, key watch items
Use specific numbers from the text. Be concise but comprehensive."""

EXTRACTION_SYSTEM_PROMPT = """You are a financial data extraction specialist analyzing SEC filings.
Extract ONLY information explicitly stated in the provided text. Return ONLY a valid JSON object.
Use null for any field not found in the text. Schema:
{
  "company": "string",
  "fiscal_year": "YYYY",
  "revenue": {"value": number, "unit": "billion USD", "growth_pct": number_or_null},
  "net_income": {"value": number_or_null, "unit": "billion USD", "growth_pct": number_or_null},
  "gross_margin": {"value": number_or_null, "unit": "percent"},
  "ebitda": {"value": number_or_null, "unit": "billion USD"},
  "operating_income": {"value": number_or_null, "unit": "billion USD"},
  "cash_and_equivalents": {"value": number_or_null, "unit": "billion USD"},
  "total_debt": {"value": number_or_null, "unit": "billion USD"},
  "eps_diluted": {"value": number_or_null, "unit": "USD"},
  "capex": {"value": number_or_null, "unit": "billion USD"},
  "r_and_d": {"value": number_or_null, "unit": "billion USD"},
  "key_risks": ["string"],
  "key_segments": ["string"],
  "mentioned_competitors": ["string"],
  "data_confidence": "high|medium|low"
}
Return ONLY the JSON. No markdown, no explanation."""


class ResponseCache:
    def __init__(self, cache_dir="data/cache"):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._hits = self._misses = 0

    def _key(self, query, mode, doc_id):
        raw = f"{mode}::{doc_id or 'all'}::{query.strip().lower()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:20]

    def get(self, query, mode, doc_id=None):
        p = self.dir / f"{self._key(query, mode, doc_id)}.json"
        if p.exists():
            self._hits += 1
            with open(p) as f: return json.load(f)
        self._misses += 1
        return None

    def set(self, query, mode, result, doc_id=None):
        p = self.dir / f"{self._key(query, mode, doc_id)}.json"
        with open(p, "w") as f: json.dump(result, f, indent=2)

    def stats(self):
        t = self._hits + self._misses
        return {"hits": self._hits, "misses": self._misses,
                "hit_rate": round(self._hits / t, 3) if t else 0.0}


class FinancialRAG:
    """
    Production RAG pipeline for SEC filing analysis.
    All responses include: answer + sources + confidence breakdown + guardrail status.
    """

    def __init__(self, vector_store, embedder, model="claude-sonnet-4-20250514",
                 top_k=5, use_cache=True, cache_dir="data/cache",
                 monitor_dir="data/monitoring", min_confidence=0.12,
                 strict_guardrails=False):
        self.vector_store = vector_store
        self.embedder = embedder
        self.model = model
        self.top_k = top_k
        self.cache = ResponseCache(cache_dir) if use_cache else None
        self.monitor = get_monitor(monitor_dir)
        self.scorer = ConfidenceScorer()
        self.guardrails = Guardrails(strict_input=strict_guardrails,
                                     min_confidence=min_confidence)
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

    def _context(self, results):
        return "\n\n---\n\n".join(
            f"[Source {i} | {r.company} | {r.section} | Score: {r.score:.3f}]\n{r.text}"
            for i, r in enumerate(results, 1)
        )

    def _claude(self, system, user, max_tokens=1200):
        r = self.client.messages.create(model=self.model, max_tokens=max_tokens,
                                         system=system, messages=[{"role":"user","content":user}])
        return r.content[0].text

    def _log(self, query, mode, latency_ms, cached, results, answer, conf, guard, doc_id):
        ev = self.monitor.make_event(
            query=query, mode=mode, latency_ms=latency_ms, cached=cached,
            retrieval_scores=[r.score for r in results], answer_length=len(answer),
            confidence_score=conf.overall_confidence if conf else 0.0,
            groundedness_score=conf.groundedness_score if conf else 0.0,
            guardrail_triggered=guard.triggered, guardrail_type=guard.guardrail_type,
            doc_id=doc_id, model=self.model,
        )
        self.monitor.log(ev)

    def qa(self, query: str, doc_id: Optional[str] = None) -> dict:
        t0 = time.time()

        # Input guardrail
        ig = self.guardrails.check_input(query)
        if not ig.passed:
            latency_ms = (time.time() - t0) * 1000
            from src.evaluation.confidence import ConfidenceResult
            dummy_conf = ConfidenceResult(0.0, 0.0, 0.0, 0.0, 0.0, verdict="Blocked", explanation="Input guardrail triggered")
            self._log(query, "qa", latency_ms, False, [], ig.message, dummy_conf, ig, doc_id)
            return {"answer": ig.message, "sources": [], "confidence": None,
                    "guardrail": {"triggered": True, "type": ig.guardrail_type,
                                  "severity": ig.severity, "message": ig.message},
                    "cached": False, "mode": "qa", "latency_ms": round(latency_ms, 1)}
            

        # Cache
        if self.cache:
            c = self.cache.get(query, "qa", doc_id)
            if c:
                c["cached"] = True
                latency_ms = (time.time() - t0) * 1000
                from src.evaluation.confidence import ConfidenceResult
                dummy_conf = ConfidenceResult(0.0, 0.0, 0.0, 0.0, 0.0, verdict="Cached", explanation="Served from cache")
                dummy_guard = type('G', (), {'triggered': False, 'guardrail_type': None})()
                self._log(query, "qa", latency_ms, True, [], c.get("answer",""), dummy_conf, dummy_guard, doc_id)
                return c

        # Retrieve
        qv = self.embedder.embed_query(query)
        results = self.vector_store.search(qv, top_k=self.top_k, filter_doc_id=doc_id)
        if not results:
            return {"answer": "No relevant context found. Please ingest documents first.",
                    "sources": [], "confidence": None,
                    "guardrail": {"triggered": False}, "cached": False, "mode": "qa", "latency_ms": 0}

        ctx = self._context(results)
        answer = self._claude(QA_SYSTEM_PROMPT,
            f"CONTEXT FROM SEC FILINGS:\n{ctx}\n\nQUESTION: {query}\n\nAnswer strictly from context above.", 1000)

        # Score + output guardrail
        conf = self.scorer.score(query, answer, results, qv)
        og = self.guardrails.check_output(answer, ctx, conf.overall_confidence)
        if not og.passed:
            answer = og.message

        latency_ms = (time.time() - t0) * 1000
        result = {
            "answer": answer,
            "sources": [r.to_dict() for r in results],
            "confidence": conf.to_dict(),
            "guardrail": {"triggered": og.triggered, "type": og.guardrail_type,
                          "severity": og.severity, "message": og.message},
            "cached": False, "latency_ms": round(latency_ms, 1), "mode": "qa",
        }
        if self.cache: self.cache.set(query, "qa", result, doc_id)
        self._log(query, "qa", latency_ms, False, results, answer, conf,
                  og if og.triggered else ig, doc_id)
        return result

    def summarize(self, doc_id: str) -> dict:
        t0 = time.time()
        if self.cache:
            c = self.cache.get(doc_id, "summarize", doc_id)
            if c: c["cached"] = True; return c

        chunks = self.vector_store.get_document_chunks(doc_id, max_chunks=40)
        if not chunks:
            return {"summary": f"No chunks for '{doc_id}'.", "doc_id": doc_id,
                    "company": "Unknown", "confidence": None, "cached": False}

        step = max(1, len(chunks) // 22)
        sel = chunks[::step][:22]
        ctx = "\n\n---\n\n".join(f"[{c.get('section','General')}]\n{c['text']}" for c in sel)
        company = chunks[0].get("company", doc_id)
        summary = self._claude(SUMMARY_SYSTEM_PROMPT,
            f"SEC 10-K FILING — {company}:\n\n{ctx}\n\nGenerate executive summary.", 1400)

        latency_ms = (time.time() - t0) * 1000
        result = {"summary": summary, "doc_id": doc_id, "company": company,
                  "chunks_used": len(sel), "confidence": None,
                  "cached": False, "latency_ms": round(latency_ms, 1)}
        if self.cache: self.cache.set(doc_id, "summarize", result, doc_id)
        self.monitor.log(self.monitor.make_event(query=doc_id, mode="summarize", latency_ms=latency_ms, cached=False, retrieval_scores=[], answer_length=len(summary), confidence_score=0.0, groundedness_score=0.0, guardrail_triggered=False, guardrail_type=None, doc_id=doc_id, model=self.model))
        return result

    def extract(self, doc_id: str) -> dict:
        t0 = time.time()
        if self.cache:
            c = self.cache.get(doc_id, "extract", doc_id)
            if c: c["cached"] = True; return c

        queries = ["revenue net sales income growth", "gross margin EBITDA profitability",
                   "risk factors material risks", "cash debt balance sheet",
                   "earnings per share dividends capital"]
        seen, uniq = set(), []
        for q in queries:
            qv = self.embedder.embed_query(q)
            for r in self.vector_store.search(qv, top_k=3, filter_doc_id=doc_id):
                if r.chunk_id not in seen:
                    seen.add(r.chunk_id); uniq.append(r)

        if not uniq:
            return {"metrics": {}, "doc_id": doc_id, "raw_response": "No content.",
                    "cached": False, "confidence_score": 0.0}

        ctx = self._context(uniq[:10])
        raw = self._claude(EXTRACTION_SYSTEM_PROMPT,
            f"Extract financial metrics:\n\n{ctx}\n\nReturn ONLY JSON.", 900)
        try:
            metrics = json.loads(raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip())
        except json.JSONDecodeError:
            metrics = {"parse_error": True}

        avg_score = sum(r.score for r in uniq) / len(uniq)
        conf_score = round(max(0.0, min(1.0, (avg_score - 0.3) / 0.5)), 3)

        latency_ms = (time.time() - t0) * 1000
        result = {"metrics": metrics, "doc_id": doc_id, "raw_response": raw,
                  "sources": [r.to_dict() for r in uniq[:6]],
                  "confidence_score": conf_score, "cached": False,
                  "latency_ms": round(latency_ms, 1)}
        if self.cache: self.cache.set(doc_id, "extract", result, doc_id)
        self.monitor.log(self.monitor.make_event(query=doc_id, mode="extract", latency_ms=latency_ms, cached=False, retrieval_scores=[r.score for r in uniq], answer_length=len(raw), confidence_score=conf_score, groundedness_score=0.0, guardrail_triggered=False, guardrail_type=None, doc_id=doc_id, model=self.model))
        return result

    def compare(self, doc_id_a: str, doc_id_b: str, aspect: str) -> dict:
        t0 = time.time()
        ck = f"compare::{doc_id_a}::{doc_id_b}::{aspect}"
        if self.cache:
            c = self.cache.get(ck, "compare")
            if c: c["cached"] = True; return c

        qv = self.embedder.embed_query(aspect)
        res_a = self.vector_store.search(qv, top_k=4, filter_doc_id=doc_id_a)
        res_b = self.vector_store.search(qv, top_k=4, filter_doc_id=doc_id_b)
        all_r = res_a + res_b

        ctx_a = "\n\n".join(f"[{r.section}] {r.text}" for r in res_a)
        ctx_b = "\n\n".join(f"[{r.section}] {r.text}" for r in res_b)

        answer = self._claude(QA_SYSTEM_PROMPT,
            f"Compare on: **{aspect}**\n\nDOC A ({doc_id_a}):\n{ctx_a}\n\nDOC B ({doc_id_b}):\n{ctx_b}\n\n"
            "Provide: 1) Key Similarities 2) Key Differences (with figures) 3) Which appears stronger", 1200)

        conf = self.scorer.score(aspect, answer, all_r)
        latency_ms = (time.time() - t0) * 1000
        result = {"answer": answer, "sources_a": [r.to_dict() for r in res_a],
                  "sources_b": [r.to_dict() for r in res_b], "confidence": conf.to_dict(),
                  "cached": False, "latency_ms": round(latency_ms, 1), "mode": "compare"}
        if self.cache: self.cache.set(ck, "compare", result)
        self.monitor.log(self.monitor.make_event(query=aspect, mode="compare", latency_ms=latency_ms, cached=False, retrieval_scores=[r.score for r in all_r], answer_length=len(answer), confidence_score=conf.overall_confidence, groundedness_score=conf.groundedness_score, guardrail_triggered=False, guardrail_type=None, doc_id=doc_id_a, model=self.model))
        return result
