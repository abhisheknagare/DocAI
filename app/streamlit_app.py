"""
DocAI — Intelligent Document Analysis
Run: streamlit run app/streamlit_app.py
"""
import sys, os, json, time, tempfile
import numpy as np
import streamlit as st
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="DocAI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@300;400;500;600&display=swap');

html, body, .stApp { background: #080C10; color: #CBD5E1; }

/* Sidebar */
.stSidebar { background: #0D1117 !important; border-right: 1px solid #1E2937; }

/* Typography */
h1 { font-family:'Syne',sans-serif !important; font-weight:800 !important;
     font-size:2.2rem !important; color:#F1F5F9; letter-spacing:-1px; }
h2 { font-family:'Syne',sans-serif !important; font-weight:700 !important;
     color:#E2E8F0; font-size:1.25rem !important; }
h3 { font-family:'Syne',sans-serif !important; color:#94A3B8; font-size:0.95rem !important; }

/* Metric cards */
.kpi { background:#0D1117; border:1px solid #1E2937; border-radius:10px;
       padding:1rem 1.25rem; }
.kpi-val { font-family:'IBM Plex Mono',monospace; font-size:1.5rem;
           font-weight:600; color:#38BDF8; }
.kpi-lbl { font-size:0.65rem; text-transform:uppercase; letter-spacing:0.1em;
           color:#475569; font-family:'IBM Plex Mono',monospace; }

/* Answer box */
.answer { background:#0D1117; border:1px solid #1E2937; border-left:3px solid #38BDF8;
          border-radius:8px; padding:1.25rem 1.5rem; font-family:'Inter',sans-serif;
          font-size:0.9rem; line-height:1.75; color:#CBD5E1; margin:1rem 0; }

/* Confidence bar */
.conf-bar-outer { background:#1E2937; border-radius:4px; height:6px; margin:4px 0; }
.conf-bar-inner { border-radius:4px; height:6px; }

/* Source chip */
.chip { display:inline-block; background:#0D1117; border:1px solid #1E2937;
        border-radius:16px; padding:2px 10px; font-size:0.68rem;
        font-family:'IBM Plex Mono',monospace; color:#64748B; margin:2px 3px 2px 0; }
.chip-score { color:#34D399; }

/* Risk tag */
.rtag { display:inline-block; background:#FF444422; border:1px solid #FF4444;
        border-radius:4px; padding:2px 8px; font-size:0.7rem; color:#FF7070;
        margin:2px 3px; }

/* Guardrail warning */
.guard-warn { background:#FBBF2411; border:1px solid #FBBF24; border-radius:8px;
              padding:0.75rem 1rem; font-size:0.82rem; color:#FDE68A; margin:0.75rem 0; }
.guard-block { background:#EF444411; border:1px solid #EF4444; border-radius:8px;
               padding:0.75rem 1rem; font-size:0.82rem; color:#FCA5A5; margin:0.75rem 0; }

/* Confidence badge */
.badge-high { background:#34D39922; border:1px solid #34D399; color:#34D399;
              border-radius:4px; padding:2px 8px; font-size:0.68rem;
              font-family:'IBM Plex Mono',monospace; }
.badge-med  { background:#FBBF2422; border:1px solid #FBBF24; color:#FBBF24;
              border-radius:4px; padding:2px 8px; font-size:0.68rem;
              font-family:'IBM Plex Mono',monospace; }
.badge-low  { background:#F9731622; border:1px solid #F97316; color:#F97316;
              border-radius:4px; padding:2px 8px; font-size:0.68rem;
              font-family:'IBM Plex Mono',monospace; }
.badge-none { background:#EF444422; border:1px solid #EF4444; color:#EF4444;
              border-radius:4px; padding:2px 8px; font-size:0.68rem;
              font-family:'IBM Plex Mono',monospace; }

/* Cached */
.cached { background:#818CF811; border:1px solid #818CF8; color:#818CF8;
          border-radius:4px; padding:1px 7px; font-size:0.65rem;
          font-family:'IBM Plex Mono',monospace; vertical-align:middle; }

/* Monitoring table */
.mon-row { display:flex; gap:0.5rem; padding:0.4rem 0;
           border-bottom:1px solid #1E2937; font-size:0.78rem; }
.mon-q { flex:3; color:#94A3B8; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.mon-v { flex:1; text-align:right; font-family:'IBM Plex Mono',monospace;
         font-size:0.72rem; color:#38BDF8; }

/* Divider */
hr { border-color:#1E2937 !important; }
#MainMenu, footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


#Helpers

def conf_badge(score: float) -> str:
    if score >= 0.75: return f'<span class="badge-high">✅ {score:.0%}</span>'
    if score >= 0.52: return f'<span class="badge-med">⚠️ {score:.0%}</span>'
    if score >= 0.30: return f'<span class="badge-low">🔶 {score:.0%}</span>'
    return f'<span class="badge-none">❌ {score:.0%}</span>'

def conf_bar(label: str, value: float, color: str = "#38BDF8"):
    pct = int(value * 100)
    return f"""
    <div style="margin:4px 0">
      <div style="display:flex;justify-content:space-between;font-size:0.68rem;
                  font-family:'IBM Plex Mono',monospace;color:#475569">
        <span>{label}</span><span style="color:{color}">{pct}%</span>
      </div>
      <div class="conf-bar-outer">
        <div class="conf-bar-inner" style="width:{pct}%;background:{color}"></div>
      </div>
    </div>"""


def render_confidence(conf: dict):
    """Render full confidence breakdown panel."""
    if not conf:
        return
    overall = conf["overall_confidence"]
    st.markdown(f"""
    <div style="background:#0D1117;border:1px solid #1E2937;border-radius:8px;
                padding:1rem 1.25rem;margin:0.75rem 0">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.75rem">
        <span style="font-family:'Syne',sans-serif;font-size:0.85rem;font-weight:600;color:#E2E8F0">
          Confidence Analysis
        </span>
        {conf_badge(overall)}
      </div>
      {conf_bar("Overall Confidence", overall, "#38BDF8")}
      {conf_bar("Retrieval Relevance", conf["retrieval_confidence"], "#818CF8")}
      {conf_bar("Groundedness", conf["groundedness_score"], "#34D399")}
      {conf_bar("Coverage", conf["coverage_score"], "#FBBF24")}
      {conf_bar("Chunk Consistency", conf["consistency_score"], "#F472B6")}
      <div style="margin-top:0.6rem;font-size:0.72rem;color:#64748B;
                  font-family:'Inter',sans-serif;line-height:1.5">
        {conf.get("verdict","")} — {conf.get("explanation","")}
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_sources(sources: list, label: str = "Sources"):
    if not sources:
        return
    chips = "".join(
        f'<span class="chip">{s["company"]} · {s["section"]}'
        f'<span class="chip-score"> ↑{s["score"]:.2f}</span></span>'
        for s in sources
    )
    st.markdown(f"**{label}:** {chips}", unsafe_allow_html=True)

    with st.expander(f"📄 {len(sources)} retrieved chunks"):
        for i, s in enumerate(sources, 1):
            contributed = ""
            chunk_scores = []
            
            st.markdown(f"""
            <div style="border:1px solid #1E2937;border-radius:8px;padding:0.85rem;margin:0.4rem 0">
              <div style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;color:#475569;
                          margin-bottom:0.5rem;display:flex;gap:1rem;flex-wrap:wrap">
                <span>[{i}]</span>
                <span style="color:#38BDF8">{s["company"]}</span>
                <span>{s["section"]}</span>
                <span>Score: <span style="color:#34D399">{s["score"]:.4f}</span></span>
                <span>Chunk {s.get("chunk_index",0)+1}/{s.get("total_chunks","?")}</span>
                <span style="color:#64748B">{s.get("doc_id","")}</span>
              </div>
              <div style="font-size:0.82rem;line-height:1.65;color:#94A3B8">{s["text"][:600]}…</div>
            </div>
            """, unsafe_allow_html=True)


def render_guardrail(g: dict):
    if not g or not g.get("triggered"):
        return
    css = "guard-warn" if g.get("severity") == "warn" else "guard-block"
    icon = "⚠️" if g.get("severity") == "warn" else "🚫"
    st.markdown(f'<div class="{css}">{icon} <b>Guardrail [{g.get("type","")}]:</b> {g.get("message","")}</div>',
                unsafe_allow_html=True)


#State 

def init():
    for k, v in {"store": None, "embedder": None, "rag": None,
                 "index_loaded": False, "qa_history": [],
                 "api_key": os.environ.get("ANTHROPIC_API_KEY", "")}.items():
        if k not in st.session_state:
            st.session_state[k] = v

init()


#Loaders

@st.cache_resource(show_spinner="Loading index…")
def load_index(index_dir: str):
    from src.embeddings.embedder import get_embedder
    from src.vectorstore.faiss_store import FAISSVectorStore
    cfg_path = Path(index_dir) / "config.json"
    dim, emb_model = 256, "minilm"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        dim = cfg.get("embedding_dim", 256)
        emb_model = cfg.get("embedding_model", "minilm")
    embedder = get_embedder(emb_model)
    store = FAISSVectorStore(dim=dim, index_dir=index_dir)
    ok = store.load()
    return store, embedder, ok


def get_rag():
    if not st.session_state.store or not st.session_state.embedder:
        return None
    key = st.session_state.get("api_key", "")
    if key:
        os.environ["ANTHROPIC_API_KEY"] = key
    from src.rag.pipeline import FinancialRAG
    return FinancialRAG(
        vector_store=st.session_state.store,
        embedder=st.session_state.embedder,
        top_k=st.session_state.get("top_k", 5),
        min_confidence=st.session_state.get("min_conf", 0.25),
        strict_guardrails=st.session_state.get("strict_guard", False),
    )


def ingest_uploads(files, index_dir="data/index"):
    from src.ingestion.extractor import extract_file
    from src.ingestion.chunker import chunk_text
    from src.embeddings.embedder import get_embedder
    from src.vectorstore.faiss_store import FAISSVectorStore
    chunks = []
    with tempfile.TemporaryDirectory() as tmp:
        for uf in files:
            dest = Path(tmp) / uf.name
            dest.write_bytes(uf.read())
            text = extract_file(str(dest))
            if text:
                chunks.extend(chunk_text(text, Path(uf.name).stem, uf.name))
    if not chunks:
        return None, None, "No text extracted."
    embedder = get_embedder("minilm")
    texts = [c.text for c in chunks]
    embedder.fit(texts)
    vecs = embedder.embed(texts, show_progress=False).astype(np.float32)
    store = FAISSVectorStore(dim=embedder.dim, index_dir=index_dir)
    store.build(vecs, [c.to_dict() for c in chunks])
    store.save()
    Path(index_dir, "config.json").write_text(json.dumps({
        "embedding_model": "minilm", "embedding_dim": embedder.dim,
        "index_dir": index_dir, "documents": store.documents, "total_chunks": len(chunks),
    }))
    return store, embedder, None


#Sidebar 

def sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding:0.5rem 0 1rem">
          <div style="font-family:'Syne',sans-serif;font-size:1.15rem;
                      font-weight:800;color:#F1F5F9;letter-spacing:-0.5px">DocAI</div>
          <div style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;
                      color:#334155;letter-spacing:0.05em">SEC FILING INTELLIGENCE</div>
        </div>""", unsafe_allow_html=True)

        st.divider()

        key = st.text_input("API Key", type="password", 
                     value=os.environ.get("ANTHROPIC_API_KEY",""),
                     placeholder="sk-ant-...", label_visibility="collapsed")
        if key:
            st.session_state.api_key = key
            os.environ["ANTHROPIC_API_KEY"] = key

        # Source
        st.markdown("##### 📂 Document Source")
        src = st.radio("Document Source", ["Pre-indexed", "Upload PDFs"], label_visibility="collapsed")

        if src == "Pre-indexed":
            idx_dir = st.text_input("Index dir", value="data/index", label_visibility="collapsed")
            if st.button("⟳ Load Index", use_container_width=True):
                with st.spinner("Loading…"):
                    store, emb, ok = load_index(idx_dir)
                    if ok:
                        st.session_state.store = store
                        st.session_state.embedder = emb
                        st.session_state.index_loaded = True
                        st.success(f"✅ {store.num_vectors:,} vectors")
                    else:
                        st.error("Index not found. Run `python scripts/ingest.py`")
        else:
            ups = st.file_uploader("Drop PDFs", type=["pdf"], accept_multiple_files=True)
            if ups and st.button("⚡ Index Now", use_container_width=True):
                with st.spinner(f"Indexing {len(ups)} files…"):
                    store, emb, err = ingest_uploads(ups)
                    if err:
                        st.error(err)
                    else:
                        st.session_state.store = store
                        st.session_state.embedder = emb
                        st.session_state.index_loaded = True
                        st.success(f"✅ {store.num_vectors:,} chunks indexed")

        st.divider()

        #Settings
        st.markdown("##### ⚙️ Settings")
        st.session_state.top_k = st.slider("Top-K chunks", 3, 10, 5)
        st.session_state.min_conf = st.slider("Min confidence threshold", 0.0, 0.5, 0.25, 0.05)
        st.session_state.strict_guard = st.checkbox("Strict guardrails", False)

        #Doc filter
        if st.session_state.index_loaded and st.session_state.store:
            st.divider()
            st.markdown("##### 🗂 Scope")
            docs = ["All documents"] + st.session_state.store.documents
            sel = st.selectbox("Filter to document", docs, label_visibility="collapsed")
            st.session_state.active_doc = None if sel == "All documents" else sel

            #Quick stats
            stats = st.session_state.store.stats()
            st.markdown(f"""
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;
                        color:#334155;line-height:2;margin-top:0.5rem">
              VECTORS <span style="color:#38BDF8">{stats['total_vectors']:,}</span><br>
              DOCS    <span style="color:#38BDF8">{stats['documents']}</span><br>
              DIM     <span style="color:#38BDF8">{stats['dim']}</span>
            </div>""", unsafe_allow_html=True)


#Tab: Q&A 

SAMPLE_QS = [
    "What was total revenue and YoY growth?",
    "What are the top risk factors?",
    "What is the gross margin percentage?",
    "Describe the main business segments",
    "What is the CEO compensation?",
    "How much cash does the company hold?",
]

def tab_qa():
    st.markdown("## 💬 Document Q&A")

    #Sample buttons
    cols = st.columns(3)
    for i, q in enumerate(SAMPLE_QS):
        if cols[i % 3].button(q, key=f"sq_{i}", use_container_width=True):
            st.session_state["_preset_q"] = q

    query = st.text_area("Your question", height=80, label_visibility="collapsed",
                         placeholder="Ask anything about the SEC filings…",
                         value=st.session_state.get("_preset_q", ""))

    c1, c2 = st.columns([1, 6])
    submit = c1.button("🔍 Ask", use_container_width=True)
    if c2.button("🗑 Clear", use_container_width=False):
        st.session_state.qa_history = []
        st.rerun()

    if submit and query.strip():
        rag = get_rag()
        if not rag:
            st.error("Configure your Anthropic API key in the sidebar.")
            return
        doc_id = st.session_state.get("active_doc")
        with st.spinner("Retrieving & generating…"):
            result = rag.qa(query, doc_id=doc_id)
        st.session_state.qa_history.append({"q": query, "r": result})

    for item in reversed(st.session_state.qa_history):
        q, r = item["q"], item["r"]
        cached = r.get("cached", False)
        lat = r.get("latency_ms", 0)

        st.markdown(f"""
        <div style="margin-top:1.5rem">
          <div style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;
                      color:#475569;margin-bottom:0.4rem;display:flex;
                      align-items:center;gap:8px">
            <span style="color:#64748B">❓</span>
            <span style="color:#94A3B8">{q}</span>
            {'<span class="cached">⚡ cached</span>' if cached else
             f'<span style="font-size:0.62rem;color:#334155">{lat:.0f}ms</span>'}
          </div>
        </div>""", unsafe_allow_html=True)

        #Guardrail first
        render_guardrail(r.get("guardrail", {}))

        #Answer
        if r.get("answer"):
            st.markdown(f'<div class="answer">{r["answer"]}</div>', unsafe_allow_html=True)

        #Confidence
        if r.get("confidence"):
            render_confidence(r["confidence"])

        #Sources
        render_sources(r.get("sources", []))
        st.divider()


#Tab: Summarize 

def tab_summarize():
    st.markdown("## 📝 Executive Summarization")
    if not st.session_state.index_loaded:
        st.info("Load an index first.")
        return

    docs = st.session_state.store.documents
    doc = st.selectbox("Select document", docs)

    if st.button("📋 Generate Summary"):
        rag = get_rag()
        if not rag: st.error("Configure API key."); return
        with st.spinner(f"Summarizing {doc}…"):
            result = rag.summarize(doc)

        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:0.75rem">
          <span style="font-family:'Syne',sans-serif;font-size:1rem;
                       font-weight:700;color:#E2E8F0">{result.get('company',doc)}</span>
          <span style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;
                       color:#334155">{result.get('chunks_used',0)} chunks · {result.get('latency_ms',0):.0f}ms</span>
          {'<span class="cached">⚡ cached</span>' if result.get("cached") else ""}
        </div>
        <div class="answer">{result["summary"]}</div>
        """, unsafe_allow_html=True)


#Tab: Extract 

def tab_extract():
    st.markdown("## 🔬 Financial Metrics Extraction")
    if not st.session_state.index_loaded:
        st.info("Load an index first.")
        return

    docs = st.session_state.store.documents
    doc = st.selectbox("Select document", docs, key="exdoc")

    if st.button("⚡ Extract Metrics"):
        rag = get_rag()
        if not rag: st.error("Configure API key."); return
        with st.spinner(f"Extracting from {doc}…"):
            result = rag.extract(doc)

        m = result.get("metrics", {})
        cached = result.get("cached", False)
        conf_s = result.get("confidence_score", 0.0)

        if not m or m.get("parse_error"):
            st.warning("Could not parse structured metrics. Raw response below:")
            st.code(result.get("raw_response", ""), language="text")
            return

        #Header row
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:1rem">
          <span style="font-family:'Syne',sans-serif;font-size:1.05rem;
                       font-weight:700;color:#E2E8F0">
            {m.get('company', doc)} — FY{m.get('fiscal_year','?')}
          </span>
          {conf_badge(conf_s)}
          {'<span class="cached">⚡ cached</span>' if cached else ""}
          <span style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;
                       color:#334155">data_confidence: {m.get('data_confidence','?')}</span>
        </div>""", unsafe_allow_html=True)

        #KPI grid
        fields = [
            ("revenue", "💰 Revenue"),
            ("net_income", "📈 Net Income"),
            ("gross_margin", "📊 Gross Margin"),
            ("operating_income", "🏦 Op. Income"),
            ("cash_and_equivalents", "💵 Cash"),
            ("total_debt", "📉 Debt"),
            ("eps_diluted", "📌 EPS (Diluted)"),
            ("capex", "🏗️ CapEx"),
            ("r_and_d", "🔬 R&D"),
        ]
        cols = st.columns(3)
        shown = 0
        for field, label in fields:
            d = m.get(field, {})
            if d and d.get("value") is not None:
                val, unit = d["value"], d.get("unit", "")
                growth = d.get("growth_pct")
                g_str = f" · YoY {growth:+.1f}%" if growth is not None else ""
                with cols[shown % 3]:
                    st.markdown(f"""
                    <div class="kpi">
                      <div class="kpi-lbl">{label}</div>
                      <div class="kpi-val">{val}</div>
                      <div style="font-size:0.68rem;color:#334155;
                                  font-family:'IBM Plex Mono',monospace">{unit}{g_str}</div>
                    </div><br>""", unsafe_allow_html=True)
                shown += 1

        #Risks
        risks = m.get("key_risks", [])
        if risks:
            st.markdown("**⚠️ Key Risks**")
            st.markdown("".join(f'<span class="rtag">{r}</span>' for r in risks),
                        unsafe_allow_html=True)

        #Segments + Competitors
        col1, col2 = st.columns(2)
        with col1:
            segs = m.get("key_segments", [])
            if segs:
                st.markdown("**🗂 Segments**")
                for s in segs: st.markdown(f"- `{s}`")
        with col2:
            comps = m.get("mentioned_competitors", [])
            if comps:
                st.markdown("**⚔️ Competitors Mentioned**")
                for c in comps: st.markdown(f"- `{c}`")

        #Sources
        render_sources(result.get("sources", []), "Evidence Chunks")

        with st.expander("🔍 Raw JSON"):
            st.json(m)


#Tab: Compare 

COMPARE_ASPECTS = [
    "Revenue and profitability",
    "Risk factors and business risks",
    "Business strategy and competitive position",
    "R&D investment and innovation",
    "Capital allocation and shareholder returns",
    "Gross margin and cost structure",
    "Market position and competitive moat",
]

def tab_compare():
    st.markdown("## ⚖️ Cross-Document Comparison")
    if not st.session_state.index_loaded:
        st.info("Load an index first.")
        return
    docs = st.session_state.store.documents
    if len(docs) < 2:
        st.info("Need at least 2 documents. Currently indexed: " + ", ".join(docs))
        return

    c1, c2 = st.columns(2)
    doc_a = c1.selectbox("Document A", docs, key="ca")
    doc_b = c2.selectbox("Document B", [d for d in docs if d != doc_a], key="cb")
    aspect = st.selectbox("Compare on", COMPARE_ASPECTS)

    if st.button("⚖️ Compare"):
        rag = get_rag()
        if not rag: st.error("Configure API key."); return
        with st.spinner("Running cross-document analysis…"):
            result = rag.compare(doc_a, doc_b, aspect)

        cached = result.get("cached", False)
        st.markdown(f"""
        <div style="margin-bottom:0.5rem;font-family:'IBM Plex Mono',monospace;
                    font-size:0.65rem;color:#334155">
          {doc_a} vs {doc_b} · {aspect} · {result.get('latency_ms',0):.0f}ms
          {'<span class="cached">⚡ cached</span>' if cached else ""}
        </div>
        <div class="answer">{result["answer"]}</div>
        """, unsafe_allow_html=True)

        if result.get("confidence"):
            render_confidence(result["confidence"])

        col1, col2 = st.columns(2)
        with col1:
            render_sources(result.get("sources_a", []), f"Sources from {doc_a}")
        with col2:
            render_sources(result.get("sources_b", []), f"Sources from {doc_b}")


#Tab: Monitoring 

def tab_monitor():
    st.markdown("## 📈 System Monitoring")

    from src.monitoring.monitor import get_monitor
    mon = get_monitor()
    hist = mon.historical_metrics(hours=24)
    session = mon.session_metrics()
    events = mon.load_history(last_n=50)

    #KPI row
    kpis = [
        (f"{session['total_queries']}", "Queries (Session)"),
        (f"{hist.get('avg_latency_ms', 0):.0f}ms", "Avg Latency"),
        (f"{session['cache_hit_rate']:.0%}", "Cache Hit Rate"),
        (f"{session['avg_confidence']:.0%}", "Avg Confidence"),
        (f"{session['avg_groundedness']:.0%}", "Avg Groundedness"),
        (f"{session['guardrail_rate']:.0%}", "Guardrail Rate"),
    ]
    cols = st.columns(3)
    for i, (val, lbl) in enumerate(kpis):
        cols[i % 3].markdown(f"""
        <div class="kpi">
          <div class="kpi-lbl">{lbl}</div>
          <div class="kpi-val">{val}</div>
        </div><br>""", unsafe_allow_html=True)

    st.divider()
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("**Recent Queries**")
        st.markdown("""
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;
                    color:#334155;margin-bottom:0.5rem;display:flex;gap:0.5rem">
          <span style="flex:3">QUERY</span>
          <span style="flex:1;text-align:right">CONF</span>
          <span style="flex:1;text-align:right">GROUND</span>
          <span style="flex:1;text-align:right">LAT</span>
          <span style="flex:1;text-align:right">CACHE</span>
        </div>""", unsafe_allow_html=True)

        for ev in reversed(events[-15:]):
            q_short = ev.get("query","")[:45] + ("…" if len(ev.get("query","")) > 45 else "")
            conf_c = "#34D399" if ev.get("confidence_score",0) >= 0.7 else \
                     "#FBBF24" if ev.get("confidence_score",0) >= 0.45 else "#EF4444"
            st.markdown(f"""
            <div class="mon-row">
              <span class="mon-q">{q_short}</span>
              <span class="mon-v" style="color:{conf_c}">{ev.get('confidence_score',0):.0%}</span>
              <span class="mon-v">{ev.get('groundedness_score',0):.0%}</span>
              <span class="mon-v">{ev.get('latency_ms',0):.0f}ms</span>
              <span class="mon-v" style="color:#818CF8">{'⚡' if ev.get('cached') else '—'}</span>
            </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("**Guardrail Events**")
        guard_events = [e for e in events if e.get("guardrail_triggered")]
        if guard_events:
            for ev in reversed(guard_events[-8:]):
                st.markdown(f"""
                <div class="guard-warn">
                  <b>[{ev.get('guardrail_type','?')}]</b><br>
                  <span style="font-size:0.72rem">{ev.get('query','')[:60]}…</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:#334155;font-size:0.8rem">No guardrail triggers yet.</div>',
                        unsafe_allow_html=True)

        st.divider()
        st.markdown("**Mode Distribution**")
        mode_d = session.get("mode_distribution", {})
        total_q = max(sum(mode_d.values()), 1)
        for mode, count in sorted(mode_d.items(), key=lambda x: -x[1]):
            pct = count / total_q
            st.markdown(f"""
            <div style="margin:4px 0">
              <div style="display:flex;justify-content:space-between;
                          font-size:0.68rem;font-family:'IBM Plex Mono',monospace;color:#475569">
                <span>{mode}</span><span style="color:#38BDF8">{count} ({pct:.0%})</span>
              </div>
              <div class="conf-bar-outer">
                <div class="conf-bar-inner" style="width:{int(pct*100)}%;background:#38BDF8"></div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.divider()
        st.markdown("**Top Queries**")
        top = mon.top_queries(6)
        for tq in top:
            q_short = tq["query"][:40] + ("…" if len(tq["query"]) > 40 else "")
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;margin:3px 0;
                        font-size:0.72rem;font-family:'IBM Plex Mono',monospace">
              <span style="color:#64748B">{q_short}</span>
              <span style="color:#38BDF8">{tq['count']}×</span>
            </div>""", unsafe_allow_html=True)

    st.divider()
    with st.expander("📋 Raw Event Log (last 20)"):
        for ev in reversed(events[-20:]):
            st.json(ev)


#Main

def main():
    sidebar()

    #Auto-load
    if not st.session_state.index_loaded and Path("data/index/faiss.index").exists():
        store, emb, ok = load_index("data/index")
        if ok:
            st.session_state.store = store
            st.session_state.embedder = emb
            st.session_state.index_loaded = True

    #Header
    index_status = ""
    if st.session_state.index_loaded and st.session_state.store:
        stats = st.session_state.store.stats()
        index_status = f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:0.65rem;color:#334155;margin-left:12px">{stats["total_vectors"]:,} vectors · {stats["documents"]} docs</span>'

    st.markdown(f'<h1>DocAI{index_status}</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#334155;font-size:0.85rem;margin-top:-0.5rem;'
                'font-family:\'IBM Plex Mono\',monospace">SEC Filing Intelligence · '
                'Q&A · Summarize · Extract · Compare · Monitor</p>', unsafe_allow_html=True)

    if not st.session_state.index_loaded:
        st.markdown("""
        <div style="background:#0D1117;border:1px solid #1E2937;border-radius:10px;
                    padding:1.5rem 2rem;margin:1rem 0">
          <div style="font-family:'Syne',sans-serif;font-size:1rem;
                      font-weight:700;color:#E2E8F0;margin-bottom:0.75rem">🚀 Quick Start</div>
          <div style="font-family:'IBM Plex Mono',monospace;font-size:0.78rem;
                      color:#475569;line-height:2">
            1. Set ANTHROPIC_API_KEY in sidebar<br>
            2a. Run ingestion: <span style="color:#38BDF8">python scripts/ingest.py</span>
                then click "Load Index"<br>
            2b. Or upload PDF filings directly from the sidebar<br>
            3. Ask questions in the Q&A tab
          </div>
        </div>""", unsafe_allow_html=True)

    if not (os.environ.get("ANTHROPIC_API_KEY") or st.session_state.api_key):
        st.warning("⚠️ Add your Anthropic API key in the sidebar to enable analysis.")

    tabs = st.tabs(["💬 Q&A", "📝 Summarize", "🔬 Extract", "⚖️ Compare", "📈 Monitor"])
    with tabs[0]: tab_qa()
    with tabs[1]: tab_summarize()
    with tabs[2]: tab_extract()
    with tabs[3]: tab_compare()
    with tabs[4]: tab_monitor()


if __name__ == "__main__":
    main()
