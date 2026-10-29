"""Streamlit mission-control interface for MoonBase RAG."""
from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

st.set_page_config(page_title="MoonBase RAG | Mission Control", page_icon="🌘", layout="wide")


def load_streamlit_secrets() -> None:
    """Copy Streamlit-managed secrets into environment variables used by rag.py."""
    try:
        secrets = st.secrets
        for key in ("OPENROUTER_API_KEY", "EMBEDDING_MODEL", "CHAT_MODEL", "EVAL_MODEL"):
            if key in secrets:
                os.environ[key] = str(secrets[key])
    except (FileNotFoundError, StreamlitSecretNotFoundError):
        # A missing secrets file is a valid unconfigured state; the UI will show setup guidance.
        return


load_streamlit_secrets()

import rag
import eval as eval_runner

st.markdown(
    """
    <style>
    :root { --navy:#101b2b; --panel:#17263a; --muted:#91a3b7; --ice:#e8f0f7; --cyan:#65d4dd; --amber:#f0b45e; --red:#f07878; }
    html, body, [class*="css"] { font-family: Arial, sans-serif; }
    .stApp { background: radial-gradient(ellipse at 78% 0%, #1a3141 0, #111c2c 38%, #0d1724 100%); color: var(--ice); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { background: #111d2d; border-right: 1px solid #26384c; }
    [data-testid="stSidebar"] * { color: var(--ice); }
    h1, h2, h3 { letter-spacing: -0.035em; color: #f3f7fb; }
    p, li, label { color: #c2cfdb; }
    .hero { padding: 1.35rem 1.6rem; border: 1px solid #2b4055; border-radius: 18px; background: linear-gradient(120deg,rgba(28,49,68,.96),rgba(20,35,53,.72)); margin: .25rem 0 1.25rem; }
    .eyebrow { color: var(--cyan); font: 500 .73rem Consolas,monospace; letter-spacing: .16em; text-transform: uppercase; }
    .hero h1 { margin: .35rem 0 .4rem; font-size: 2.2rem; }
    .hero p { margin: 0; color: #aebed0; max-width: 850px; }
    .pill { display:inline-block; padding:.24rem .58rem; border-radius:999px; font:500 .7rem 'DM Mono',monospace; color:#9be7e8; background:#153740; border:1px solid #2d6770; margin: .15rem .2rem .15rem 0; }
    .notice { border:1px solid #8c6838; border-left:4px solid var(--amber); background:#302a22; padding:.8rem 1rem; border-radius:10px; color:#f3d6a8; margin:.5rem 0 1rem; }
    .subtle { color:#91a3b7; font-size:.87rem; }
    .mono { font-family:Consolas,monospace; }
    [data-testid="stMetric"] { background:rgba(23,38,58,.82); border:1px solid #2b4055; border-radius:13px; padding:.8rem 1rem; }
    [data-testid="stMetricLabel"] { color:#9db0c3; }
    [data-testid="stMetricValue"] { color:#f4f8fb; }
    .stTabs [data-baseweb="tab-list"] { gap:.35rem; border-bottom:1px solid #293c50; }
    .stTabs [data-baseweb="tab"] { color:#9eb1c5; background:transparent; border-radius:8px 8px 0 0; }
    .stTabs [aria-selected="true"] { color:#72dce1 !important; border-bottom:2px solid #65d4dd; }
    [data-testid="stExpander"] { background:rgba(21,35,52,.65); border:1px solid #2a3d52; border-radius:11px; }
    .stButton>button, .stFormSubmitButton>button { border-radius:9px; border:1px solid #3b6975; background:#164552; color:#ecfbfc; font-weight:700; }
    .stButton>button:hover, .stFormSubmitButton>button:hover { border-color:#65d4dd; color:white; }
    [data-testid="stDataFrame"] { border:1px solid #2a3d52; border-radius:10px; overflow:hidden; }
    code { color:#9de8e7; }
    </style>
    """,
    unsafe_allow_html=True,
)

DATA = rag.DATA
PDF = DATA / "sample.pdf"
REPORT = eval_runner.REPORT
PERSONAS = {
    "Crew member": ("crew-member", ["all-staff"]),
    "Finance officer": ("finance-officer", ["all-staff", "finance"]),
    "Flight engineer": ("flight-engineer", ["all-staff", "engineers"]),
}


@st.cache_data(show_spinner=False, max_entries=2)
def load_index(pdf_modified_ns: int, provider: str, embedding_model: str) -> list[dict]:
    """Parse the sample PDF and cache chunks with hosted embeddings for this app process."""
    del pdf_modified_ns, provider, embedding_model  # These arguments form the cache key.
    return rag.load_or_ingest(PDF)


def get_index() -> list[dict]:
    """Load or create the embedded document index."""
    if not PDF.exists():
        raise FileNotFoundError("Sample PDF is missing. Run `python make_pdf.py` before starting the app.")
    return load_index(PDF.stat().st_mtime_ns, rag.MODEL_PROVIDER, rag.EMBEDDING_MODEL)


def read_jsonl(path: Path) -> list[dict]:
    """Read non-empty JSON Lines records, returning an empty list if absent."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def metric(label: str, value: object, help_text: str | None = None) -> None:
    """Keep metric-card rendering consistent across dashboard sections."""
    st.metric(label, value, help=help_text)


with st.sidebar:
    st.markdown("<div class='eyebrow'>LUNAR OPERATIONS / 01</div>", unsafe_allow_html=True)
    st.markdown("## MoonBase RAG")
    st.caption("Role-aware knowledge control")
    st.divider()
    selected_role = st.selectbox("Viewing as", list(PERSONAS))
    st.markdown("<div class='notice'><b>Demo access mode</b><br>Role selection is not authentication. This app uses synthetic documents only.</div>", unsafe_allow_html=True)
    has_key = bool(os.getenv("OPENROUTER_API_KEY"))
    st.markdown(f"**Model provider** · `{rag.MODEL_PROVIDER}`")
    st.markdown(f"**Embedding model** · `{rag.EMBEDDING_MODEL}`")
    st.markdown(f"**Answer model** · `{rag.CHAT_MODEL}`")
    st.markdown("**API connection** · " + ("configured" if has_key else "key required"))
    if st.button("Re-index sample documents", use_container_width=True):
        load_index.clear()
        st.rerun()
    st.divider()
    st.caption("Hosted embeddings + generation · Hybrid retrieval · ACL-aware context")

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">MISSION CONTROL &nbsp; / &nbsp; DOCUMENT INTELLIGENCE</div>
      <h1>Knowledge, with the right clearance.</h1>
      <p>Search MoonBase operating documents with role-filtered evidence, grounded answers, an evaluation suite, and an auditable trail.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("<span class='pill'>PDF · TEXT</span><span class='pill'>TABLES</span><span class='pill'>FIGURE CAPTIONS / OCR</span><span class='pill'>HYBRID SEARCH</span><span class='pill'>EVALUATED</span>", unsafe_allow_html=True)

tabs = st.tabs(["Ask MoonBase", "Evaluation", "Security & audit", "System details"])

with tabs[0]:
    left, right = st.columns([1.55, 1], gap="large")
    with left:
        st.subheader("Ask a question")
        st.caption(f"Active clearance: **{selected_role}** · permitted groups: `{', '.join(PERSONAS[selected_role][1])}`")
        if not has_key:
            st.warning("Add `OPENROUTER_API_KEY` in Streamlit secrets to enable hosted embeddings and answer generation.")
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        for entry in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(entry["question"])
            with st.chat_message("assistant"):
                st.markdown(entry["answer"])
                if entry["hits"]:
                    with st.expander(f"Retrieved evidence · {len(entry['hits'])} chunks"):
                        for hit in entry["hits"]:
                            st.markdown(f"**{hit['source']} · page {hit['page']} · {hit['kind']}**")
                            st.write(hit["text"])
                            st.caption(f"Hybrid {hit['hybrid_score']:.4f} · rerank {hit['rerank_score']:.4f} · groups {', '.join(hit['groups'])}")
        with st.form("question_form", clear_on_submit=True):
            question = st.text_input("Question", placeholder="What should I do if an oxygen alarm sounds?")
            submitted = st.form_submit_button("Search authorized knowledge", use_container_width=True)
        if submitted and question.strip():
            try:
                user, groups = PERSONAS[selected_role]
                with st.spinner("Embedding the query, filtering by role, and preparing a grounded answer…"):
                    chunks = get_index()
                    hits = rag.retrieve(question.strip(), chunks, user, groups)
                    answer = rag.generate_answer(question.strip(), hits)
                    rag.audit(question.strip(), user, groups, hits, answer)
                st.session_state.chat_history.append({"question": question.strip(), "answer": answer, "hits": hits})
                st.rerun()
            except Exception as exc:
                if type(exc).__name__ == "RateLimitError":
                    st.error("The model provider rate limit or free-tier quota was reached. Wait for quota to reset, then retry.")
                else:
                    st.error(f"The request could not be completed ({type(exc).__name__}). Check the API key, model access, and sample PDF.")
                with st.expander("Provider response details"):
                    st.code(str(exc)[:2000])
    with right:
        st.subheader("Clearance map")
        st.markdown("| Document | Crew | Finance | Engineer |\n|---|:---:|:---:|:---:|\n| Staff handbook | ✓ | ✓ | ✓ |\n| Compensation schedule | — | ✓ | — |\n| Oxygen runbook & diagram | — | — | ✓ |")
        st.markdown("**Security boundary**")
        st.write("`retrieve()` filters allowed groups before ranking and before any passage reaches the answer model. The role picker is only a walkthrough; a deployed public app needs trusted authentication and server-verified group claims.")
        st.markdown("**Try these**")
        st.write("• What should I do if an oxygen alarm sounds?\n\n• What is the flight engineer salary?\n\n• What does the airlock diagram show?")
        if (DATA / "airlock-diagram.png").exists():
            st.image(DATA / "airlock-diagram.png", caption="Airlock A · synthetic training illustration", use_container_width=True)

with tabs[1]:
    st.subheader("RAG quality & regression checks")
    st.caption("Run the labeled suite against the hosted retrieval and answer path. Every case is also inspectable below.")
    run_col, info_col = st.columns([1, 2])
    with run_col:
        run_eval = st.button("Run full evaluation", type="primary", use_container_width=True)
    with info_col:
        st.caption(f"{eval_runner.CASES.name} · {len(json.loads(eval_runner.CASES.read_text(encoding='utf-8')))} role-aware cases · embedding + answer + judge API usage")
    if run_eval:
        try:
            with st.spinner("Loading the cached document vectors and running the evaluation suite…"):
                chunks = get_index()
                report = eval_runner.run_evaluation(chunks=chunks)
            st.session_state.eval_report = report
            st.success(f"Evaluation complete: {report['passed']} passed, {report['failed']} failed.")
        except Exception as exc:
            if type(exc).__name__ == "RateLimitError":
                st.error("The model provider rate limit or free-tier quota was reached. Wait for quota to reset, then retry.")
            else:
                st.error(f"Evaluation could not complete ({type(exc).__name__}). Check your API key, model access, and sample PDF.")
            with st.expander("Provider response details"):
                st.code(str(exc)[:2000])
    report = st.session_state.get("eval_report")
    report_is_example = False
    if report is None and REPORT.exists():
        try:
            report = json.loads(REPORT.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            report = None
    if report is None:
        example_path = DATA / "eval-report-example.json"
        if example_path.exists():
            report = json.loads(example_path.read_text(encoding="utf-8"))
            report_is_example = True
    if report:
        if report_is_example or report.get("example_only"):
            st.info("Illustrative sample metrics are shown below. Run the suite to replace them with live results.")
        else:
            st.caption(f"Run created {report.get('created_at', 'unknown time')} · models: {report.get('embedding_model')} / {report.get('answer_model')} / {report.get('judge_model')}")
        summary = report["summary"]
        cols = st.columns(4)
        cards = [("Cases passed", f"{report['passed']}/{report['case_count']}", "All checks must pass"),
                 ("Retrieval recall", f"{summary['retrieval_recall']:.2f}", "Expected evidence found"),
                 ("Context precision", f"{summary['context_precision']:.2f}", "Relevant chunks / retrieved chunks"),
                 ("Groundedness", f"{summary['groundedness']:.2f}", "Hosted judge score")]
        for col, (label, value, help_text) in zip(cols, cards):
            with col:
                metric(label, value, help_text)
        cols = st.columns(4)
        for col, label, key in zip(cols, ("Answer relevance", "Retrieval ms", "Generation ms", "Total ms"),
                                   ("relevance", "retrieval_ms", "generation_ms", "total_ms")):
            with col:
                value = summary[key]
                metric(label, f"{value:.2f}" if key == "relevance" else f"{value:.0f} ms")

        st.markdown("#### Case results")
        import pandas as pd
        rows = [{"Case": c["name"], "User": c["user"], "Pass": "PASS" if c["passed"] else "FAIL",
                 "Recall": c["retrieval_recall"], "Precision": c["context_precision"],
                 "Grounded": c["groundedness"], "Relevant": c["relevance"], "Total ms": c["total_ms"]}
                for c in report["cases"]]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        for case in report["cases"]:
            state = "PASS" if case["passed"] else "FAIL"
            with st.expander(f"{state} · {case['name']} · {case['user']}"):
                st.markdown(f"**Question:** {case['question']}")
                st.markdown(f"**Answer:** {case['answer']}")
                st.markdown(f"**Judge rationale:** {case.get('judge_reason', '—')}")
                st.markdown(f"**Expected evidence:** {', '.join(case['expected_evidence']) or '(none)'}")
                st.markdown(f"**Found:** {', '.join(case['expected_evidence_found']) or '(none)'}")
                st.markdown(f"**Relevant-context labels:** {', '.join(case.get('relevant_evidence', [])) or '(none)'}")
                st.markdown(f"**Relevant context found:** {', '.join(case.get('relevant_evidence_found', [])) or '(none)'}")
                st.markdown(f"**Forbidden matches:** {', '.join(case['forbidden_terms_found']) or 'none'}")
                st.markdown("**Checks**")
                st.json(case["checks"])
                st.markdown("**Retrieved chunk text**")
                for text in case["retrieved_text"]:
                    st.code(text, language=None)

with tabs[2]:
    st.subheader("Security posture & audit trail")
    st.markdown("<div class='notice'><b>Important:</b> the bundled ACL is an educational example over synthetic files. The role selector is user-controlled and is not suitable as production authentication.</div>", unsafe_allow_html=True)
    a, b, c = st.columns(3)
    with a:
        st.markdown("#### Authorization")
        st.write("Group filtering happens before retrieval scoring and before generation context is built.")
    with b:
        st.markdown("#### Data handling")
        st.write("OpenRouter receives document chunks for embeddings and only authorized retrieved chunks for generation. The default free embedding model may retain requests for model improvement; use synthetic documents here and review provider data policies before sending sensitive material.")
    with c:
        st.markdown("#### Audit events")
        st.write("The log records user/groups, UTC time, result IDs, count, and query/answer hashes—not raw query or answer text.")
    st.markdown("#### Control matrix")
    st.markdown("| Control | Current behavior | Production requirement |\n|---|---|---|\n| ACL enforcement | Owner/group filter in retrieval | Trusted identity claims; server-side enforcement at storage/search layer |\n| Least-privilege context | Only permitted hits reach generation | Tenant isolation and deny-by-default policies |\n| Secret handling | Environment variables / Streamlit secrets | Secret manager; never commit credentials |\n| Audit | Local append-only JSONL | Durable, access-controlled, tamper-resistant audit store |\n| Prompt injection | Context marked as untrusted in system prompt | Defense in depth, output validation, adversarial testing |\n| Evaluation | Synthetic labeled cases and hard leak checks | Reviewed domain cases, monitoring, retention, release gates |")
    events = read_jsonl(rag.AUDIT)
    audit_is_example = False
    if not events:
        events = read_jsonl(DATA / "audit-example.jsonl")
        audit_is_example = bool(events)
    if audit_is_example:
        st.info("Showing illustrative audit rows; run a query to create live events.")
    st.markdown("#### Recent audit events")
    if events:
        import pandas as pd
        audit_rows = [{"Time (UTC)": e["time"], "User": e["user"], "Groups": ", ".join(e["groups"]),
                       "Results": e["result_count"], "Result IDs": ", ".join(e["result_ids"]),
                       "Query hash": e["query_sha256"][:14] + "…", "Answer hash": e["answer_sha256"][:14] + "…"}
                      for e in reversed(events[-100:])]
        st.dataframe(pd.DataFrame(audit_rows), use_container_width=True, hide_index=True)
        with st.expander("Inspect one complete audit record"):
            st.json(events[-1])
    else:
        st.caption("No audit events yet. Ask a question in the first tab.")
    st.caption("On Streamlit Community Cloud, this demo's local JSONL log is ephemeral and may reset on app restart. Use durable storage for production audit records.")

with tabs[3]:
    st.subheader("System details")
    st.markdown("#### Request path")
    st.code("PDF → text / table / image-caption extraction → chunks + ACL metadata\n"
            "→ permission filter → BM25 + hosted embedding similarity → rank fusion\n"
            "→ lightweight reranker → hosted answer model → audit JSONL", language="text")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("#### Active configuration")
        st.json({"provider": rag.MODEL_PROVIDER, "embedding_model": rag.EMBEDDING_MODEL,
                 "answer_model": rag.CHAT_MODEL, "evaluation_model": eval_runner.EVAL_MODEL,
                 "minimum_keyword_coverage": rag.MIN_KEYWORD_COVERAGE,
                 "minimum_vector_similarity": rag.MIN_VECTOR_SIMILARITY,
                 "index_path": str(rag.INDEX), "audit_path": str(rag.AUDIT)})
    with d2:
        st.markdown("#### Document set")
        st.write("Three synthetic PDF pages: shared handbook, Finance compensation table, and engineer oxygen runbook with an airlock illustration.")
        if PDF.exists():
            st.download_button("Download sample PDF", PDF.read_bytes(), file_name="moonbase-documents.pdf", mime="application/pdf")
        st.markdown("#### Evaluation thresholds")
        st.json({"retrieval_recall": eval_runner.MIN_RECALL,
                 "context_precision": eval_runner.MIN_CONTEXT_PRECISION,
                 "groundedness": eval_runner.MIN_GROUNDEDNESS,
                 "relevance": eval_runner.MIN_RELEVANCE})
    st.markdown("#### Deployment note")
    st.write("This UI is suitable for a portfolio deployment with synthetic content. Before using real documents, replace the role picker with authenticated identity, move the index and audit log to durable access-controlled services, add request limits, and protect uploaded content and reports.")

