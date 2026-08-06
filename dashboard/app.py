import streamlit as st
import requests
import plotly.graph_objects as go

st.set_page_config(
    page_title="ClinicalAgent Dashboard",
    page_icon="🩺",
    layout="wide",
)

API_BASE_URL = "http://localhost:8000"

# ---- Static evaluation data (from Phase 3 baseline + Phase 5 comparison) ----
PHASE3_BASELINE = {
    "Faithfulness": 0.79,
    "Answer Relevancy": 0.78,
    "Context Precision": 0.00,
    "Context Recall": 0.00,
}

PHASE5_COMPARISON = {
    "Faithfulness": {"Base": 0.7222, "Fine-tuned v2": 0.8333},
    "Answer Relevancy": {"Base": 0.9059, "Fine-tuned v2": 0.9310},
    "Context Precision": {"Base": 0.20, "Fine-tuned v2": 0.20},
    "Context Recall": {"Base": 0.33, "Fine-tuned v2": 0.33},
}

st.title("🩺 ClinicalAgent Dashboard")
st.caption("Agentic RAG system for clinical question answering — evaluation results & live query interface")

tab1, tab2, tab3 = st.tabs(["📊 Evaluation Scores", "🔍 Live Query", "ℹ️ About"])

# ---- TAB 1: Evaluation Scores ----
with tab1:
    st.subheader("Phase 3 — Full Pipeline RAGAS Baseline")
    st.caption("4-agent pipeline (Retrieval → NER → Reasoning → Critique), 5 clinical eval questions")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Faithfulness", f"{PHASE3_BASELINE['Faithfulness']:.2f}")
    col2.metric("Answer Relevancy", f"{PHASE3_BASELINE['Answer Relevancy']:.2f}")
    col3.metric("Context Precision", f"{PHASE3_BASELINE['Context Precision']:.2f}")
    col4.metric("Context Recall", f"{PHASE3_BASELINE['Context Recall']:.2f}")

    st.caption(
        "Context Precision/Recall are 0.00 at this vector store scale — expected, "
        "not a bug. These metrics compare retrieved chunks against ground-truth "
        "answers, and a small corpus won't always contain the exact supporting facts."
    )

    st.divider()

    st.subheader("Phase 5 — Fine-tuning Comparison (Base vs. LoRA v2)")
    st.caption("Raw LLM comparison, isolated from the agent pipeline — base Llama-3.1-8B-Instruct vs. mixed-format fine-tuned adapter")

    metrics = list(PHASE5_COMPARISON.keys())
    base_scores = [PHASE5_COMPARISON[m]["Base"] for m in metrics]
    finetuned_scores = [PHASE5_COMPARISON[m]["Fine-tuned v2"] for m in metrics]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Base", x=metrics, y=base_scores, marker_color="#94a3b8"))
    fig.add_trace(go.Bar(name="Fine-tuned v2", x=metrics, y=finetuned_scores, marker_color="#2563eb"))
    fig.update_layout(
        barmode="group",
        yaxis_range=[0, 1],
        yaxis_title="Score",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Fine-tuning improved Faithfulness by +0.11 and Answer Relevancy by +0.03. "
        "Context Precision/Recall are unchanged — expected, since fine-tuning the "
        "generator doesn't affect retrieval quality. The v2 adapter also became more "
        "conservative, declining to answer 3/5 questions rather than risk an ungrounded claim."
    )

# ---- TAB 2: Live Query ----
with tab2:
    st.subheader("Ask a Clinical Question")
    st.caption("Runs the full 4-agent pipeline via the FastAPI backend (Retrieval → NER → Reasoning → Critique)")

    question = st.text_area(
        "Question",
        placeholder="e.g. What are the treatment options for heart failure with reduced ejection fraction?",
        height=100,
    )

    if st.button("Run Query", type="primary"):
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            with st.spinner("Running pipeline..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/query",
                        json={"question": question},
                        timeout=120,
                    )
                    response.raise_for_status()
                    result = response.json()

                    reliable = result.get("is_reliable", False)
                    confidence = result.get("confidence_score", 0.0)

                    if reliable:
                        st.success(f"✓ RELIABLE — Confidence: {confidence:.2f}")
                    else:
                        st.warning(f"⚠ LOW CONFIDENCE — Confidence: {confidence:.2f}")

                    st.markdown("### Answer")
                    st.markdown(result.get("answer", "No answer returned."))

                    with st.expander("Sources"):
                        for c in result.get("citations", []):
                            st.write(f"- {c}")

                    with st.expander("Critique"):
                        st.write(result.get("critique", "No critique returned."))

                    if result.get("cached"):
                        st.caption("⚡ Served from Redis cache")

                except requests.exceptions.ConnectionError:
                    st.error(
                        f"Could not reach the API at {API_BASE_URL}. "
                        "Make sure it's running (`docker compose up -d` or `uvicorn src.api.main:app --port 8000`)."
                    )
                except requests.exceptions.HTTPError as e:
                    st.error(f"API returned an error: {e}")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

# ---- TAB 3: About ----
with tab3:
    st.markdown("""
    ### ClinicalAgent
    A production-grade multi-agent medical intelligence system that answers clinical
    questions using retrieved biomedical literature, medical NER, LLM reasoning grounded
    strictly in retrieved context, and automated hallucination detection with confidence scoring.

    **Architecture (6 layers):**
    1. **Data Ingestion** — PubMed API + PDF parser → chunker → PubMedBERT embedder → PostgreSQL + pgvector
    2. **Multi-Agent Pipeline** — Retrieval → NER → Reasoning → Critique (LangGraph)
    3. **Evaluation** — RAGAS metrics + LangSmith tracing
    4. **ML Risk Module** — XGBoost readmission classifier (MIMIC-III Demo), MLflow tracked
    5. **Fine-tuning** — LoRA on MedQA via Unsloth, before/after RAGAS comparison
    6. **Production Deployment** — FastAPI + Docker + GitHub Actions + Render

    Built by Ashhar Iqbal — B.Tech Biomedical Engineering, IIT Hyderabad.
    """)