# ClinicalAgent

A production-grade multi-agent medical intelligence system that answers complex clinical questions using retrieved biomedical literature, medical NER, LLM reasoning grounded strictly in retrieved context, and automated hallucination detection with confidence scoring.

Built by **Md Ashhar Iqbal** — B.Tech, IIT Hyderabad.

This is not a chatbot wrapper. It's an agentic RAG system with evaluation, built to production standards: 10,000+ indexable papers, RAGAS-scored responses, a persistent vector store, LangSmith-traced agent decisions, and a fully containerized deployment.

---

## Architecture

```
Layer 1 — Data Ingestion
  PubMed API + PDF Parser → Chunker → PubMedBERT Embedder → PostgreSQL + pgvector

Layer 2 — Multi-Agent Pipeline (LangGraph)
  Retrieval Agent → NER Agent → Reasoning Agent → Critique Agent

Layer 3 — Evaluation
  RAGAS metrics (faithfulness, answer relevancy, context precision, context recall)
  LangSmith observability and tracing

Layer 4 — ML Risk Module
  MIMIC-III Demo → XGBoost readmission classifier → MLflow tracking
  Exposed as a tool the Reasoning Agent calls when a patient profile is detected

Layer 5 — Fine-tuning
  LoRA fine-tune (Llama-3.1-8B-Instruct via Unsloth) on MedQA
  v1 → format collapse diagnosed → v2 mixed-format retrain → RAGAS before/after comparison

Layer 6 — Production Deployment
  FastAPI (async) → Docker → GitHub Actions CI → Render
  Streamlit dashboard for evaluation results and live queries
```

The core differentiator: the system doesn't just answer — the Critique Agent fact-checks every answer against the retrieved literature and assigns a confidence score, flagging hallucinations rather than presenting them as fact.

## Tech Stack

| Layer | Technology |
|---|---|
| Embeddings | NeuML/pubmedbert-base-embeddings (768-dim, domain-specific) |
| Vector Store | PostgreSQL + pgvector |
| Agent Framework | LangGraph |
| LLM | OpenRouter (`openrouter/free`, auto-routed) |
| NER | spaCy + custom medical keyword matching |
| Evaluation | RAGAS, LangSmith |
| ML Module | XGBoost + MLflow |
| Fine-tuning | LoRA via Unsloth |
| Backend | FastAPI (async) |
| Caching | Redis |
| Containerization | Docker + docker-compose |
| CI | GitHub Actions |
| Deployment | Render |
| Dashboard | Streamlit |

## Results

- **Phase 3 baseline** (full 4-agent pipeline, 5 clinical eval questions): Faithfulness 0.79, Answer Relevancy 0.78. Context Precision/Recall are 0.00 at this vector-store scale — an expected, documented consequence of a small corpus, not a bug.
- **Phase 5 fine-tuning** (base vs. LoRA v2, isolated LLM comparison): Faithfulness improved +0.11 (0.72 → 0.83), Answer Relevancy +0.03. The retrain also introduced a documented tradeoff — the fine-tuned model became more conservative, declining to answer 3/5 questions rather than risk an ungrounded claim.
- **Critique Agent** correctly distinguishes reliable answers (0.90 confidence, grounded) from hallucinated ones (0.45 confidence, fabricated statistics flagged) on held-out questions.

## Running Locally

The fully containerized stack (Postgres + Redis + API) is the verified, reliable way to run this system end-to-end.

```bash
git clone https://github.com/mdashhariqbal/clinicalagent.git
cd clinicalagent
cp .env.example .env   # fill in your API keys (OpenRouter, NCBI, LangSmith)

docker compose up -d
docker compose ps       # postgres, redis, api should all be healthy

curl http://localhost:8000/health
```

To ingest data and populate the vector store, and to run the Streamlit dashboard, see the setup notes in `src/` and `dashboard/app.py`.

Data ingestion (`main.py`), evaluation (`scripts/run_evaluation.py`), and risk model training (`scripts/train_risk_model.py`) are documented via their own docstrings/`--help` — these require local data and API keys the quick-start above doesn't cover.

## Deployment

**Live attempt:** deployed to Render as a Docker web service, wired to GitHub Actions CI (see below).

**Known constraint — free-tier RAM:** Render's free tier caps a service at 512MB RAM. This stack loads PyTorch, sentence-transformers (PubMedBERT), spaCy, and XGBoost simultaneously inside one process. Measured actual runtime memory via `docker stats` on the fully running container: **~945MB** — well above the free-tier ceiling, causing the deployed instance to crash-loop on Render's free plan.

This was diagnosed, not guessed at:
1. Confirmed the failure was a runtime OOM, not a build failure — the image built and ran cleanly locally and via `docker run`/`docker compose` against real Postgres/Redis.
2. Ruled out image size as the cause (image size and runtime memory are different things) by measuring live process memory with `docker stats` while the container was actually serving requests.
3. Considered swapping to a CPU-only PyTorch build to shrink the footprint, and considered moving embeddings to a hosted API to remove the local model entirely — both viable but nontrivial (a hosted embedding model would need to match, or force re-embedding, the existing 768-dim PubMedBERT vector store).
4. Decided against reworking the architecture just to fit a free-tier ceiling for a portfolio project. Render Standard ($25/mo) or a smaller/CPU-only footprint would resolve this in a paid or production setting — documenting the tradeoff is the honest and correct call here rather than either paying to force a green checkmark or silently shipping a broken link.

**The reliable way to see this system running is the local Docker Compose stack above** — it's the exact same image, same code, same containerized shape as what's deployed, just without the free-tier RAM ceiling.

## CI/CD

GitHub Actions (`.github/workflows/deploy.yml`) runs on every push to `main`:
- `lint-and-check` — installs `requirements.txt` fresh on Ubuntu, then import-checks every core module (`src.config`, `src.database.*`, `src.agents.pipeline`, `src.evaluation.ragas_evaluator`, `src.ml.risk_tool`, `src.api.main`). This is a real fresh dependency-resolution test — it's what originally caught a pydantic/fastapi/langchain-core version conflict that only surfaced during a clean install, never locally where the environment was built incrementally.
- `docker-build` — gated on `lint-and-check` passing; builds the production image to confirm it still builds cleanly.

Integration tests against a live Postgres/Redis are deliberately not run in CI, to stay within free-tier CI minutes for a personal project. This is a documented, deliberate scope decision, not an oversight — it's a natural next step in a paid or team CI setup.

## Why This Domain

Clinical Q&A was chosen deliberately because it's one of the harder domains for LLM reliability — that constraint is what makes every engineering decision in this system defensible rather than incidental. The Critique Agent, RAGAS evaluation, and confidence scoring all exist because hallucination is more costly in a clinical context than in a general-purpose chatbot. The engineering stack itself (LangGraph, pgvector, RAGAS, FastAPI, Docker) is domain-agnostic and transfers directly to other domains — this project goes deep on one domain rather than shallow across several, deliberately.

## License

MIT
