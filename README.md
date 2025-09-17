# BelgeNavi — Technical README (RAG Assistant for TR Admin Workflows)

**BelgeNavi** is a **citation‑first** Retrieval‑Augmented Generation (RAG) assistant for Turkish administrative workflows.  
It uses an **agentic LangGraph pipeline** to turn a user question into a **checked checklist with citations**. Retrieval runs on **FAISS** (per‑language indexes: **TR / EN / AR**). Observability is via **Langfuse**. The project exposes a **FastAPI** service and a minimal **Streamlit** UI. A **Vector DB abstraction** enables portability to Qdrant/Chroma/Pinecone/Azure Search.

> **Disclaimer**: Informational guidance only — **not** legal advice. Always verify with official sources.

---

## Features at a Glance
- **Citation‑first answers** (URL + “last seen”).  
- **Multilingual retrieval** (Turkish, English, Arabic).  
- **Checklist composer** with required/optional docs, steps, fees, and links.  
- **Form preview** (optional) and **ICS schedule export** (optional).  
- **Safe Live Refresh** for public pages you explicitly allow (robots/ToS respected).  
- **Observability** with Langfuse; experiments with MLflow.  
- **Portable retrieval layer** (FAISS now; Qdrant/Chroma/Pinecone/Azure AI Search supported via abstraction).

---

## Architecture (High‑Level)

```
flowchart LR
  Q[Query] --> C[Classifier]
  C --> R[Retriever (FAISS/Qdrant)]
  R --> T[Citer (force citations)]
  T --> K[Checklist Composer (JSON)]
  K --> F[Form Preview (optional)]
  F --> G[Guardrails]
  G --> OUT[(Checklist + citations)]
```

**RAG flow**: Classify intent & language → Retrieve top passages → Force citations → Compose checklist JSON → (Optionally) suggest form fields → Guardrails (JSON shape & disclaimers) → Respond.

---

## Data & Retrieval

**Indexes (included as artifacts):**
- `belge_vdb_tr.faiss` / `belge_vdb_tr.pkl`
- `belge_vdb_en.faiss` / `belge_vdb_en.pkl`
- `belge_vdb_ar.faiss` / `belge_vdb_ar.pkl`

**ETL (lightweight), typical steps:**
1. Convert official PDFs/HTML to markdown  
2. Chunk text for retrieval  
3. Embed and build FAISS (optionally upsert to Qdrant)  

**Cataloging** tracks: `id`, `service`, `authority`, `lang`, `url`, `local_path`, `last_seen`, `ttl_days` to manage staleness and live refresh.

---

## API (Contracts)

### `POST /ask`
**Request**
```json
{ "query": "ستنتهي إقامتي القصيرة بإسطنبول، ما المطلوب؟", "lang": "ar" }
```

**Response (shape example)**
```json
{
  "checklist": {
    "required_docs": [],
    "optional_docs": [],
    "steps": [],
    "fees": "TRY ... (last seen 2025-08-27)",
    "links": [{ "label": "PMM fees", "url": "https://...", "last_updated": "2025-08-27" }],
    "disclaimer": "Informational, not legal advice."
  },
  "citations": [
    "PMM — Fees 2025 (source: https://... — last seen 2025-08-27)"
  ]
}
```

### `POST /form/preview`
Returns a suggested field schema (labels, types, hints) for likely online forms.

### `POST /schedule`
Returns an **ICS** file (`text/calendar`) suitable for calendar reminders.

> Tip: expose a lightweight `GET /health` for readiness/liveness checks.

---

## Configuration (Environment Variables)

**Models**
- `MODEL_NAME` — chat model (e.g., `Qwen-2-Instruct` / `Llama-3`)
- `EMBED_MODEL` — embeddings model (e.g., `gte-multilingual` / `bge-m3`)

**Indexes**
- `FAISS_TR_PATH`, `FAISS_EN_PATH`, `FAISS_AR_PATH` — absolute or repo‑relative paths

**Observability**
- `LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`

**Vector DB (optional)**
- `QDRANT_URL`, `QDRANT_API_KEY`

**Live Refresh (optional)**
- Per‑source allowlist inside the catalog + `ttl_days`

---

## End‑to‑End Flow (What happens on a request)

1. **POST /ask** arrives.  
2. **Classifier** returns language, service filters, freshness, and sections (JSON).  
3. **Retriever** picks FAISS index by language (TR/EN/AR), runs similarity search, returns top chunks.  
4. **Citer** converts chunks into cited bullets.  
5. **Checklist Composer** builds the checklist JSON (docs/steps/fees/links).  
6. **Form Preview** proposes a field schema (optional).  
7. **Guardrails** enforce JSON shape, decision sets, and disclaimers.  
8. **Summary** returns compact top‑level JSON (`bullets`, `citations`, `disclaimer`).

---

## Running Locally (Minimal)

### 1) Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then edit values
```

### 2) Indexes
Place FAISS artifacts for TR/EN/AR at the paths referenced by `FAISS_*_PATH` in `.env`.

### 3) Start API
```bash
uvicorn belge_fastapi:app --reload --port 8000
```

### 4) Smoke Test
```bash
curl -s -X POST http://127.0.0.1:8000/ask   -H "Content-Type: application/json"   -d '{"query":"ستنتهي إقامتي القصيرة بإسطنبول، ما المطلوب؟","lang":"ar"}'
```

### 5) Optional UI
Run the Streamlit client and point it to the API base URL.

---

## Known Gaps & Quick Fixes
- Add `GET /health` for the Streamlit check.  
- Fix a minor env typo (`LANGFUSE_SERCET_KEY` → `LANGFUSE_SECRET_KEY`) if present.  
- Move any hard‑coded FAISS/model paths into `.env`.  
- Prefer `langchain_community.embeddings.OpenAIEmbeddings` to avoid deprecation warnings.  
- Replace loose `eval` with `json.loads` when parsing model JSON output.

---

## Troubleshooting
- **Empty or off‑topic citations** → verify FAISS paths and correct `lang` index.  
- **JSON parse errors** → tighten output schema in prompts and validate with a JSON parser.  
- **High latency** → reduce chunk size/top‑k, use a smaller instruct model, or warm‑load models on startup.  
- **UI unhealthy** → ensure API has `GET /health` and correct base URL.

---

## Artifacts Included
- Prebuilt FAISS stores: `belge_vdb_{tr|en|ar}.faiss/.pkl`  
- Vector DB layer diagram (Mermaid export): `Untitled diagram _ Mermaid Chart-2025-08-02-134529.png`

---

## Credits
Thanks to the open‑source ecosystem behind Qwen/Llama, FAISS, Qdrant, LangChain, LangGraph, Langfuse, RAGAS, and FastAPI.
