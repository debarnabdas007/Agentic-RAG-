# Agentic RAG Architecture

Instead of building a rigid, linear RAG pipeline using a "RAG-in-a-box" framework, this repo is a **FastAPI** backend where an LLM (Groq **Llama 3.1 8B**) chooses tools in a **native Python** loop: retrieve from a local cs.AI arXiv corpus, run a **bounded** calculator, clarify or refuse in plain text, then answer.

## Table of Contents

1. [Setup Instructions](#setup-instructions)
2. [API: sessions and observability](#api-sessions-and-observability)
3. [Evaluation and ablations](#evaluation-and-ablations)
4. [Architecture Overview](#architecture-overview)
5. [Folder structure](#folder-structure)
6. [Engineering Decisions Log](#engineering-decisions-log)
7. [Handling Failure Modes](#handling-failure-modes)
8. [Known Limitations](#known-limitations)
9. [What you'd do with another week](#what-youd-do-with-another-week)
10. [Demo Video](#demo-video)

---

## Setup Instructions

**Realistic first-time setup:** downloading PDFs, embedding, and pulling PyTorch / SentenceTransformers models usually takes **roughly 15–40 minutes** depending on bandwidth and CPU/GPU. The API container starts quickly **after** `data/vector_store/` exists.

1. **Clone the repository**

   ```bash
   git clone https://github.com/debarnabdas007/Agentic-RAG-.git
   cd Agentic-RAG-
   ```

2. **Environment variables**

   Create `.env` in the **repository root**:

   ```env
   GROQ_API_KEY=your_api_key_here
   ```

3. **Python venv + one-time data pipeline** (from repo root)

   ```bash
   python -m venv agenticRAG_venv
   agenticRAG_venv\Scripts\activate          # Windows
   # source agenticRAG_venv/bin/activate     # macOS/Linux

   pip install -r backend/requirements.txt

   python -m backend.data_pipeline.ingest    # downloads PDFs → data/raw_pdfs/
   python -m backend.data_pipeline.build_index
   ```

   Chunking is **character-based** sliding windows (`CHUNK_SIZE` / `CHUNK_OVERLAP` in `backend/config.py`), not tokenizer tokens.

4. **Run the API**

   **Option A — Docker (after index exists):**

   ```bash
   docker compose up --build
   ```

   `docker-compose.yml` mounts `./data` so the container can read `data/vector_store/` built in step 3.

   **Option B — local uvicorn:**

   ```bash
   uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
   ```

5. **Tests**

   ```bash
   python -m pytest tests
   ```

Open **http://localhost:8000/docs** — use **`POST /chat`**.

---

## API: sessions and observability

- **`session_id` (required for correct behavior):** Pass a stable client-generated id (e.g. UUID) on every `/chat` call. The server maps each id to its **own** `SkycladAgent` + `MemoryManager`. If you omit `session_id`, each request starts a **new** conversation (no memory carryover).
- **Response:** `POST /chat` returns `{ "reply", "session_id", "trace" }`. Always persist `session_id` from the response if you did not send one.
- **Debug trace:** `POST /chat?debug=true` adds a small JSON `trace` (`loops`, `tool_calls` with argument previews).
- **Reset:** `DELETE /session/{session_id}` drops that conversation from the in-memory registry (LRU cap: `SESSION_MAX` in config).

**Memory model (honest):**

| Type | What ships | Role |
|------|----------------|------|
| **Conversation** | Sliding window of recent chat + tool messages | Coreference and multi-step tool use |
| **Semantic** | Regex-extracted user lines like `Remember that …` / `Always …` → injected into system prompt | Long-lived preferences without an extra LLM call |
| **Episodic** | Not a separate store | Raw tool transcripts live in the sliding window only |

---

## Evaluation and ablations

**Harness (≥10 hand-written cases, ≥4 refusal/clarify-style):**

```bash
python -m backend.eval.evaluation_harness
```

Requires `GROQ_API_KEY` and a built index. Each case uses a **fresh** agent so history does not leak between tests.

**Ablations (chunk counts + harness pass rate with vs without reranker):**

```bash
python -m backend.eval.ablation_study
```

---

## Architecture Overview

```
┌───────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: OFFLINE DUAL-INDEX KNOWLEDGE BASE                                │
│                                                                           │
│ arXiv PDFs ──> PyMuPDF ──> Overlap Chunking ──┬─> Embed (MiniLM) ──> FAISS│
│ (Layout safe)                                 └─> Tokenize ────────> BM25 │
└───────────────────────────────────────────────▲───────────────────────────┘
                                                │ (Vector/Keyword Sync)
┌───────────────────────────────────────────────▼───────────────────────────┐
│ PHASE 2: ONLINE AGENTIC WORKFLOW (bounded while-loop)                   │
│                                                                           │
│  POST /chat ──> per-session MemoryManager (sliding window + semantic)     │
│                                      │                                    │
│      ┌───────────────────────────────▼──────────────────────────────┐     │
│      │  LLM: Groq tool-calling (retrieve / calculate)               │<──┐ │
│      └─┬──────────────────┬──────────────────────┬────────────────┬─┘   │ │
│        │                  │                      │                │     │ │
│   ┌────▼────┐        ┌────▼─────┐           ┌────▼────┐           │     │ │
│   │ SEARCH  │        │CALCULATE │           │ CLARIFY │           │     │ │
│   │ CORPUS  │        │ (AST)    │           │ /REFUSE │           │     │ │
│   └────┬────┘        └────┬─────┘           └────┬────┘           │     │ │
│        │                  │                      │                │     │ │
│        └──> Hybrid RRF + cross-encoder rerank + RERANK_THRESHOLD ─┘     │ │
│                                                                           │
│  max_loops=5  (circuit breaker)                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## Folder structure

```text
repo/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI, lifespan, session registry
│   │   ├── agent.py             # Agent loop
│   │   ├── retriever.py         # FAISS + BM25 + RRF + rerank
│   │   ├── tools.py             # Calculator + tool schemas
│   │   ├── memory.py            # Sliding window + lightweight semantic facts
│   │   └── session_registry.py  # Per-session agents, shared retriever
│   ├── data_pipeline/
│   │   ├── ingest.py
│   │   └── build_index.py
│   ├── eval/
│   │   ├── evaluation_harness.py
│   │   └── ablation_study.py
│   ├── paths.py                 # Stable project_root()
│   ├── config.py
│   ├── requirements.txt
│   └── Dockerfile
├── tests/                       # pytest (calculator + memory)
├── pytest.ini
├── data/                        # gitignored: raw_pdfs/, vector_store/
├── docker-compose.yml
├── evaluation_cursor.md         # prior audit + changelog
└── README.md
```

---

## Engineering Decisions Log

- **Agent:** Raw Python loop + Groq tools instead of LangChain/LangGraph — full control, easy logging, hard **`max_loops`** cap.
- **PDFs:** PyMuPDF for two-column arXiv layouts.
- **Vectors:** L2-normalized embeddings + **FAISS IndexFlatIP** (= cosine similarity).
- **Retrieval:** BM25 + FAISS fused with **RRF (k=60)**; **MS MARCO cross-encoder** rerank; chunks filtered by **`RERANK_THRESHOLD`** (default **0.0** so irrelevant scores drop out and the agent sees **empty** context rather than hallucination bait).
- **Calculator:** `ast` walking — **no code execution**; caps on **AST size**, **exponent**, and **estimated result magnitude** to avoid `2**100000000`-style worker hangs.
- **Sessions:** One **shared** `AdvancedRetriever` per process, one **`SkycladAgent` per `session_id`** — fixes cross-user memory bleed in the old global agent design.

---

## Handling Failure Modes

- **No relevant corpus:** rerank scores ≤ threshold → **no chunks** → tool message states DB empty → system prompt forces admitting the gap.
- **Ambiguous:** model may ask a clarifying question without calling retrieve (observable in logs / `?` in reply).
- **Out-of-domain:** prompt-level refusal; empirically imperfect on small models — **run `evaluation_harness`** and iterate prompts from measured failures.
- **Contradicting sources:** prompt asks the model to surface disagreement.

---

## Known Limitations

- Ingest pulls the **latest N cs.AI papers** (default 50), not a strict rolling 90-day window; argue or tighten in `ingest.py` if you need exact dates.
- Session store is **in-process RAM**; scale-out needs Redis (or sticky sessions) + external session store.
- Refusal/clarify behavior is still **LLM-dependent** for free-text turns; the harness exists to track regression.

---

## What you'd do with another week

- Redis-backed `session_id` + optional user auth.
- Explicit **query rewrite** step using chat history before retrieval.
- Stronger refusal path (lightweight classifier or structured output).
- Layout-aware **section-based chunking** instead of fixed character windows.
- CI running pytest + a **mocked** subset of the eval harness without network.

---

## Demo Video

[Watch the Architecture & Live Demo](https://www.youtube.com/watch?v=mV9ksCoA5Xs)


---

## Local audit notes

See **`evaluation_cursor.md`** for a prior rubric-style review and a **changelog** of post-feedback fixes.
