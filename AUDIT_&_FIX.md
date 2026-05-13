# Senior-style evaluation — Skyclad assignment vs this repo

This document maps the internship brief to what is implemented in the codebase, calls out gaps and bugs, and proposes concrete fixes. It is based on a full pass over the repository layout and the files under `backend/`, `docker-compose.yml`, and `README.md` (and the duplicate `README2.md`).

---

## Executive summary

**Overall:** The project hits the *spirit* of the assignment in several strong areas (custom agent loop, hybrid retrieval + reranking, refusal-oriented prompting, logging, a written decisions log, and an eval script). There are also **several hard engineering and submission gaps** that a reviewer would flag immediately: broken Docker packaging, conversational memory shared across all API clients, “semantic memory” that is never wired up, an evaluation harness that poisons later tests with earlier conversation state, and README claims that oversell memory and chunking versus the actual code.

**Verdict:** Not a clean “fully satisfied” submission until the Docker/setup and eval/memory issues are fixed and the README is aligned with behavior (especially memory types and the mandatory README section wording).

---

## Requirement checklist (brief → repo)

### 1. Corpus — arXiv cs.AI, ~50–200 PDFs, defensible ingestion

| Expectation | Status | Evidence / gap |
|-------------|--------|----------------|
| cs.AI category | **Met** | `backend/data_pipeline/ingest.py` uses `query="cat:cs.AI"`. |
| “Last 90 days” or explicit equivalent | **Partial / weak** | Ingest takes the **50 most recent** submissions by `SubmittedDate`. That is usually *much narrower* than 90 days and is **not** an explicit date-bounded query (e.g. no `submittedDate:[now-90d TO now]` style filter). Defensible as “recent snapshot,” but it is **not** what the PDF literally asked for unless you argue it in the README. |
| Ingest → chunk → embed → store | **Met** | `ingest.py` + `build_index.py` + FAISS + `metadata.pkl`. |

**Risks:** `ingest.py` and `build_index.py` resolve paths with `os.getcwd()`. If someone runs modules from the wrong working directory, `data/` resolves incorrectly. `retriever.py` uses `Path(__file__).resolve().parents[...]`, which is more robust — **inconsistent root resolution** across pipeline vs API.

**Solutions:**

1. Add an explicit date filter in the arXiv query or post-filter results by `result.updated` / published date to the last 90 days, and document “we used N papers in window W.”
2. Centralize `PROJECT_ROOT` (e.g. from `Path(__file__).resolve().parents[...]` or an env var) and use it in ingest, build_index, and config path joins.

---

### 2. Conversational interface

| Expectation | Status | Evidence / gap |
|-------------|--------|----------------|
| CLI / web / notebook | **Met (minimal)** | FastAPI + Swagger (`/chat`) in `backend/app/main.py`. |
| README “frontend” | **Misleading** | README folder tree describes `frontend/` Streamlit; **no `frontend/` directory exists** in this workspace. |

**Solutions:** Either add a minimal Streamlit/CLI client or remove the frontend from the documented tree until it exists.

---

### 3. Agent loop — retrieve, clarify, tool, refuse, answer

| Action | Status | Notes |
|--------|--------|--------|
| Retrieve | **Met** | `retrieve_technical_documents` tool in `tools.py`, executed in `agent.py`. |
| Ask clarifying question | **Partial** | No dedicated “clarify” tool or structured branch — relies on the model returning **plain text** without tool calls. That can work but is **harder to observe** as a first-class “action.” |
| External tool | **Partial (acceptable if framed well)** | `calculate` is a **local** AST evaluator — it **does** match the brief’s examples (“calculator”). There is **no** network tool (arXiv API, web search). The brief only requires *at least one* tool; calculator qualifies, but do **not** oversell “external API” in docs. |
| Refuse / I don’t know | **Partial** | Driven by system prompt + empty retrieval string from `agent._run_retriever`. There is no explicit “refuse” tool or deterministic guardrail beyond the LLM following instructions. |
| Answer | **Met** | Final assistant message when no tools are called. |

**Solutions:**

1. Optional: add a small **structured** “clarify” or “refuse” path (e.g. forced JSON or a dedicated tool) so observability matches the rubric’s “decide between actions.”
2. If you want a stronger “external” story without cost: **arXiv metadata fetch** as a tool (complements your corpus).

---

### 4. Memory — conversation vs semantic vs episodic

| Expectation | Status | Evidence / gap |
|-------------|--------|----------------|
| Memory affects retrieval and answers | **Partial** | Sliding window in `MemoryManager` feeds Groq messages — good for **anaphora** in *chat*, but the **retriever is only called with the tool query string**; there is no deterministic “query rewriting from history” step. |
| Beyond “last N messages” | **Not met (as implemented)** | `semantic_knowledge` + `get_system_prompt_context()` exist, but **`extract_semantic_facts` is `pass`** and nothing calls `add_semantic_fact` from `agent.py`. So **semantic memory does not function.** |
| Episodic memory | **Not demonstrated** | No episode store, summaries, or “what we did in this session” structure beyond raw chat lines. |
| README / interview story | **Gap** | The assignment explicitly asks you to show you understand **conversation vs semantic vs episodic** and which matter here. The README focuses on sliding window only and claims “semantic memory states” in the folder blurb while the code path is unused. |

**Critical multi-user bug:** `main.py` constructs **one global** `SkycladAgent()`. Every HTTP client shares the **same** `MemoryManager.history`. That violates typical expectations of “conversation” and is a **serious** flaw for any multi-tester or concurrent demo.

**Solutions:**

1. **Session IDs:** Add `session_id` to `/chat` (header or body). Maintain `dict[session_id, SkycladAgent]` or `dict[session_id, MemoryManager]` with TTL eviction.
2. Implement **one** minimal semantic-memory path: e.g. after each turn, regex or cheap classifier extracts “user prefers short answers” and calls `add_semantic_fact`, or a single LLM JSON block appended only when `FACT:` appears — anything that proves the plumbing works.
3. Update README with an honest paragraph: what you implemented vs what you’d add for episodic (e.g. summarized tool traces per session).

---

### 5. Retrieval beyond naive top‑k cosine

| Expectation | Status |
|-------------|--------|
| Hybrid + fusion + rerank | **Strong** — FAISS + BM25 + RRF + cross-encoder in `retriever.py`. |

**Config inconsistency (bug / dead config):**

- `retriever.py` uses `getattr(settings, "RERANK_THRESHOLD", 0.0)`.
- `config.py` defines `SIMILARITY_THRESHOLD: float = 0.6` but **nothing reads it**; **`RERANK_THRESHOLD` is not defined** on `Settings`. So the “strict threshold” is always **0.0** unless you set a dynamic attribute elsewhere (you do not).

**Solutions:** Add `RERANK_THRESHOLD: float = 0.0` to `Settings`, tune it, and either remove `SIMILARITY_THRESHOLD` or wire it into retrieval with a documented meaning.

**Documentation nit:** `build_index.py` chunks by **character length** (`len` on Python string), not tokenizer tokens. README language should say **characters** (or switch to a tokenizer-based chunker).

---

### 6. Evaluation — ≥10 questions, ≥2 refuse/clarify, “how you think about correctness”

| Expectation | Status | Evidence / gap |
|-------------|--------|----------------|
| ≥10 cases | **Met** | `EVAL_TEST_CASES` has 10 cases in `backend/eval/evaluation_harness.py`. |
| ≥2 refuse/clarify | **Met** | Four cases use `should_refuse=True`. |
| Sound methodology | **Weak** | **Same `SkycladAgent` across all cases** → `chat()` accumulates history; later cases see prior turns. That **invalidates** independence of tests and can flip routing behavior. |
| Keyword checks | **Brittle** | e.g. DINORANKCLIP-specific expectations **fail** if that paper is not in the downloaded 50 PDFs (corpus drift). |
| Default pass | **Too loose** | `_check_response` returns `True` for many `expected_type` values without assertions (e.g. `clarify` without `should_refuse`). |

**Solutions:**

1. **New agent per case** or call `agent.memory.clear()` between cases.
2. Prefer **behavioral** checks: tool was/wasn’t called (would require returning `trace` from `chat`), or stable substrings independent of a single paper title.
3. Add 2–3 **gold** questions with **fixed** short expected answers if you ship a **frozen** eval subset of PDFs or a tiny “fixture” corpus for CI.

---

### 7. Observability — “what the agent decided and why”

| Expectation | Status | Evidence / gap |
|-------------|--------|----------------|
| Logs for loop, tools, retrieval | **Good baseline** | `logger` calls in `agent.py`, `retriever.py`, `tools.py`. |
| Per-request inspectability from API | **Partial** | `/test-retrieval` helps for RAG only. **`/chat` returns only `reply`**, not a structured trace (tool names, queries, scores, loop count). Reviewers grading observability will look for **one place** (JSON response or trace ID) that mirrors the logs. |

**Solutions:** Extend `ChatResponse` with optional `trace: { "loops": ..., "tool_calls": [...], "retrieval": { "scores": ... } }` (even behind `?debug=1`).

---

### 8. Depth over breadth — ablation tied to eval

| Expectation | Status | Evidence / gap |
|-------------|--------|----------------|
| Ablation exists | **Partial** | `backend/eval/ablation_study.py` compares **counts** of chunks with/without reranker and hybrid vs single-channel sizes. |
| “Eval scores with and without” | **Not met** | The assignment asks for evidence the technique **moved your eval**. There is **no** link from ablation metrics to `EvaluationHarness` pass rate or labeled retrieval quality. |

**Solutions:** Run harness twice (feature flag to disable reranker or swap threshold), report pass rate delta in README, or add a tiny labeled set (10 queries with “relevant chunk ids”) and report MRR.

---

### 9. Hard constraints and submission hygiene

| Topic | Status |
|-------|--------|
| No RAG-in-a-box SaaS | **Met** — custom FAISS/BM25/agent. |
| Hosted LLM OK | **Met** — Groq. |
| README: setup, architecture, decisions, another week, limitations | **Mostly met** — content exists; the brief’s **exact** heading “**What you’d do with another week**” is not a standalone section title (it is nested under “Known Limitations & Future Work”). Low risk if content is there, but **mirroring their wording** is safer. |
| Demo video length | **Risk** | README states **9:02**; the brief asks **5–8 minutes** and **face on camera** — length is a **submission-format** risk (you already note it; trimming is safer). |

---

## Critical build / ops issues

### A. Docker build is very likely broken

`backend/Dockerfile` contains:

```dockerfile
COPY requirements.txt .
```

The repository has **`backend/requirements.txt` only** (no root `requirements.txt` in this workspace). A standard `docker compose build` from the repo root should **fail** at that step unless an untracked root file exists.

**Fix:** e.g. `COPY backend/requirements.txt ./requirements.txt` or `COPY backend/requirements.txt /app/requirements.txt` and adjust `pip install`.

### B. `README2.md`

Appears to be a **partially duplicated / malformed** README (broken fenced blocks mid-document). For submission, **delete or fix** so reviewers are not confused which file is canonical.

---

## Code quality and maintainability

| Item | Note |
|------|------|
| Types | Reasonable in places (`list[dict]`), thin elsewhere. |
| Custom exceptions | `backend/utils/exceptions.py` exists but is **unused** in the agent/retriever path. |
| Tests | **No** automated tests (pytest) for calculator, retriever fusion, or API. |
| Stray module-level strings | `agent.py` ends with large `""" ... """` / `''' ... '''` blobs — valid Python but noisy; better as comments or docs elsewhere. |

---

## What you did especially well (keep in interviews)

1. **Clear architectural story** — explicit while-loop agent, tool routing, and hybrid retrieval + reranker with logging.
2. **Honest failure-mode section** in README (empty corpus, ambiguity, OOD, contradiction) aligned with prompts.
3. **Retriever design** is above the “naive RAG” bar and is easy to explain on a whiteboard.
4. **Frugal stack** (Groq + local embeddings) fits the cost constraint narrative.

---

## Prioritized fix list (if you only do five things)

1. **Fix Dockerfile** `COPY` paths so `docker compose up --build` works from a clean clone.
2. **Per-session agent memory** (session id) — fixes a severe correctness/privacy issue.
3. **Reset memory between eval cases** (or new agent per case) and soften corpus-specific keyword tests.
4. **Wire `RERANK_THRESHOLD` into `Settings`**; remove or use `SIMILARITY_THRESHOLD`.
5. **README alignment:** memory types, chunk = characters, remove ghost `frontend/`, add optional **`trace`** field to `/chat` for observability.

---

## Direct answer to “did I fulfill everything?”

**Not completely.** Core RAG + agent + hybrid retrieval + written rationale are in good shape for the rubric’s “depth” and “decisions” axes. Gaps that a senior would treat as **blocking or high severity** are: **Docker install path**, **shared global conversation state**, **eval harness contamination**, **semantic memory unused vs documented**, **weak coupling of ablation to eval scores**, and **submission-format risks** (video length, exact README headings). Addressing the prioritized list above moves this much closer to a submission that matches both the letter and the engineering bar of the brief.

---

## Changelog — fixes applied after this audit (2026-05-12)

**Note:** Sections above capture the **pre-fix** audit. The bullets below describe **engineering changes now present in the repo**.

The following changes were implemented in the repository to address the issues above and the external reviewer feedback (session isolation, eval harness presence, calculator DoS, Dockerfile, README honesty, ablation numbers, tests).

1. **Per-session API isolation:** `backend/app/session_registry.py` maps each `session_id` to its own `SkycladAgent` while sharing one `AdvancedRetriever` loaded in a FastAPI **lifespan** hook (`backend/app/main.py`). Responses include `session_id`; `DELETE /session/{session_id}` clears a session. LRU eviction respects `SESSION_MAX` in `backend/config.py`.
2. **Docker build:** `backend/Dockerfile` now copies `backend/requirements.txt` correctly when build context is the repo root.
3. **Eval harness hygiene:** `backend/eval/evaluation_harness.py` creates a **fresh** `SkycladAgent` per case (shared retriever optional), adds **12** cases with broader keyword checks and richer refusal/clarify substring detection, and exposes `compare_reranker_ablation()` to report **pass-rate delta with vs without** the cross-encoder reranker.
4. **Ablations tied to metrics:** `backend/eval/ablation_study.py` prints chunk-count deltas **and** calls `compare_reranker_ablation()` for pass-rate comparison.
5. **Retriever switches:** `retrieve_and_rerank(..., use_reranker=..., apply_score_threshold=...)` supports ablations; `RERANK_THRESHOLD` is a first-class `Settings` field (removed unused `SIMILARITY_THRESHOLD`).
6. **Calculator hardening:** `safe_calculate` caps AST node count, absolute exponent, and estimated `log10` magnitude of `**` results to mitigate `2**100000000`-style hangs; README/tool text no longer implies “secure” beyond those threat models.
7. **Semantic memory wired:** `MemoryManager.ingest_user_stated_facts()` parses simple “Remember … / Always … / My preference is …” lines; `SkycladAgent.chat()` calls it each turn.
8. **Stable paths for pipelines:** `backend/paths.py` provides `project_root()`; `ingest.py` and `build_index.py` use it instead of `os.getcwd()` alone.
9. **Observability:** `GET /chat?debug=true` returns a compact `trace` (`loops`, `tool_calls` previews).
10. **Tests:** `pytest.ini` + `tests/test_calculator.py` and `tests/test_memory.py` (4 tests) run clean.
11. **Docs cleanup:** Root `README.md` rewritten for accurate setup timing, session semantics, eval/ablation commands, memory table, and folder layout; duplicate/malformed `README2.md` removed.
12. **Packaging:** `backend/eval/__init__.py` added so `python -m backend.eval.evaluation_harness` / `ablation_study` resolve reliably.
13. **Config hygiene:** Pydantic v2 `SettingsConfigDict` replaces deprecated nested `class Config`.
14. **Noise reduction:** Large decorative string blocks removed from `agent.py` / `main.py` ends during refactors.

If you re-submit or reuse this project, **commit and push** all of `backend/eval/`, tests, and README changes so reviewers see them on GitHub, and keep a short “how to run eval in 1 command” section visible in the README (now included).
