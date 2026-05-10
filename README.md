# Skyclad Ventures Agentic RAG

A compact autonomous Agentic RAG repository that routes user queries through a Python agent loop, chooses between technical document search, calculator execution, clarification, or refusal, and uses a Groq-native tool interface for deterministic tool invocation.

## Table of Contents

1. Setup & Run Instructions
2. Architecture Overview
3. Engineering Decisions Log (CRITICAL)
4. Failure Modes & Defenses
5. Future Work
6. Demo Video

## Setup & Run Instructions

```powershell
# Activate the workspace virtual environment
agenticRAG_venv\Scripts\activate

# Install backend dependencies
pip install -r backend/requirements.txt

# Optional: install frontend dependencies if required
pip install -r frontend/requirements.txt
```

Create a `.env` file in the repository root and set your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key
```

Run data ingestion and indexing:

```powershell
python backend/data_pipeline/ingest.py
python backend/data_pipeline/build_index.py
```

Start the FastAPI server:

```powershell
uvicorn backend.app.main:app --reload --port 8000
```

## Architecture Overview

- `frontend/app.py` sends user requests to FastAPI in `backend/app/main.py`.
- FastAPI forwards requests into `SkycladAgent.chat()` in `backend/app/agent.py`.
- The agent runs a raw Python `while` loop, assembles the system prompt and memory window, and dispatches native tool calls through the Groq SDK.
- Tool implementations are defined in `backend/app/tools.py`.
- Search tools combine the advanced retriever in `backend/app/retriever.py` with a calculator tool and explicit clarification/refusal handling.

## Engineering Decisions Log (CRITICAL)

- PDF Extraction
	- Uses `PyMuPDF` in `backend/data_pipeline/ingest.py` because it preserves whitespace and layout for complex arXiv-style papers better than PyPDF2.
- Vector Search
	- Uses FAISS `IndexFlatIP` with L2-normalized embeddings in `backend/data_pipeline/build_index.py`, which mathematically equates to exact cosine similarity and ignores document magnitude.
- Retrieval Engine
	- Builds a hybrid retrieval pipeline in `backend/app/retriever.py`: BM25 lexical search plus FAISS semantic search, merged with Reciprocal Rank Fusion (RRF), then reranked by `ms-marco` cross-encoder with a confidence threshold.
- Agent Framework
	- Uses raw Python control flow in `backend/app/agent.py` and the native Groq SDK instead of LangChain/LangGraph for full observability, deterministic tool execution, and prevention of hidden recursive loops.

## Failure Modes & Defenses

- Empty corpus results
	- If no document chunk meets the reranker threshold, the system refuses rather than hallucinating.
- Ambiguous queries
	- The agent exits the tool loop and asks for clarification when user intent is vague.
- Out-of-domain questions
	- The system prompt enforces strict refusal for off-domain or non-technical requests.

## Future Work

- Migrate the in-memory sliding window in `backend/app/memory.py` to a Redis-backed persistent memory store.
- Add an LLM-based query rewriter to resolve pronouns in follow-up questions before invoking search tools.

## Demo Video

- Demo link placeholder: `<INSERT UNLISTED YOUTUBE/LOOM LINK HERE>`
