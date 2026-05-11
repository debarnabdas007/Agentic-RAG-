# Agentic RAG Architecture

Instead of building a rigid, linear RAG pipeline using a "RAG-in-a-box" framework, I built a deployable, stateful backend where a Large Language Model acts as an autonomous routing engine. It decides for itself when to calculate math, when to search a dual-engine knowledge base, when to ask for clarity, and when to refuse a prompt.

## Table of Contents
1. [Setup Instructions](#setup-instructions)
2. [Architecture Overview](#architecture-overview)
3. [Folder strcuture](#folder-structure)
4. [Engineering Decisions Log](#engineering-decisions-log)
5. [Handling Failure Modes](#handling-failure-modes)
7. [Known Limitations & Future Work](#known-limitations--future-work)
8. [Demo Video](#demo-video)

---

## Setup Instructions

You can clone and run this entire system in under 5 minutes using Docker.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/debarnabdas007/Agentic-RAG-.git
   cd Agentic-RAG-
   ```

2. **Set up your environment variables:**
   Create a `.env` file in the root directory and add your Groq API key:

   ```env
   GROQ_API_KEY=your_api_key_here
   ```

3. **Run the Data Pipeline (One-time setup):**
   (Note: The repo does not contain the 50 arXiv PDFs to save space. Run these to pull the papers and build the FAISS/BM25 indices locally).

   ```bash
   # Create a virtual environment and activate it
   python -m venv agenticRAG_venv
   source agenticRAG_venv/bin/activate  # On Windows: agenticRAG_venv\Scripts\activate

   # Install requirements (Note: file is located inside the backend directory)
   pip install -r backend/requirements.txt

   # Download papers and build the vector database
   python -m backend.data_pipeline.ingest
   python -m backend.data_pipeline.build_index
   ```
4. **Spin up the Backend:**

   ```bash
   docker-compose up --build
   ```

   The API will be live at http://localhost:8000/docs where you can interact with the Agent via the Swagger UI /chat endpoint.

---

## Architecture Overview

```
┌───────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: OFFLINE DUAL-INDEX KNOWLEDGE BASE                                │
│                                                                           │
│ arXiv PDFs ──> PyMuPDF ──> Overlap Chunking ──┬──> FAISS (IndexFlatIP)    │
│ (Layout safe)                                 └──> BM25 (Keyword Index)   │
└───────────────────────────────────────────────▲───────────────────────────┘
                                                │ (Vector/Keyword Sync)
┌───────────────────────────────────────────────▼───────────────────────────┐
│ PHASE 2: ONLINE AGENTIC WORKFLOW (The While Loop)                         │
│                                                                           │
│  User Query (via FastAPI) ──> Chat Memory (Sliding Window)                │
│                                      │                                    │
│      ┌───────────────────────────────▼──────────────────────────────┐     │
│      │  LLM Brain: Groq Llama 3.1 8B (Intent Routing & Thought)     │<──┐ │
│      └─┬──────────────────┬──────────────────────┬────────────────┬─┘   │ │
│        │                  │                      │                │     │ │
│   ┌────▼────┐        ┌────▼─────┐           ┌────▼────┐           │     │ │
│   │ SEARCH  │        │CALCULATE │           │ CLARIFY │           │     │ │
│   │ CORPUS  │        │  (Math)  │           │ /REFUSE │           │     │ │
│   └────┬────┘        └────┬─────┘           └────┬────┘           │     │ │
│        │                  │                      │                │     │ │
│        │   ┌──────────────┴───────────────┐      │ (Handles:      │     │ │
│        │   │ ast.parse (Safe Evaluation)  │      │  Ambiguity,    │     │ │
│        │   └──────────────┬───────────────┘      │  Out-of-Domain)│     │ │
│        │                  │                      │                │     │ │
│        │   ┌──────────────┴───────────────┐      │                │     │ │
│        ├──>│ 1. FAISS + BM25 Retrieval    │      │                │     │ │
│        │   │ 2. Reciprocal Rank Fusion    │      │                │     │ │
│        │   │ 3. MS-MARCO Cross-Encoder    │      │                │     │ │
│        │   │ 4. Strict > 0.0 Threshold    │      │                │     │ │
│        │   └──────────────┬───────────────┘      │                │     │ │
│        │                  │                      │                │     │ │
│        └──────────────────┴──────> Update Context ──[Loop < max_loops]──┘ │
│                                                                           │
│ Final Action: Generate Context-Grounded Answer (Or "I don't know") <──────┘
└───────────────────────────────────────────────────────────────────────────┘
```

* The Brain: A native Python while loop running a Llama 3.1 8B model via the Groq SDK.

* The Memory: A session-scoped sliding window that maintains conversational state.

* The Tools: The LLM routes between an AST-based Safe Calculator, an Ambiguity Clarifier, and an Advanced RAG Retriever.

* The Retriever: A multi-stage engine combining Semantic Search (FAISS) + Keyword Search (BM25), merged via Reciprocal Rank Fusion (RRF), and filtered by a Cross-Encoder Reranker.

---

## Folder structure

```text
agentic-rag/
│
├── backend/                            #  The core ML & API microservice
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                     # FastAPI application & API routes
│   │   ├── agent.py                    # The while-loop state machine & routing
│   │   ├── retriever.py                # FAISS loading, hybrid search, reranking
│   │   ├── tools.py                    # External API logic (search, calc, etc.)
│   │   └── memory.py                   # Sliding window & semantic memory states
│   │
│   ├── data_pipeline/                  # One-time offline execution scripts
│   │   ├── ingest.py                   # Downloads/reads arXiv PDFs
│   │   └── build_index.py              # Chunks -> Embeds -> Saves FAISS index
│   │
│   ├── config.py                       # Centralized hyperparameters (Chunk size, Top-K, LLM models)
│   ├── requirements.txt
│   └── Dockerfile                      # Backend container setup
|
├── utils/                              # Centralized utilities
│   ├── __init__.py
│   ├── logger.py                       # Configures terminal + file logging
│   └── exceptions.py                   # Custom error classes (AgentError, RAGError)
|
├── frontend/                           #  The UI microservice
│   ├── app.py                          # Streamlit application(not yet!!)
│   ├── requirements.txt
│   └── Dockerfile                      # Frontend container setup
│
├── data/                               #  Ignored by .Git
│   ├── raw_pdfs/                       # Downloaded arXiv papers (50)
│   └── vector_store/                   # The saved .faiss and .pkl index files
|
├── docker-compose.yml                  # One-click local deployment 
├── README.md                          
└── logs/
    └── agent.log                       # proper log files stored

```
---

## Engineering Decisions Log

This section breaks down why I built the system this way, prioritizing control and observability over framework magic.

* **Agent Framework (Raw Python vs. LangChain):** I intentionally bypassed heavy abstractions like LangChain or LangGraph. I built the state machine using a native while loop and Groq's tool-calling API. This gave me 100% observability into the execution state and allowed me to implement a hard max_loops circuit breaker to prevent infinite, expensive LLM recursion.

* **PDF Extraction (PyMuPDF):** I chose PyMuPDF over PyPDF2 because arXiv papers have dense two-column layouts and complex math. PyMuPDF is significantly better at preserving spatial layouts and whitespace, ensuring my text chunks weren't scrambled.

* **Vector Mathematics (FAISS IndexFlatIP):** Instead of using standard L2 distance, I L2-normalized my embeddings before insertion and used FAISS IndexFlatIP (Inner Product). Mathematically, an inner product of normalized vectors yields exact Cosine Similarity, which is the gold standard for measuring semantic text distance, regardless of document length.

* **Retrieval Engine (Depth > Breadth):** I didn't want to just return top-K vectors. I implemented Hybrid Search (Semantic + BM25) to catch both contextual meaning and exact acronyms. However, the most critical addition was the MS-MARCO Cross-Encoder Reranker.

* Ablation Note: Without the reranker, FAISS would occasionally return chunks that matched keywords but lacked context, confusing the LLM. By adding the Cross-Encoder with a strict 0.0 relevance threshold, the system actively drops weak chunks. If no chunks pass, it returns an empty array, forcing the LLM to admit it doesn't know rather than hallucinating.

* **Memory Design:** I implemented a Conversational Memory (sliding window of the last N turns) because resolving pronouns (e.g., "What did that paper say?") is critical for natural RAG interactions.

---
## Handling Failure Modes

The system was heavily tested against edge cases. Here is how it reacts:

* **The corpus doesn't contain the answer:** The Cross-Encoder assigns negative scores to irrelevant chunks. The 0.0 threshold blocks them, and the Retriever returns an empty context to the LLM. The system prompt strictly forces the Agent to reply, "The corpus does not contain this information," preventing hallucination.

* **The user asks something ambiguous:** (e.g., "Summarize the paper"). The LLM recognizes the missing entity, bypasses the retrieval tool to save compute, and asks the user, "Which specific paper are you referring to?"

* **The user asks something outside the domain:** (e.g., "Who won the World Cup?"). The system prompt dictates strict domain boundaries. The agent will refuse to call search tools and politely state it only handles AI research and math.

* **The retrieved context contradicts itself:** The system prompt explicitly instructs the LLM that if multiple retrieved papers offer conflicting methodologies or results, it must highlight the contradiction to the user rather than forcing a single "truth."

---
## Known Limitations & Future Work

If I had another week to work on this, here is exactly what I would improve:

* **Persistent Memory Migration:** Currently, the sliding window memory is stored in RAM (memory.py). It works perfectly for a single session, but it is volatile. I would migrate this state to a lightweight Redis store to allow cross-session memory and scale across multiple API workers.

* **LLM Query Rewriter:** Right now, the sliding window provides context, but the LLM still has to generate the search query. I would add a small, fast pre-processing LLM step to explicitly rewrite pronouns based on history (e.g., translating "What are its drawbacks?" to "DINORANKCLIP drawbacks") before hitting the Vector database.

* **Semantic Chunking:** I used a fixed token-size sliding window for chunking. While standard, it's a blunt instrument for scientific PDFs and risks cutting mathematical proofs in half. I would implement a layout-aware parser to chunk documents by their actual structural headers (Abstract, Methodology, Conclusion) to preserve perfect semantic boundaries.

---
## Demo Video

[🔗 **Watch the Architecture & Live Demo Here**](https://www.youtube.com/watch?v=mV9ksCoA5Xs)

##### *(Note: The video runs slightly over the 8-minute mark at 9:02 to ensure I fully demonstrated the architeccture, state machine logic, live tool execution, and configs and tradeoffs).*
---