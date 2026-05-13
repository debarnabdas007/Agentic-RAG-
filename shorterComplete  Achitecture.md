=============================================================================
              AGENTIC RAG: STATE MACHINE ARCHITECTURE
=============================================================================

┌───────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: OFFLINE DUAL-INDEX KNOWLEDGE BASE                                │
│                                                                           │
│ arXiv PDFs ──> PyMuPDF ──> Overlap Chunking ──┬─> Embed (MiniLM) ──> FAISS│
│ (Layout safe)                                 └─> Tokenize ────────> BM25 │  
└───────────────────────────────────────────────▲───────────────────────────┘
                                                │ (Vector/Keyword Sync)
┌───────────────────────────────────────────────▼───────────────────────────┐
│ PHASE 2: ONLINE AGENTIC WORKFLOW (The While Loop)                         │
│                                                                           │
│  User Query + session_id (FastAPI) ──> Session Registry (LRU Cache)       │
│                                                │                          │
│                                 [Retrieves/Creates Per-User Agent]        │
│                                                │                          │
│      ┌─────────────────────────────────────────▼────────────────────┐     │
│      │  Isolated Chat Memory (Sliding Window + Semantic Facts)      │     │
│      └───────────────────────────────┬──────────────────────────────┘     │
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