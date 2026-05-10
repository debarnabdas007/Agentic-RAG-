=============================================================================
                 SKYCLAD AGENTIC RAG SYSTEM ARCHITECTURE
=============================================================================

#############################################################################
                    PHASE I — OFFLINE KNOWLEDGE PIPELINE
                   (`ingest.py` + `build_index.py`)
#############################################################################

        arXiv cs.AI PDFs
                │
                ▼
┌─────────────────────────────────────────────┐
│ 1. PDF INGESTION & TEXT EXTRACTION          │
│    - Load PDFs using PyMuPDF (fitz)         │
│    - Extract raw page text                  │
│    - Preserve source + page metadata        │
└─────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────┐
│ 2. TEXT CLEANING / NORMALIZATION            │
│    - Remove extra spaces/newlines           │
│    - Standardize formatting                 │
└─────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────┐
│ 3. SLIDING WINDOW CHUNKING                  │
│    - Split into overlapping chunks          │
│    - Preserve semantic continuity           │
│    - Example: 512 chars + overlap           │
└─────────────────────────────────────────────┘
                │
                ▼
═══════════════════════════════════════════════════════════════
                  PARALLEL INDEX CONSTRUCTION
═══════════════════════════════════════════════════════════════
                │                                │
                ▼                                ▼

┌────────────────────────────-──┐   ┌───────-───────────────────────┐
│ 4A. SEMANTIC INDEXING         │   │ 4B. KEYWORD INDEXING          │
│                               │   │                               │
│ SentenceTransformer           │   │ Regex Tokenization            │
│ ↓                             │   │ ↓                             │
│ Dense Vector Embeddings       │   │ BM25 Corpus Creation          │
│ ↓                             │   │                               │
│ Cosine Similarity Space       │   │ Sparse Keyword Search Space   │
└────────────────────────────-──┘   └─────────────────────────────-─┘
                │                                │
                └──────────────┬─────────────────┘
                               ▼

┌─────────────────────────────────────────────┐
│ 5. VECTOR DATABASE PERSISTENCE              │
│                                             │
│ Save:                                       │
│ - FAISS Vector Index  → index.faiss         │
│ - Chunk Metadata     → metadata.pkl         │
│                                             │
│ Metadata stores:                            │
│ - source filename                           │
│ - page number                               │
│ - original chunk text                       │
└─────────────────────────────────────────────┘

                               │
                               ▼

#############################################################################
                   PHASE II — ONLINE AGENTIC WORKFLOW
                     (`main.py` + `agent.py`)
#############################################################################

                            USER
                              │
                              ▼
┌─────────────────────────────────────────────┐
│ 1. FASTAPI API LAYER                        │
│                                             │
│ Endpoints:                                  │
│ - /chat                                     │
│ - /test-retrieval                           │
│ - /health                                   │
└─────────────────────────────────────────────┘
                              │
                              ▼

═══════════════════════════════════════════════════════════════
                     AGENT REASONING LOOP
═══════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────┐
│ 2. MEMORY MANAGER (`memory.py`)             │
│                                             │
│ Short-Term Memory:                          │
│ - Sliding conversation window               │
│                                             │
│ Long-Term Semantic Memory:                  │
│ - User preferences / learned facts          │
└─────────────────────────────────────────────┘
                              │
                              ▼

┌─────────────────────────────────────────────┐
│ 3. LLM REASONING ENGINE                     │
│    (Groq Llama 3.1 8B)                      │
│                                             │
│ Reads:                                      │
│ - System Prompt                             │
│ - User Query                                │
│ - Conversation History                      │
│ - Semantic Memory                           │
│                                             │
│ Decides:                                    │
│ - Retrieve documents                        │
│ - Use calculator                            │
│ - Ask clarification                         │
│ - Refuse out-of-domain query                │
└─────────────────────────────────────────────┘
                              │
                              ▼

═══════════════════════════════════════════════════════════════
                           TOOL CALLING
═══════════════════════════════════════════════════════════════

          ┌──────────────────┴──────────────────┐
          ▼                                     ▼

┌──────────────────────────────┐   ┌──────────────────────────────┐
│ CALCULATOR TOOL               │   │ ADVANCED RETRIEVER TOOL      │
│                               │   │ (`retriever.py`)             │
│ Safe AST-based evaluation     │   │                               │
│ No unsafe eval()              │   │ Hybrid Retrieval Pipeline     │
└──────────────────────────────┘   └──────────────────────────────┘
                                                  │
                                                  ▼

#############################################################################
                     ADVANCED RETRIEVAL PIPELINE
#############################################################################

        User Query
              │
              ▼

┌─────────────────────────────────────────────┐
│ 1. QUERY EMBEDDING (BI-ENCODER)             │
│ SentenceTransformer converts query → vector │
└─────────────────────────────────────────────┘
              │
              ▼

═══════════════════════════════════════════════════════════════
                    PARALLEL HYBRID RETRIEVAL
═══════════════════════════════════════════════════════════════

              │                                  │
              ▼                                  ▼

┌──────────────────────────────┐   ┌──────────────────────────────┐
│ 2A. SEMANTIC SEARCH           │   │ 2B. BM25 KEYWORD SEARCH      │
│                               │   │                               │
│ FAISS Cosine Similarity       │   │ Exact keyword relevance       │
│ Vector vs Vector              │   │ Sparse lexical retrieval      │
└──────────────────────────────┘   └──────────────────────────────┘
              │                                  │
              └──────────────┬───────────────────┘
                             ▼

┌─────────────────────────────────────────────┐
│ 3. RECIPROCAL RANK FUSION (RRF)             │
│                                             │
│ Merge BM25 + Semantic rankings              │
│ using rank-based score fusion               │
└─────────────────────────────────────────────┘
                             │
                             ▼

┌─────────────────────────────────────────────┐
│ 4. CROSS-ENCODER RERANKING                  │
│                                             │
│ Query Text + Chunk Text                     │
│ ↓                                           │
│ Deep relevance scoring                      │
│                                             │
│ Much slower but highly accurate             │
└─────────────────────────────────────────────┘
                             │
                             ▼

┌─────────────────────────────────────────────┐
│ 5. THRESHOLD FILTERING                      │
│                                             │
│ Remove low-confidence chunks                │
│ Foundation of refusal logic                 │
└─────────────────────────────────────────────┘
                             │
                             ▼

┌─────────────────────────────────────────────┐
│ 6. TOP-K FINAL CONTEXT CHUNKS               │
│ Returned to the Agent                       │
└─────────────────────────────────────────────┘

                             │
                             ▼

┌─────────────────────────────────────────────┐
│ 4. CONTEXT INJECTION INTO MEMORY            │
│                                             │
│ Tool outputs appended into conversation     │
│ history for reasoning continuity            │
└─────────────────────────────────────────────┘
                             │
                             ▼

┌─────────────────────────────────────────────┐
│ 5. FINAL RESPONSE GENERATION                │
│                                             │
│ LLM synthesizes final grounded answer       │
│ using retrieved context + memory            │
└─────────────────────────────────────────────┘
                             │
                             ▼

┌─────────────────────────────────────────────┐
│ 6. MEMORY UPDATE                            │
│                                             │
│ Save latest interaction into memory         │
└─────────────────────────────────────────────┘
                             │
                             ▼

                            USER