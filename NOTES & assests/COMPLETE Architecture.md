=============================================================================
                    SKYCLAD AGENTic RAG ARCHITECTURE: FULL SYSTEM
=============================================================================

#############################################################################
                     PHASE I: OFFLINE DATA INGESTION PIPELINE
                     (Knowledge Base Construction - `ingest.py`, `build_index.py`)
#############################################################################

            Corpus: arXiv cs.AI PDFs
            (Last 90 days, ~50 papers)
                  │
                  ▼
┌─────────────────────────────────────────┐
│ 1. Data Ingestion & Extraction (PyMuPDF)│ Fetch raw PDFs and extract text,
│ Fetch PDFs → Extract text/page/source   │ preserving layout where possible.
└─────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│ 2. Text Normalization                   │ Clean extracted text (remove artifacts, 
│ Clean text → Canonicalize formatting    │ extra whitespace).
└─────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│ 3. Text Chunking (Sliding Window)       │ Split full text into overlapping chunks
│ e.g., 512 tokens with 50-token overlap  │ (e.g., recursive character splitter).
└─────────────────────────────────────────|
                  │
                  ▼
   ┌──────────────┴──────────────┐
   │PARALLEL INDEXING PIPELINE   │
   ▼                             ▼
┌──────────────────────┐      ┌──────────────────────┐
│ 4A. Semantic Indexing│      │ 4B. Keyword Indexing │
│ Chunks → Embeddings  │      │ Chunks → Tokenize    │ Generate dense embeddings
│ (SentenceTransformer)│      │ → Build BM25 Corpus  │ and build the BM25
└──────────────────────┘      └──────────────────────┘ corpus for keyword search.
          │                             │
          ▼                             ▼
┌─────────────────────────────────────────┐
│ 5. FAISS Index & Metadata Build         │
│ Store Chunks + IDs mapped to Metadata   │ Save the vector index, BM25 corpus,
│ → Save FAISS/pkl Knowledge Base         │ and serialized metadata store.
└─────────────────────────────────────────┘

                  │
                  │ (Save `vector_store/` and `metadata.pkl`)
                  │
                  ▼

#############################################################################
                     PHASE II: ONLINE AGENTIC WORKFLOW
                     (The Reasoning & Generation Loop - `main.py`, `agent.py`)
#############################################################################

                  User
                  │
                  ▼ Query
┌─────────────────────────────────────────┐
│ 1. Chat Interface (FastAPI)             │
│ Receive Query → Initialize State        │
└─────────────────────────────────────────┘
                  │
                  ▼
═══════════════════════════════════════
        AGENT REASONING LOOP (STATE MACHINE)
═══════════════════════════════════════
                  │
                  │ User Query + History
                  ▼
┌─────────────────────────────────────────┐
│ 2. Chat History / Memory Recall (`memory.py`)│ Recall sliding window of last N turns
│ Fetch last N messages                   │ to establish immediate context.
└─────────────────────────────────────────┘
                  │
                  ▼ Combined Context
┌─────────────────────────────────────────┐
│ 3. Agent Reasoning (Llama 3.1 8B LLM 1)  │ Read System Prompt, History, User
│ [Thought/Decision Step]                 │ Query. Identify user intent.
│ Decide on: SEARCH, CALCULATE, or REFUSE │
└─────────────────────────────────────────┘
                  │
                  ▼ Tool Name + Arguments
                  │
   ┌──────────────┴──────────────┬─────────────────────────--------------───┐
   │      TOOL EXECUTION         ▼                                          ▼
   ▼                             ┌────────────────────────────┐         ┌────────────────────────────-----------┐
┌────────────────────────────┐   │    AGENT REFRESER TOOL     │         │    AGENT CLARIFIER TOOL               │
│    CALCULATOR TOOL         │   │ (FAILURE MODE HANDLED: MISSING)      │ (FAILURE MODE HANDLED: AMBIGUITY)     │
│ ast.parse -> Safe Eval     │   │ Refuse Out-of-Domain Prompt│         │ Inform User of Ambiguity              │
└────────────────────────────┘   └────────────────────────────┘         │ → Loop Ends                           │
                                                                        └────────────────────────────-----------┘
                  │
                  ▼ Search Query
┌─────────────────────────────────────────┐
│ 4. ADVANCED RETRIEVER TOOL              │
│ (`retriever.py`)                        │ This tool executes the specific 7-step
│ Uses FAISS + BM25 Knowledge Bases       │ Parallel Retrieval & Reranking logic
│ [The USER’S DETAILED DIAGRAM goes here]  │ you provided earlier.
│ Returns: Top-N Highly Relevant Chunks   │
└─────────────────────────────────────────┘
                  │
                  ▼ Tool Outputs
┌─────────────────────────────────────────┐
│ 5. CONTEXT UPDATE & LOOP BACK           │ Incorporate successful tool outputs
│ Append results to history → Loop Back    │ into the chat context and restart the
│ to step 3 for final thought             │ reasoning loop to synthesize answer.
└─────────────────────────────────────────┘
                  │
                  │ [Final Thought: synthesise answer]
                  │
                  ▼ Chunks + Results + Memory
┌─────────────────────────────────────────┐
│ 6. Final Answer Generation (LLM 2)      │ Synthesize a coherent, final response
│ Read context and results → Synthesize   │ derived only from the validated context.
└─────────────────────────────────────────┘
                  │
                  ▼ Final Answer
┌─────────────────────────────────────────┐
│ 7. MEMORY UPDATE (`memory.py`)          │ Save user prompt and final agent
│ Append [User Prompt, Agent Answer] to   │ response to the conversational state
│ Conversational History (Memory)         │ for future turns.
└─────────────────────────────────────────┘
                  │
                  ▼ Answer
                  User