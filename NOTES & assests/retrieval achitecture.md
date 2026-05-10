My Retrieval Architecture :-



┌─────────────────────────────────────────────────────────────────────┐
│                         USER QUERY / API CALL                       │
│                 FastAPI Endpoint  →  /chat or /ask                  │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           AGENT LOOP                                │
│                     backend/app/agent.py                            │
│                                                                     │
│  Responsibilities:                                                  │
│  • Maintains agent state                                            │
│  • Reads conversation memory                                        │
│  • Decides next action                                              │
│  • Tool routing                                                     │
│  • Refusal / clarification logic                                    │
│  • Final answer generation                                          │
└─────────────────────────────────────────────────────────────────────┘
                                  │
             ┌────────────────────┼────────────────────┐
             │                    │                    │
             ▼                    ▼                    ▼

   ┌────────────────┐   ┌────────────────┐   ┌────────────────┐
   │ RETRIEVE TOOL  │   │ CALCULATOR     │   │ REFUSE / ASK   │
   │                │   │ TOOL           │   │ CLARIFICATION  │
   └────────────────┘   └────────────────┘   └────────────────┘
             │
             ▼

═══════════════════════════════════════════════════════════════════════
                    ADVANCED RETRIEVER PIPELINE
═══════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────┐
│                     AdvancedRetriever Class                         │
│                    backend/app/retriever.py                         │
└─────────────────────────────────────────────────────────────────────┘

                             INITIALIZATION
═══════════════════════════════════════════════════════════════════════

                ┌────────────────────────────┐
                │ SentenceTransformer Model  │
                │ all-MiniLM-L6-v2           │
                │                            │
                │ Used for query embeddings  │
                └────────────────────────────┘

                ┌────────────────────────────┐
                │ CrossEncoder Reranker      │
                │ ms-marco-MiniLM-L-6-v2     │
                │                            │
                │ Deep semantic reranking    │
                └────────────────────────────┘

                ┌────────────────────────────┐
                │ FAISS Index                │
                │ index.faiss                │
                │                            │
                │ Vector similarity search   │
                └────────────────────────────┘

                ┌────────────────────────────┐
                │ Metadata Store             │
                │ metadata.pkl               │
                │                            │
                │ Chunk text + source/page   │
                └────────────────────────────┘

                ┌────────────────────────────┐
                │ BM25 Corpus                │
                │ rank_bm25                  │
                │                            │
                │ Keyword retrieval engine   │
                └────────────────────────────┘


═══════════════════════════════════════════════════════════════════════
                         RETRIEVAL FLOW
═══════════════════════════════════════════════════════════════════════

USER QUERY
   │
   ▼

┌─────────────────────────────────────────────────────────────────────┐
│                    EMPTY QUERY GUARD                                │
│        if not query or not query.strip(): return []                 │
└─────────────────────────────────────────────────────────────────────┘
   │
   ▼

═══════════════════════════════════════════════════════════════════════
                      PARALLEL RETRIEVAL
═══════════════════════════════════════════════════════════════════════

        ┌─────────────────────────┐
        │     SEMANTIC SEARCH     │
        └─────────────────────────┘
                    │
                    ▼
      Query → Embedding Generation
                    │
                    ▼
      normalize_embeddings=True
                    │
                    ▼
      Unit Vector Conversion
                    │
                    ▼
      FAISS IndexFlatIP Search
                    │
                    ▼
      Top-K Similar Chunks
                    │
                    ▼
      {chunk_index : rank}

───────────────────────────────────────────────────────────────────────

        ┌─────────────────────────┐
        │      BM25 SEARCH        │
        └─────────────────────────┘
                    │
                    ▼
      Regex Tokenization
                    │
                    ▼
      Keyword Matching
                    │
                    ▼
      BM25 Scoring
                    │
                    ▼
      Top-K Ranked Chunks
                    │
                    ▼
      {chunk_index : rank}

═══════════════════════════════════════════════════════════════════════
                    HYBRID SEARCH FUSION
═══════════════════════════════════════════════════════════════════════

        Semantic Results
                +
        BM25 Results
                │
                ▼

┌─────────────────────────────────────────────────────────────────────┐
│                 Reciprocal Rank Fusion (RRF)                        │
│                                                                     │
│     score += 1 / (k_constant + rank_position)                       │
│                                                                     │
│     Uses ranking positions instead of raw scores                    │
│                                                                     │
│     Robust merging between:                                         │
│     • cosine similarity scores                                      │
│     • BM25 keyword scores                                           │
└─────────────────────────────────────────────────────────────────────┘

                │
                ▼

      Top Hybrid Candidates
                │
                ▼

═══════════════════════════════════════════════════════════════════════
                          RERANKING
═══════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────┐
│                    Cross-Encoder Reranker                           │
│                                                                     │
│  Input:                                                             │
│     [query, chunk_text] pairs                                       │
│                                                                     │
│  Example:                                                           │
│     ["How does reranking work?", "Cross-encoders improve..."]       │
│                                                                     │
│  Output:                                                            │
│     Deep relevance score                                            │
└─────────────────────────────────────────────────────────────────────┘

                │
                ▼

      Attach rerank_score
                │
                ▼

      Sort descending
                │
                ▼

═══════════════════════════════════════════════════════════════════════
                      THRESHOLD FILTERING
═══════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────┐
│                  RERANK_THRESHOLD FILTER                            │
│                                                                     │
│  Keep only chunks where:                                            │
│      rerank_score > threshold                                       │
│                                                                     │
│  Purpose:                                                           │
│  • Prevent hallucinations                                           │
│  • Enable refusal logic                                             │
│  • Remove weak retrievals                                           │
│  • Improve answer quality                                           │
└─────────────────────────────────────────────────────────────────────┘

                │
                ▼

      Final Top-N Chunks
                │
                ▼

═══════════════════════════════════════════════════════════════════════
                       FINAL OUTPUT
═══════════════════════════════════════════════════════════════════════

[
  {
    "source": "...pdf",
    "page": 6,
    "text": "...",
    "rerank_score": 2.14
  },
  ...
]

                │
                ▼

┌─────────────────────────────────────────────────────────────────────┐
│                  PASSED TO THE LLM AGENT                           │
│                                                                     │
│  The agent now decides:                                             │
│  • answer                                                           │
│  • refuse                                                           │
│  • ask clarification                                                │
│  • use another tool                                                 │
└─────────────────────────────────────────────────────────────────────┘



═══════════════════════════════════════════════════════════════════════
                    OFFLINE INDEXING PIPELINE
═══════════════════════════════════════════════════════════════════════

                    arXiv PDFs
                         │
                         ▼

                 Text Extraction
                    (PyMuPDF)
                         │
                         ▼

                    Cleaning
                         │
                         ▼

             Sliding Window Chunking
                         │
                         ▼

               Chunk Overlapping
                         │
                         ▼

              Embedding Generation
            (SentenceTransformer)
                         │
                         ▼

         normalize_embeddings=True
                         │
                         ▼

                 Vector Embeddings
                         │
             ┌───────────┴───────────┐
             ▼                       ▼

      index.faiss              metadata.pkl
   (vector storage)        (text + source info)

             │                       │
             └───────────┬───────────┘
                         ▼

                Runtime Retrieval












-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------





## In detail retriver.py :-


═══════════════════════════════════════════════════════════════════════
                    retriever.py ARCHITECTURE
═══════════════════════════════════════════════════════════════════════


                    ┌──────────────────────┐
                    │  User Query String   │
                    │ "How does RAG work?" │
                    └──────────┬───────────┘
                               │
                               ▼

┌─────────────────────────────────────────────────────────────────────┐
│                 retrieve_and_rerank(query)                          │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼

═══════════════════════════════════════════════════════════════════════
                        STAGE 0 : VALIDATION
═══════════════════════════════════════════════════════════════════════

                ┌──────────────────────────┐
                │ Empty Query Guard        │
                │                          │
                │ if not query.strip():    │
                │      return []           │
                └────────────┬─────────────┘
                             │
                             ▼

═══════════════════════════════════════════════════════════════════════
                 STAGE 1 : PARALLEL RETRIEVAL
═══════════════════════════════════════════════════════════════════════


        ┌─────────────────────────────────────────────┐
        │           _semantic_search()                │
        └─────────────────────────────────────────────┘

                         QUERY
                           │
                           ▼

              SentenceTransformer Encoder
                           │
                           ▼

             Query Embedding Vector
                           │
                           ▼

             normalize_embeddings=True
                           │
                           ▼

                   Unit Vector
                           │
                           ▼

             FAISS IndexFlatIP Search
                           │
                           ▼

              Top-K Similar Chunks
                           │
                           ▼

            distances, indices = search()
                           │
                           ▼

                  Convert to:
                 {chunk_id : rank}

───────────────────────────────────────────────────────────────────────

        ┌─────────────────────────────────────────────┐
        │            _keyword_search()                │
        └─────────────────────────────────────────────┘

                         QUERY
                           │
                           ▼

                  Regex Tokenization
                re.findall(r"\w+")
                           │
                           ▼

                   Tokenized Query
                           │
                           ▼

                BM25 Keyword Matching
                           │
                           ▼

             BM25 Score for Every Chunk
                           │
                           ▼

              np.argsort(scores)[::-1]
                           │
                           ▼

                 Top-K Ranked Chunks
                           │
                           ▼

                  Convert to:
                 {chunk_id : rank}


═══════════════════════════════════════════════════════════════════════
                  STAGE 2 : HYBRID FUSION (RRF)
═══════════════════════════════════════════════════════════════════════


            Semantic Results
          {5023:0, 1605:1...}

                      +
                      
            BM25 Results
          {1617:0, 608:1...}

                      │
                      ▼

┌─────────────────────────────────────────────────────────────────────┐
│                 Reciprocal Rank Fusion                              │
│                                                                     │
│ score += 1 / (k_constant + rank)                                    │
│                                                                     │
│ Example:                                                            │
│                                                                     │
│ Semantic rank = 1                                                   │
│ BM25 rank = 3                                                       │
│                                                                     │
│ Final score =                                                       │
│ 1/(60+1) + 1/(60+3)                                                 │
└─────────────────────────────────────────────────────────────────────┘

                      │
                      ▼

              Combined RRF Scores
                      │
                      ▼

         Sort by highest fusion score
                      │
                      ▼

             fused_indices[:TOP_K]

═══════════════════════════════════════════════════════════════════════
                STAGE 3 : METADATA RETRIEVAL
═══════════════════════════════════════════════════════════════════════


                 Fused Chunk IDs
               [5023, 1617, ...]

                         │
                         ▼

┌─────────────────────────────────────────────────────────────────────┐
│               metadata.pkl lookup                                   │
│                                                                     │
│ self.metadata[idx].copy()                                           │
└─────────────────────────────────────────────────────────────────────┘

                         │
                         ▼

                Candidate Documents

[
  {
    "source": "...pdf",
    "page": 6,
    "text": "Cross encoders improve..."
  }
]

═══════════════════════════════════════════════════════════════════════
                  STAGE 4 : CROSS-ENCODER RERANKING
═══════════════════════════════════════════════════════════════════════


                 Candidate Documents
                         │
                         ▼

             Build sentence pairs

[
  [query, chunk_text],
  [query, chunk_text],
  ...
]

                         │
                         ▼

                CrossEncoder.predict()
                         │
                         ▼

                Deep Relevance Scores

[
  2.14,
  0.87,
  -3.2
]

                         │
                         ▼

            Attach rerank_score to docs
                         │
                         ▼

             Sort descending by score


═══════════════════════════════════════════════════════════════════════
                  STAGE 5 : THRESHOLD FILTERING
═══════════════════════════════════════════════════════════════════════


                  Sorted Candidates
                         │
                         ▼

┌─────────────────────────────────────────────────────────────────────┐
│             rerank_score > threshold                                │
│                                                                     │
│ Example threshold = 0.0                                             │
│                                                                     │
│ Keep:                                                               │
│   2.14                                                              │
│   0.87                                                              │
│                                                                     │
│ Remove:                                                             │
│  -3.2                                                               │
└─────────────────────────────────────────────────────────────────────┘

                         │
                         ▼

               Top Final Chunks
             [:settings.RERANKER_TOP_K]

═══════════════════════════════════════════════════════════════════════
                        FINAL OUTPUT
═══════════════════════════════════════════════════════════════════════

[
  {
    "source": "...pdf",
    "page": 6,
    "text": "...",
    "rerank_score": 2.14
  }
]

═══════════════════════════════════════════════════════════════════════
                      CLASS INITIALIZATION
═══════════════════════════════════════════════════════════════════════


                 AdvancedRetriever()
                          │
                          ▼

                 __init__() executes
                          │
          ┌───────────────┼────────────────┐
          ▼               ▼                ▼

 SentenceTransformer   CrossEncoder      FAISS
      Model              Reranker        Index

          ▼               ▼                ▼

         RAM             RAM              RAM

                          │
                          ▼

                   metadata.pkl
                          │
                          ▼

                     BM25 Corpus
                          │
                          ▼

               Retriever Ready Once
                          │
                          ▼

        Future Queries reuse same models
        (NO reloading every request)




