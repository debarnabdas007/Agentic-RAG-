                    ┌────────────────────┐
                    │   FastAPI Server   │
                    └─────────┬──────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼

┌────────────────┐  ┌────────────────┐  ┌────────────────┐
│  Middleware    │  │ Global Agent   │  │ Pydantic Models│
│ (CORS Setup)   │  │ SkycladAgent() │  │ Request/Reply  │
└────────┬───────┘  └────────┬───────┘  └────────────────┘
         │                   │
         ▼                   ▼

 ┌────────────────────────────────────────────┐
 │                API Endpoints               │
 └────────────────────────────────────────────┘
                        │
        ┌───────────────┼────────────────┐
        │               │                │
        ▼               ▼                ▼

┌────────────────┐ ┌────────────────┐ ┌────────────────┐
│ /health        │ │ /test-retrieval│ │ /chat          │
│ Health Check   │ │ Debug Retriever│ │ Main Agent API │
└──────┬─────────┘ └──────┬─────────┘ └──────┬─────────┘
       │                  │                  │
       ▼                  ▼                  ▼

 return healthy     agent.retriever     agent.chat()
                     .retrieve_and_      Full Agentic
                     rerank()            Loop

                                             │
                                             ▼

                              ┌────────────────────────┐
                              │   LLM + Tools + RAG    │
                              │   + Memory Execution   │
                              └────────────────────────┘