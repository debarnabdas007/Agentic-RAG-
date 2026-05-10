from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from backend.app.agent import SkycladAgent
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


app = FastAPI(
    title="Agentic RAG API",
    description="An autonomous agent that decides when to search technical documents, use a calculator, or ask clarifying questions.",
    version="1.0.0"
)

# Standard CORS configuration so your eventual Streamlit frontend can talk to it safely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the Brain globally so memory persists across API requests in this session
logger.info("Booting up the Agent...")
agent = SkycladAgent()

# Data Models for strict JSON validation 
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

class RetrievalRequest(BaseModel):
    query: str



#  Endpoints :-
@app.get("/health")
def health_check():
    """Simple ping to verify the server is alive."""
    return {"status": "healthy", "agent": "online"}

@app.post("/test-retrieval")
def test_retriever_endpoint(request: RetrievalRequest):
    """Debug endpoint to test Hybrid Search and Reranking directly."""
    try:
        logger.info(f"--- Testing Retriever for query: {request.query} ---")
        # Bypass the LLM entirely and just run the search engine
        results = agent.retriever.retrieve_and_rerank(request.query)
        return {
            "status": "success",
            "chunks_returned": len(results),
            "results": results
        }
    except Exception as e:
        logger.error(f"Retriever Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """The main interface to talk to the agent"""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    
    try:
        logger.info("--- New API Request ---")
        reply = agent.chat(request.message)
        return {"reply": reply}
    except Exception as e:
        logger.error(f"Critical API Error: {str(e)}")
        # We catch the error but don't leak the exact stack trace to the client
        raise HTTPException(status_code=500, detail="Internal server error. Check agent logs.")
    



'''

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


'''