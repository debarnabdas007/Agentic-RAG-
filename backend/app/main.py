from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.app.retriever import AdvancedRetriever
from backend.app.session_registry import SessionRegistry
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load FAISS/BM25/reranker once; all chat sessions share this retriever."""
    logger.info("Loading shared retriever (one-time)...")
    retriever = AdvancedRetriever()
    app.state.retriever = retriever
    app.state.session_registry = SessionRegistry(retriever)
    yield


app = FastAPI(
    title="Agentic RAG API",
    description=(
        "Autonomous agent: search technical documents, calculator, clarify, or refuse. "
        "Pass session_id per client to isolate conversation memory."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str | None = Field(
        default=None,
        max_length=128,
        description="Client-owned id (e.g. UUID). Reuse across requests to keep memory isolated per user.",
    )


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    trace: dict | None = None


class RetrievalRequest(BaseModel):
    query: str


@app.get("/health")
def health_check():
    return {"status": "healthy", "agent": "online"}


@app.post("/test-retrieval")
def test_retriever_endpoint(request: RetrievalRequest, http_request: Request):
    try:
        logger.info("--- Testing Retriever for query: %s ---", request.query)
        retriever: AdvancedRetriever = http_request.app.state.retriever
        results = retriever.retrieve_and_rerank(request.query)
        return {
            "status": "success",
            "chunks_returned": len(results),
            "results": results,
        }
    except Exception as e:
        logger.error("Retriever Error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(
    request: ChatRequest,
    http_request: Request,
    debug: bool = Query(False, description="If true, include a lightweight tool/loop trace."),
):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    registry: SessionRegistry = http_request.app.state.session_registry
    sid = registry.resolve_session_id(request.session_id)
    agent = registry.get_agent(sid)

    try:
        logger.info("--- New API request session=%s ---", sid)
        if debug:
            reply, trace = agent.chat(request.message.strip(), return_trace=True)
            return ChatResponse(reply=reply, session_id=sid, trace=trace)
        reply = agent.chat(request.message.strip(), return_trace=False)
        return ChatResponse(reply=reply, session_id=sid, trace=None)
    except Exception as e:
        logger.error("Critical API Error: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail="Internal server error. Check agent logs.",
        ) from e


@app.delete("/session/{session_id}")
def delete_session(session_id: str, http_request: Request):
    registry: SessionRegistry = http_request.app.state.session_registry
    if registry.clear_session(session_id):
        return {"status": "ok", "session_id": session_id}
    raise HTTPException(status_code=404, detail="Unknown session_id")
