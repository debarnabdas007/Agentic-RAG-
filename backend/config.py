import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    # Application Info
    APP_NAME: str = "Skyclad Agentic RAG"
    APP_VERSION: str = "1.0.0"

    # API Keys
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

    # LLM Settings
    AGENT_MODEL: str = "llama-3.1-8b-instant"
    TEMPERATURE: float = 0.0  # Keeping  0 for deterministic agent routing
    MAX_TOKENS: int = 1024

    # RAG Hyperparameters
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # Local SentenceTransformer downloaded !!
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    RETRIEVER_TOP_K: int = 15      # Top 15 relevant chunks after Hybrid search
    RERANKER_TOP_K: int = 5        # After Reranking, Top 5 Final chunks passed to the LLM
    # Cross-encoder scores below this are dropped (empty retrieval --> honest "not in corpus").
    RERANK_THRESHOLD: float = 0.0

    # API session eviction (in-memory; laster may use Redis for production scale-out)
    SESSION_MAX: int = 200

    # Calculator bounds (mitigate exponent / size DoS while keeping AST evaluation)
    CALC_MAX_AST_NODES: int = 64
    CALC_MAX_ABS_EXPONENT: int = 10_000
    CALC_MAX_RESULT_LOG10: float = 300.0  # reject if estimated log10(|result|) exceeds this

    # Paths (relative to repository root — see `backend/paths.py`)
    DATA_DIR: str = "data"
    RAW_PDF_DIR: str = "data/raw_pdfs"
    VECTOR_STORE_DIR: str = "data/vector_store"
    LOG_FILE_PATH: str = "logs/agent.log"

# Instantiation
settings = Settings()



'''
# BaseSettings:

> validates env variables
> handles typing
> automatically loads configs safely
> supports .env

'''