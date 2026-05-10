class AgenticRAGBaseException(Exception):
    """Base exception for the entire application."""
    pass

class DataIngestionError(AgenticRAGBaseException):
    """Raised when PDF downloading or chunking fails."""
    pass

class RetrievalError(AgenticRAGBaseException):
    """Raised when FAISS or the Reranker fails."""
    pass

class ToolExecutionError(AgenticRAGBaseException):
    """Raised when an external tool (like calculator or API) crashes."""
    pass

class LLMRoutingError(AgenticRAGBaseException):
    """Raised when the LLM outputs invalid JSON or hallucinations during the decision step."""
    pass