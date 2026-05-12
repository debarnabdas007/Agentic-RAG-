from __future__ import annotations

import threading
import uuid
from collections import OrderedDict

from backend.app.agent import SkycladAgent
from backend.app.retriever import AdvancedRetriever
from backend.config import settings
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


class SessionRegistry:
    """
    One SkycladAgent per session_id so concurrent users do not share MemoryManager state.
    All sessions share the same AdvancedRetriever (read-heavy index in RAM).
    """

    def __init__(self, retriever: AdvancedRetriever):
        self.retriever = retriever
        self._lock = threading.Lock()
        self._sessions: OrderedDict[str, SkycladAgent] = OrderedDict()
        self._max_sessions = settings.SESSION_MAX

    def resolve_session_id(self, session_id: str | None) -> str:
        sid = (session_id or "").strip()
        if not sid:
            return str(uuid.uuid4())
        return sid

    def get_agent(self, session_id: str) -> SkycladAgent:
        with self._lock:
            if session_id in self._sessions:
                self._sessions.move_to_end(session_id)
                return self._sessions[session_id]

            while len(self._sessions) >= self._max_sessions:
                evicted, _ = self._sessions.popitem(last=False)
                logger.info("Evicted idle session to cap memory: %s", evicted)

            agent = SkycladAgent(retriever=self.retriever)
            self._sessions[session_id] = agent
            logger.info("Created new agent session: %s", session_id)
            return agent

    def clear_session(self, session_id: str) -> bool:
        with self._lock:
            agent = self._sessions.pop(session_id, None)
        if agent is None:
            return False
        agent.memory.clear()
        return True
