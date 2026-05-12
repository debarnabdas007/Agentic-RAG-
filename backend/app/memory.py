import re
from typing import List, Dict
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)

# Lightweight user-stated preferences (semantic memory), no extra LLM call.
_PREFERENCE_PATTERNS = [
    re.compile(r"(?i)^\s*(?:please\s+)?always\s+(.{3,240})\s*$"),
    re.compile(r"(?i)^\s*remember\s+(?:that\s+)?(.{3,240})\s*[.!]?\s*$"),
    re.compile(r"(?i)^\s*my\s+preference\s+is\s+(.{3,240})\s*[.!]?\s*$"),
]


class MemoryManager:
    def __init__(self, max_window_size: int = 5):
        """
        max_window_size: number of full user+assistant turns to retain in the sliding window.
        """
        logger.info("Initializing Memory Manager...")
        self.max_window_size = max_window_size

        # Conversational (short-term): raw chat for the LLM API
        self.history: List[Dict] = []

        # Semantic (long-term): distilled user preferences injected into the system prompt
        self.semantic_knowledge: set[str] = set()

    def add_message(self, role: str, content: str):
        """Adds a message to the conversational history."""
        self.history.append({"role": role, "content": content})

        limit = self.max_window_size * 2
        if len(self.history) > limit:
            self.history = self.history[-limit:]

    def get_conversational_window(self) -> List[Dict]:
        return self.history

    def ingest_user_stated_facts(self, user_text: str) -> None:
        """Extract simple preference phrases from the user turn (deterministic)."""
        text = user_text.strip()
        if not text:
            return
        for pat in _PREFERENCE_PATTERNS:
            m = pat.match(text)
            if m:
                fact = m.group(1).strip()
                if fact:
                    self.add_semantic_fact(f"User preference: {fact}")
                return

    def add_semantic_fact(self, fact: str):
        if fact not in self.semantic_knowledge:
            logger.info(f"Learned new semantic fact: {fact}")
            self.semantic_knowledge.add(fact)

    def get_system_prompt_context(self) -> str:
        if not self.semantic_knowledge:
            return ""
        facts = "\n- ".join(sorted(self.semantic_knowledge))
        return f"\n\nCRITICAL LONG-TERM KNOWLEDGE:\n- {facts}\nAlways respect these facts in your answers."

    def clear(self):
        logger.info("Clearing memory session...")
        self.history = []
        self.semantic_knowledge.clear()
