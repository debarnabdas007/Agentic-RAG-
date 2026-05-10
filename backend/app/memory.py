from typing import List, Dict
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)

class MemoryManager:
    def __init__(self, max_window_size: int = 5):
        """
        Initializes the memory system
        * max_window_size: The number of full interactions (user + assistant) to keep in immediate context
        """
        logger.info("Initializing Memory Manager...")
        self.max_window_size = max_window_size
        
        # Conversational Memory (Short-term)
        self.history: List[Dict[str, str]] = []
        
        # Semantic Memory (Long-term state)
        self.semantic_knowledge: set = set()


    def add_message(self, role: str, content: str):
        """Adds a message to the conversational history."""
        self.history.append({"role": role, "content": content})
        
        # Safe slicing: Keep only the most recent N messages
        limit = self.max_window_size * 2
        if len(self.history) > limit:
            self.history = self.history[-limit:]


    def get_conversational_window(self) -> List[Dict[str, str]]:
        """Returns the current sliding window of history."""
        return self.history



    def extract_semantic_facts(self, llm_response: str):
        """
        A lightweight mechanism to extract and store long-term facts.
        This can be made a separate LLM call also...
        But here, we will allow the main agent to append facts manually if it detects a long-term user preference
        """
        # We will design our agent to output a special JSON flag if it learns a fact
        # Example: {"fact_learned": "User prefers concise mathematical proofs."}
        pass # The actual extraction logic will be handled inside the agent loop

    def add_semantic_fact(self, fact: str):
        """Stores a distinct fact about the user or conversation"""
        if fact not in self.semantic_knowledge:
            logger.info(f"Learned new semantic fact: {fact}")
            self.semantic_knowledge.add(fact)

    def get_system_prompt_context(self) -> str:
        """
        Injects semantic memory into the system prompt..
        Even if the conversation window slides past the original message, these facts permanently alter the agent's behavior.
        """
        if not self.semantic_knowledge:
            return ""
            
        facts = "\n- ".join(self.semantic_knowledge)
        return f"\n\nCRITICAL LONG-TERM KNOWLEDGE:\n- {facts}\nAlways respect these facts in your answers."

    def clear(self):
        """Resets the memory for a new session"""
        logger.info("Clearing memory session...")
        self.history = []
        self.semantic_knowledge = set()








'''
                   ┌──────────────────┐
                   │   User Message   │
                   └────────┬─────────┘
                            │
                            ▼
                 ┌────────────────────┐
                 │   MemoryManager    │
                 └────────┬───────────┘
                          │
          ┌───────────────┴────────────────┐
          │                                │
          ▼                                ▼

┌──────────────────────┐      ┌────────────────────────┐
│ Short-Term Memory    │      │ Long-Term Memory       │
│ (Conversation Window)│      │ (Semantic Memory)      │
└──────────┬───────────┘      └──────────┬─────────────┘
           │                             │
           ▼                             ▼

 ┌────────────────────┐      ┌────────────────────────┐
 │ self.history       │      │ semantic_knowledge     │
 │ Stores recent msgs │      │ Stores user facts      │
 └─────────┬──────────┘      └──────────┬─────────────┘
           │                             │
           ▼                             ▼

 ┌────────────────────┐      ┌────────────────────────┐
 │ Sliding Window     │      │ Prompt Injection       │
 │ Keeps last N msgs  │      │ Adds facts to system   │
 └─────────┬──────────┘      │ prompt dynamically     │
           │                 └──────────┬─────────────┘
           ▼                            │
     ┌───────────────┐                  │
     │ Current Chat  │                  │
     │ Context       │                  │
     └──────┬────────┘                  │
            └──────────────┬────────────┘
                           ▼

                ┌────────────────────┐
                │   Sent to LLM      │
                │ (Agent + Memory)   │
                └────────────────────┘



'''        