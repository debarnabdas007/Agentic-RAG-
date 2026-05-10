import json
from groq import Groq
from backend.config import settings
from backend.utils.logger import setup_logger
from backend.app.tools import AGENT_TOOLS_SCHEMA, safe_calculate
from backend.app.retriever import AdvancedRetriever
from backend.app.memory import MemoryManager

logger = setup_logger(__name__)

class SkycladAgent:
    def __init__(self):
        logger.info("Initializing Skyclad Agent...")
        self.client = Groq(api_key=settings.GROQ_API_KEY)
        self.retriever = AdvancedRetriever()
        self.memory = MemoryManager(max_window_size=5)
        
        # A dictionary mapping the *JSON tool* names to our actual Python functions
        self.available_tools = {
            "retrieve_technical_documents": self._run_retriever,
            "calculate": safe_calculate
        }

    def _run_retriever(self, query: str) -> str:
        """Wrapper to format retrieved chunks into a clean string for the LLM."""
        results = self.retriever.retrieve_and_rerank(query)
        if not results:
            return "DATABASE SEARCH YIELDED NO RESULTS. The corpus does not contain information matching this query."
        
        # Format chunks into a readable context block
        context = []
        for i, doc in enumerate(results):
            context.append(f"[Source: {doc['source']}, Page: {doc['page']}]\n{doc['text']}")
        return "\n\n---\n\n".join(context)

    def _get_system_prompt(self) -> str:
        """Dynamically generates the system prompt, injecting semantic memory."""
        base_prompt = """You are a highly analytical AI engineering assistant built by Skyclad Ventures. 
Your goal is to answer technical questions over a corpus of recent cs.AI arXiv papers or perform calculations.

CRITICAL INSTRUCTIONS:
1. REPORT ALL FINDINGS: If the user asks a multi-part question, you MUST report the results of EVERY successful tool you called.
2. MISSING CONTEXT: If the document search returns no results, explicitly state that the corpus does not contain the information. 
3. AMBIGUITY & SEARCHING: If a query completely lacks specific nouns (e.g., "explain it"), ask for clarity. HOWEVER, if the user mentions ANY specific paper name (e.g., "DINORANKCLIP"), concept, or keyword, consider this SUFFICIENT context. DO NOT ask for authors or titles. IMMEDIATELY call the retrieval tool using that keyword.
4. OUTSIDE DOMAIN: If the user asks about non-technical topics, politely refuse.
5. CONTRADICTIONS: If retrieved documents contradict each other, point it out.

Always think step-by-step."""
        
        semantic_context = self.memory.get_system_prompt_context()
        return base_prompt + semantic_context

    def chat(self, user_input: str) -> str:
        """The main Agentic While-Loop."""
        logger.info(f"User Input: {user_input}")
        
        # Add user message to memory
        self.memory.add_message("user", user_input)
        
        # We cap the loop at 5 iterations to prevent infinite AI loops !!
        max_loops = 5  # Loop guard
        loop_count = 0
        
        while loop_count < max_loops:
            loop_count += 1
            logger.info(f"Agent Loop Iteration: {loop_count}")
            
            # Build the message array for Groq/LLM (System + History)
            messages = [{"role": "system", "content": self._get_system_prompt()}]
            messages.extend(self.memory.get_conversational_window())
            
            # Call the LLM
            response = self.client.chat.completions.create(
                model=settings.AGENT_MODEL,
                messages=messages,
                tools=AGENT_TOOLS_SCHEMA,
                tool_choice="auto", # Let LLM decide whether to use a tool or just talk..
                temperature=settings.TEMPERATURE,
                max_tokens=settings.MAX_TOKENS
            )
            
            response_message = response.choices[0].message
            
            # Check if the LLM wants to use a tool ?..
            tool_calls = response_message.tool_calls
            
            if tool_calls:
                logger.info(f"LLM decided to call {len(tool_calls)} tool(s).")
                
                # Append the LLM's tool request to memory so it remembers asking for it
                self.memory.history.append({
                    "role": "assistant",
                    "content": response_message.content, # Might be None
                    "tool_calls": [t.model_dump() for t in tool_calls]
                })
                
                # Execute each tool
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    logger.info(f"Executing tool: {function_name} with args: {function_args}")
                    
                    # Safety check: does the tool exist?
                    if function_name not in self.available_tools:
                        tool_result = f"Error: Tool {function_name} is not available."
                    else:
                        try:
                            # Actually run the python function
                            function_to_call = self.available_tools[function_name]
                            if function_name == "calculate":
                                tool_result = function_to_call(expression=function_args.get("expression"))
                            elif function_name == "retrieve_technical_documents":
                                tool_result = function_to_call(query=function_args.get("query"))
                        except Exception as e:
                            tool_result = f"Error executing tool: {str(e)}"
                    
                    # Append the tool's output back to memory so the LLM can read it
                    self.memory.history.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": str(tool_result),
                    })
                
                # Loop back to the top of the while-loop so the LLM can read the new tool data
                continue
                
            else:
                # No tools called.. The LLM is talking directly to the user
                final_answer = response_message.content
                logger.info("LLM provided final textual answer.")
                
                # Adding the final answer to memory and break the loop
                self.memory.add_message("assistant", final_answer)
                return final_answer
                
        # Fallback if it gets stuck in a loop
        error_msg = "Agent halted: Maximum reasoning steps reached without a final answer."
        logger.error(error_msg)
        return error_msg
    

    


"""

agent.py
│
├── Maintains conversation state
├── Injects memory
├── Calls LLM
├── Lets LLM choose tools
├── Executes tools
├── Stores tool outputs
├── Repeats reasoning loop
├── Prevents infinite loops
└── Returns final answer



"""


'''
                ┌─────────────────┐
                │ User Query      │
                └────────┬────────┘
                         ▼
              ┌────────────────────┐
              │   Agent Loop       │
              │ (Reasoning Brain)  │
              └────────┬───────────┘
                       ▼
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   Retrieval Tool   Calculator     Refusal /
                                    Clarify
        ▼
   Tool Results
        ▼
   Back Into Context
        ▼
   Final Response
    

'''