import json
from typing import Any

from groq import Groq

from backend.config import settings
from backend.utils.logger import setup_logger
from backend.app.tools import AGENT_TOOLS_SCHEMA, safe_calculate
from backend.app.retriever import AdvancedRetriever
from backend.app.memory import MemoryManager

logger = setup_logger(__name__)


class SkycladAgent:
    """Tool-calling agent with bounded loop and pluggable retriever (shared per process)."""

    def __init__(self, retriever: AdvancedRetriever | None = None):
        logger.info("Initializing Skyclad Agent...")
        self.client = Groq(api_key=settings.GROQ_API_KEY)
        self.retriever = retriever or AdvancedRetriever()
        self.memory = MemoryManager(max_window_size=5)
        # When False, retrieval uses RRF order only (for ablations / comparisons).
        self.use_reranker: bool = True

        self.available_tools = {
            "retrieve_technical_documents": self._run_retriever,
            "calculate": safe_calculate,
        }

    def _run_retriever(self, query: str) -> str:
        results = self.retriever.retrieve_and_rerank(
            query,
            use_reranker=self.use_reranker,
            apply_score_threshold=self.use_reranker,
        )
        if not results:
            return (
                "DATABASE SEARCH YIELDED NO RESULTS. The corpus does not contain "
                "information matching this query."
            )

        context = []
        for doc in results:
            context.append(f"[Source: {doc['source']}, Page: {doc['page']}]\n{doc['text']}")
        return "\n\n---\n\n".join(context)

    def _get_system_prompt(self) -> str:
        base_prompt = """You are a highly analytical AI engineering assistant built by Skyclad Ventures.
Your goal is to answer technical questions over a corpus of recent cs.AI arXiv papers or perform calculations.

CRITICAL INSTRUCTIONS:
1. REPORT ALL FINDINGS: If the user asks a multi-part question, you MUST report the results of EVERY successful tool you called.
2. MISSING CONTEXT: If the document search returns no results, explicitly state that the corpus does not contain the information.
3. AMBIGUITY & SEARCHING: If a query completely lacks specific nouns (e.g., "explain it"), ask for clarity. HOWEVER, if the user mentions ANY specific paper name, concept, or keyword, consider this SUFFICIENT context. DO NOT ask for authors or titles. IMMEDIATELY call the retrieval tool using that keyword.
4. OUTSIDE DOMAIN: If the user asks about non-technical topics, politely refuse.
5. CONTRADICTIONS: If retrieved documents contradict each other, point it out.

Always think step-by-step."""

        semantic_context = self.memory.get_system_prompt_context()
        return base_prompt + semantic_context

    def chat(self, user_input: str, *, return_trace: bool = False) -> str | tuple[str, dict[str, Any]]:
        logger.info(f"User Input: {user_input}")
        trace: dict[str, Any] = {"loops": 0, "tool_calls": []}

        self.memory.ingest_user_stated_facts(user_input)
        self.memory.add_message("user", user_input)

        max_loops = 5
        loop_count = 0

        while loop_count < max_loops:
            loop_count += 1
            trace["loops"] = loop_count
            logger.info(f"Agent Loop Iteration: {loop_count}")

            messages = [{"role": "system", "content": self._get_system_prompt()}]
            messages.extend(self.memory.get_conversational_window())

            response = self.client.chat.completions.create(
                model=settings.AGENT_MODEL,
                messages=messages,
                tools=AGENT_TOOLS_SCHEMA,
                tool_choice="auto",
                temperature=settings.TEMPERATURE,
                max_tokens=settings.MAX_TOKENS,
            )

            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls

            if tool_calls:
                logger.info(f"LLM decided to call {len(tool_calls)} tool(s).")
                self.memory.history.append(
                    {
                        "role": "assistant",
                        "content": response_message.content,
                        "tool_calls": [t.model_dump() for t in tool_calls],
                    }
                )

                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments or "{}")

                    logger.info(f"Executing tool: {function_name} with args: {function_args}")

                    if function_name not in self.available_tools:
                        tool_result = f"Error: Tool {function_name} is not available."
                    else:
                        try:
                            fn = self.available_tools[function_name]
                            if function_name == "calculate":
                                tool_result = fn(expression=function_args.get("expression"))
                            elif function_name == "retrieve_technical_documents":
                                tool_result = fn(query=function_args.get("query"))
                            else:
                                tool_result = fn(**function_args)
                        except Exception as e:
                            tool_result = f"Error executing tool: {str(e)}"

                    trace["tool_calls"].append(
                        {
                            "name": function_name,
                            "arguments": function_args,
                            "result_preview": str(tool_result)[:500],
                        }
                    )

                    self.memory.history.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": str(tool_result),
                        }
                    )
                continue

            final_answer = response_message.content or ""
            logger.info("LLM provided final textual answer.")
            self.memory.add_message("assistant", final_answer)
            if return_trace:
                return final_answer, trace
            return final_answer

        error_msg = "Agent halted: Maximum reasoning steps reached without a final answer."
        logger.error(error_msg)
        trace["error"] = error_msg
        if return_trace:
            return error_msg, trace
        return error_msg
