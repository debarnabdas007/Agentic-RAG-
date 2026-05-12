"""
Evaluation harness for the Agentic RAG system.
Run from repo root:  python -m backend.eval.evaluation_harness
Requires GROQ_API_KEY and a built vector index for retrieval cases.
"""

from __future__ import annotations

import re
from typing import Any

from backend.app.agent import SkycladAgent
from backend.app.retriever import AdvancedRetriever


class EvalCase:
    def __init__(
        self,
        question: str,
        expected_type: str,
        keywords: list[str] | None = None,
        should_refuse: bool = False,
    ):
        self.question = question
        self.expected_type = expected_type
        self.keywords = keywords or []
        self.should_refuse = should_refuse


REFUSAL_OR_CLARIFY_MARKERS = (
    "i don't know",
    "i do not know",
    "corpus does not contain",
    "does not contain",
    "which specific",
    "which paper",
    "what paper",
    "could you clarify",
    "please clarify",
    "please specify",
    "unclear",
    "outside the domain",
    "not able to",
    "cannot help",
    "can't help",
    "i'm specialized",
    "i am specialized",
    "only handles",
    "only answer",
    "need more context",
    "vague",
)


class EvaluationHarness:
    def __init__(self, shared_retriever: AdvancedRetriever | None = None):
        self._shared_retriever = shared_retriever
        self.results: list[dict[str, Any]] = []

    def run(
        self,
        test_cases: list[EvalCase],
        *,
        use_reranker: bool = True,
    ) -> dict[str, Any]:
        retriever = self._shared_retriever or AdvancedRetriever()
        passed = 0
        failed = 0
        self.results = []

        for i, test in enumerate(test_cases):
            agent = SkycladAgent(retriever=retriever)
            agent.use_reranker = use_reranker
            response = agent.chat(test.question)
            ok = self._check_response(response, test)
            status = "PASS" if ok else "FAIL"
            if ok:
                passed += 1
            else:
                failed += 1
            self.results.append(
                {
                    "case_id": i + 1,
                    "question": test.question,
                    "expected_type": test.expected_type,
                    "response": response,
                    "status": status,
                    "use_reranker": use_reranker,
                }
            )

        return {
            "total": len(test_cases),
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / len(test_cases) if test_cases else 0.0,
            "use_reranker": use_reranker,
            "results": self.results,
        }

    def _check_response(self, response: str, test: EvalCase) -> bool:
        response_lower = response.lower()

        if test.should_refuse:
            if any(m in response_lower for m in REFUSAL_OR_CLARIFY_MARKERS):
                return True
            if "?" in response and test.expected_type in ("clarify", "refuse"):
                return True
            return False

        if test.expected_type == "retrieve" and test.keywords:
            return all(k.lower() in response_lower for k in test.keywords)

        if test.expected_type == "calculate":
            return bool(re.search(r"\d+\.?\d*", response))

        if test.expected_type == "clarify":
            return "?" in response

        return True


EVAL_TEST_CASES: list[EvalCase] = [
    EvalCase(
        question="What is attention in the context of neural networks?",
        expected_type="retrieve",
        keywords=["attention", "neural"],
    ),
    EvalCase(
        question="Explain the concept of Reciprocal Rank Fusion in information retrieval.",
        expected_type="retrieve",
        keywords=["rank", "fusion"],
    ),
    EvalCase(
        question="What is 144 divided by 12?",
        expected_type="calculate",
    ),
    EvalCase(
        question="Calculate: (100 + 50) * 2 - 30",
        expected_type="calculate",
    ),
    EvalCase(
        question="Summarize the paper.",
        expected_type="clarify",
        should_refuse=True,
    ),
    EvalCase(
        question="What are its main contributions?",
        expected_type="clarify",
        should_refuse=True,
    ),
    EvalCase(
        question="Who won the World Cup in 2022?",
        expected_type="refuse",
        should_refuse=True,
    ),
    EvalCase(
        question="What's the capital of France?",
        expected_type="refuse",
        should_refuse=True,
    ),
    EvalCase(
        question=(
            "What is the computational complexity of transformer attention, "
            "and what notation is typically used?"
        ),
        expected_type="retrieve",
        keywords=["attention", "complex"],
    ),
    EvalCase(
        question="Tell me about reinforcement learning.",
        expected_type="retrieve",
        keywords=["reinforcement", "learning"],
    ),
    EvalCase(
        question="What is the exact architecture of GPT-47 released in 2099?",
        expected_type="refuse",
        should_refuse=True,
    ),
    EvalCase(
        question="Define overfitting in machine learning in one or two sentences.",
        expected_type="retrieve",
        keywords=["overfit"],
    ),
]


def compare_reranker_ablation() -> dict[str, Any]:
    """Run the same cases with reranker on vs off; prints pass-rate delta."""
    retriever = AdvancedRetriever()
    harness = EvaluationHarness(shared_retriever=retriever)
    with_r = harness.run(EVAL_TEST_CASES, use_reranker=True)
    without_r = harness.run(EVAL_TEST_CASES, use_reranker=False)
    return {
        "with_reranker_pass_rate": with_r["pass_rate"],
        "without_reranker_pass_rate": without_r["pass_rate"],
        "delta": with_r["pass_rate"] - without_r["pass_rate"],
        "with_reranker": with_r,
        "without_reranker": without_r,
    }


if __name__ == "__main__":
    harness = EvaluationHarness()
    results = harness.run(EVAL_TEST_CASES, use_reranker=True)
    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS: {results['passed']}/{results['total']} PASSED")
    print(f"Pass Rate: {results['pass_rate']:.1%} (reranker={results['use_reranker']})")
    print("=" * 60 + "\n")
    for result in results["results"]:
        sym = "PASS" if result["status"] == "PASS" else "FAIL"
        print(f"{sym} Case {result['case_id']}: {result['expected_type'].upper()}")
        print(f"   Q: {result['question']}")
        preview = result["response"][:200].replace("\n", " ")
        print(f"   R: {preview}...")
        print()

    print("\n--- Reranker on vs off (same retriever instance) ---\n")
    ab = compare_reranker_ablation()
    print(f"With reranker:    {ab['with_reranker_pass_rate']:.1%}")
    print(f"Without reranker: {ab['without_reranker_pass_rate']:.1%}")
    print(f"Delta:            {ab['delta']:+.1%}")
