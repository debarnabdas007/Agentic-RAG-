import ast
import math
import operator
from backend.config import settings
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


def _count_ast_nodes(node: ast.AST) -> int:
    n = 0
    for child in ast.walk(node):
        n += 1
    return n


def safe_calculate(expression: str) -> str:
    """
    Evaluates basic arithmetic via AST walking (no code execution).
    Exponent and magnitude are bounded to avoid pathological cases like 2**100000000
    tying up the worker.
    """
    logger.info(f"Executing Calculator tool with expression: {expression}")

    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    def eval_expr(node: ast.AST) -> float | int:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise TypeError("Only numeric constants are allowed.")
        if isinstance(node, ast.BinOp):
            if type(node.op) not in operators:
                raise TypeError(f"Operator {type(node.op).__name__} is not allowed.")
            if isinstance(node.op, ast.Pow):
                left_val = eval_expr(node.left)
                right_val = eval_expr(node.right)
                if not isinstance(right_val, (int, float)):
                    raise TypeError("Exponent must be numeric.")
                if isinstance(right_val, float) and not right_val.is_integer():
                    raise TypeError("Non-integer exponents are not supported.")
                exp = int(right_val)
                if abs(exp) > settings.CALC_MAX_ABS_EXPONENT:
                    raise ValueError("Exponent is too large.")
                base = float(left_val)
                if base != 0 and exp > 1:
                    est = abs(exp) * math.log10(max(abs(base), 1e-300))
                    if est > settings.CALC_MAX_RESULT_LOG10:
                        raise ValueError("Result would be too large to compute safely.")
                return operators[type(node.op)](left_val, right_val)
            left_val = eval_expr(node.left)
            right_val = eval_expr(node.right)
            return operators[type(node.op)](left_val, right_val)
        if isinstance(node, ast.UnaryOp):
            if type(node.op) not in operators:
                raise TypeError(f"Operator {type(node.op).__name__} is not allowed.")
            return operators[type(node.op)](eval_expr(node.operand))
        raise TypeError(f"Unsupported mathematical syntax: {type(node).__name__}")

    try:
        tree = ast.parse(expression, mode="eval")
        if not isinstance(tree, ast.Expression):
            raise TypeError("Invalid expression.")
        if _count_ast_nodes(tree) > settings.CALC_MAX_AST_NODES:
            raise ValueError("Expression has too many nodes.")
        result = eval_expr(tree.body)
        logger.info(f"Calculator result: {result}")
        return str(result)
    except ZeroDivisionError:
        logger.warning("Division by zero attempted.")
        return "Error: Division by zero is mathematically undefined."
    except Exception as e:
        logger.error(f"Calculator error: {e}")
        return f"Error evaluating expression: {e}."


AGENT_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_technical_documents",
            "description": "Searches the local vector database of recent cs.AI arXiv papers. Use this tool ONLY when the user asks about specific AI concepts, architectures, or facts from technical papers. Do not use for general conversational queries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The precise search query. Extract ONLY the raw technical entity or concept, nothing else. Example: 'cross-encoder reranking'.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluates a mathematical expression with AST bounds (no code injection; large exponents rejected). Use ONLY for explicit arithmetic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A mathematical expression using numbers and +, -, *, /, **. Example: '100 / 4'.",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]






"""
Now calculator is :
safe from code execution
safer from CPU/RAM abuse
bounded mathematically
"""