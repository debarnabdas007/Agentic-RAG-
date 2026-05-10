import ast
import operator
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)

#  The Actual Python Functions ---

def safe_calculate(expression: str) -> str:
    """ A secure calculator that evaluates basic math expressions
    This Prevents malicious code execution and handles mathematical edge cases...
    """
    logger.info(f"Executing Calculator tool with expression: {expression}")
    
    # Supported operators
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg
    }

    def eval_expr(node):
        if isinstance(node, ast.Constant): # Modern Python approach.. we are not using .eval()
            if isinstance(node.value, (int, float)):
                return node.value
            raise TypeError("Only numeric constants are allowed.")
        elif isinstance(node, ast.BinOp):
            if type(node.op) not in operators:
                raise TypeError(f"Operator {type(node.op).__name__} is not allowed.")
            return operators[type(node.op)](eval_expr(node.left), eval_expr(node.right))
        elif isinstance(node, ast.UnaryOp):
            if type(node.op) not in operators:
                raise TypeError(f"Operator {type(node.op).__name__} is not allowed.")
            return operators[type(node.op)](eval_expr(node.operand))
        else:
            raise TypeError(f"Unsupported mathematical syntax: {type(node).__name__}")

    try:
        node = ast.parse(expression, mode='eval').body
        result = eval_expr(node)
        logger.info(f"Calculator result: {result}")
        return str(result)
    except ZeroDivisionError:
        logger.warning("Division by zero attempted.")
        return "Error: Division by zero is mathematically undefined."
    except Exception as e:
        logger.error(f"Calculator error: {e}")
        return f"Error evaluating expression: {e}."



#  The JSON Schemas for the LLM ---

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
                        "description": "The precise search query. Extract ONLY the raw technical entity or concept, nothing else. Example: 'cross-encoder reranking'."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluates a mathematical expression. Use this tool ONLY when explicit mathematical computation or numerical calculation is required. NEVER estimate arithmetic mentally.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A standard mathematical expression, strictly using numbers and basic operators (+, -, *, /, **). Example: '100 / 4'."
                    }
                },
                "required": ["expression"]
            }
        }
    }
]