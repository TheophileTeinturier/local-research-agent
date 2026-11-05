"""Calculator tool: safe arithmetic evaluation without `eval`.

The expression is parsed with `ast.parse(mode="eval")`, and the resulting tree is walked by
an interpreter that only knows numbers, an allowlist of arithmetic operators, and parentheses
(which leave no node in the tree; they only shape it). Any other node, such as a name,
call, attribute, subscript or lambda, is rejected before anything is evaluated, so input
like `__import__('os').system(...)` can never run. Resource guards (expression length,
exponent size, result magnitude) stop inputs such as `9**9**9` from using up CPU or memory.
"""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field

from .base import Tool, ToolError

Number = int | float

MAX_EXPRESSION_LENGTH = 500
MAX_EXPONENT = 1000
MAX_ABS_VALUE = 10**300  # just below the float limit (~1.8e308), so int and float agree

_ALLOWED_SYNTAX = "Only numbers, + - * / // % ** and parentheses are allowed."


def _safe_pow(base: Number, exponent: Number) -> Number:
    # Checked *before* computing: 2 ** 10**9 would hang. With |base| <= MAX_ABS_VALUE
    # (enforced on every intermediate value) and |exponent| <= MAX_EXPONENT, the worst case is
    # about a million bits, computed in milliseconds and then rejected by the magnitude check.
    if abs(exponent) > MAX_EXPONENT:
        raise ToolError(
            f"Exponent {exponent} is too large (maximum absolute value {MAX_EXPONENT})."
        )
    result = base**exponent
    if isinstance(result, complex):
        raise ToolError("Result is not a real number (fractional power of a negative number).")
    return result


_BINARY_OPERATORS: dict[type[ast.operator], Callable[[Number, Number], Number]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: _safe_pow,
}

_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[Number], Number]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def evaluate(expression: str) -> Number:
    """Evaluate an arithmetic expression. Raises `ToolError` with a model-readable message."""
    expression = expression.strip()
    if not expression:
        raise ToolError("Empty expression.")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ToolError(f"Expression is too long (maximum {MAX_EXPRESSION_LENGTH} characters).")
    try:
        tree = ast.parse(expression, mode="eval")
        return _eval_node(tree.body)
    except SyntaxError as exc:
        raise ToolError(f"Invalid syntax: {exc.msg}. {_ALLOWED_SYNTAX}") from None
    except RecursionError:
        raise ToolError("Expression is too deeply nested.") from None


def _eval_node(node: ast.expr) -> Number:
    if isinstance(node, ast.Constant):
        value = node.value
        # bool is a subclass of int: reject True/False explicitly.
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ToolError(f"Unsupported value {value!r}. {_ALLOWED_SYNTAX}")
        return _check_magnitude(value)

    if isinstance(node, ast.BinOp):
        binary_op = _BINARY_OPERATORS.get(type(node.op))
        if binary_op is None:
            raise ToolError(_unsupported_operator_message(node.op))
        left, right = _eval_node(node.left), _eval_node(node.right)
        try:
            return _check_magnitude(binary_op(left, right))
        except ZeroDivisionError:
            raise ToolError("Division by zero.") from None
        except OverflowError:
            raise ToolError("Result is too large (maximum absolute value 1e300).") from None

    if isinstance(node, ast.UnaryOp):
        unary_op = _UNARY_OPERATORS.get(type(node.op))
        if unary_op is None:
            raise ToolError(_unsupported_operator_message(node.op))
        return unary_op(_eval_node(node.operand))

    raise ToolError(f"Unsupported syntax: {type(node).__name__}. {_ALLOWED_SYNTAX}")


def _check_magnitude(value: Number) -> Number:
    if isinstance(value, float) and not math.isfinite(value):
        raise ToolError("Result is not a finite number (overflow).")
    if abs(value) > MAX_ABS_VALUE:
        raise ToolError("Result is too large (maximum absolute value 1e300).")
    return value


def _unsupported_operator_message(op: ast.operator | ast.unaryop) -> str:
    if isinstance(op, ast.BitXor):
        return "'^' is not supported (it is bitwise XOR in this syntax); use '**' for powers."
    return f"Operator {type(op).__name__} is not supported. {_ALLOWED_SYNTAX}"


def format_number(value: Number) -> str:
    if isinstance(value, int):
        return str(value)
    # 15 significant digits hide binary floating-point noise: 0.1 + 0.2 -> "0.3", 10 / 2 -> "5".
    return format(value, ".15g")


class CalculatorArgs(BaseModel):
    # Unknown fields are an error rather than silently ignored, so a confused model gets told.
    model_config = ConfigDict(extra="forbid")

    expression: str = Field(
        min_length=1,
        max_length=MAX_EXPRESSION_LENGTH,
        description="Arithmetic expression in Python operator syntax, e.g. '(17 * 23 + 4) / 3'.",
    )


def calculate(args: CalculatorArgs) -> str:
    result = format_number(evaluate(args.expression))
    # Echoing the expression lets the model check that it computed what it meant to.
    return f"{args.expression.strip()} = {result}"


CALCULATOR_DESCRIPTION = (
    "Evaluate an arithmetic expression exactly and return its numeric result. "
    "Use it for any calculation instead of computing in your head. "
    "Supports integers, decimals, scientific notation (1.5e3), parentheses and the operators "
    "+ - * / // (floor division) % (modulo) ** (power). "
    "'%' is modulo, not percent: write '15% of 80' as '0.15 * 80'. Use '**' for powers, not '^'. "
    "No variables, functions (sqrt, sin, log...), units or text: for a square root use '** 0.5'. "
    "Put the whole calculation in one expression."
)

calculator_tool: Tool[CalculatorArgs] = Tool(
    name="calculator",
    description=CALCULATOR_DESCRIPTION,
    args_model=CalculatorArgs,
    function=calculate,
)
