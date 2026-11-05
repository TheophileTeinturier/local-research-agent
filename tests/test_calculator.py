import re

import pytest

from local_research_agent.tools import ToolError, build_default_registry
from local_research_agent.tools.calculator import (
    MAX_EXPRESSION_LENGTH,
    CalculatorArgs,
    calculate,
    evaluate,
)


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("17 * 23 + 4", 395),
        ("2 + 3 * 4", 14),
        ("(2 + 3) * 4", 20),
        ("-3 + +5", 2),
        ("--4", 4),
        ("7 / 2", 3.5),
        ("7 // 2", 3),
        ("-7 // 2", -4),
        ("7 % 3", 1),
        ("2 ** 10", 1024),
        ("2 ** -1", 0.5),
        ("9 ** 0.5", 3.0),
        ("1.5e3 + 0.5", 1500.5),
        ("  42  ", 42),
    ],
)
def test_evaluate_valid(expression: str, expected: float) -> None:
    assert evaluate(expression) == pytest.approx(expected)


def test_int_arithmetic_is_exact() -> None:
    assert evaluate("123456789 * 987654321") == 121932631112635269


def test_calculate_formats_result() -> None:
    assert calculate(CalculatorArgs(expression="17 * 23 + 4")) == "17 * 23 + 4 = 395"
    assert calculate(CalculatorArgs(expression="0.1 + 0.2")) == "0.1 + 0.2 = 0.3"
    assert calculate(CalculatorArgs(expression="10 / 4")) == "10 / 4 = 2.5"


@pytest.mark.parametrize(
    ("expression", "message_part"),
    [
        ("", "Empty"),
        ("2 +", "Invalid syntax"),
        ("1 / 0", "Division by zero"),
        ("5 % 0", "Division by zero"),
        ("2 ^ 3", re.escape("use '**'")),
        ("2 << 3", "not supported"),
        ("1 < 2", "Unsupported syntax: Compare"),
        ("True + 1", "Unsupported value"),
        ("'a' * 3", "Unsupported value"),
        ("1j * 2", "Unsupported value"),
        ("(-8) ** 0.5", "not a real number"),
        ("x + 1", "Unsupported syntax: Name"),
        ("1" * (MAX_EXPRESSION_LENGTH + 1), "too long"),
    ],
)
def test_evaluate_invalid(expression: str, message_part: str) -> None:
    with pytest.raises(ToolError, match=message_part):
        evaluate(expression)


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo pwned')",
        "open('secrets.txt').read()",
        "().__class__.__bases__[0].__subclasses__()",
        "(lambda: 1)()",
        "[1, 2, 3]",
        "{'a': 1}",
        "abs(-1)",
        "1 if 1 else 2",
        "(x := 5)",
        "print(1)",
    ],
)
def test_evaluate_rejects_code(expression: str) -> None:
    with pytest.raises(ToolError, match="Unsupported syntax"):
        evaluate(expression)


@pytest.mark.parametrize(
    "expression",
    [
        "2 ** 100000",  # exponent guard, checked before computing
        "9 ** 9 ** 9",  # right-associative: 9 ** 387420489
        "10 ** 1000",  # within the exponent limit, but beyond the magnitude limit
        "(10 ** 300) * 100",
        "1e308 * 10",  # float overflow to inf
        "10.0 ** 400",  # float OverflowError
    ],
)
def test_evaluate_resource_guards(expression: str) -> None:
    with pytest.raises(ToolError, match=r"too large|not a finite"):
        evaluate(expression)


def test_nesting_up_to_parser_limit() -> None:
    assert evaluate("(" * 150 + "1" + ")" * 150) == 1
    # Beyond CPython's nested-parentheses limit (200): a clean ToolError, not a crash.
    with pytest.raises(ToolError, match="Invalid syntax"):
        evaluate("(" * 240 + "1" + ")" * 240)


def test_calculator_through_registry() -> None:
    registry = build_default_registry()

    ok = registry.execute("calculator", {"expression": "17 * 23 + 4"})
    assert ok.ok and ok.content == "17 * 23 + 4 = 395"

    malicious = registry.execute("calculator", {"expression": "__import__('os').getcwd()"})
    assert not malicious.ok
    assert malicious.metadata["error_type"] == "tool_error"
    assert malicious.content.startswith("Error: Unsupported syntax")
