import pytest

from backend.app.tools import safe_calculate


def test_basic_calculator():
    assert float(safe_calculate("144 / 12")) == 12.0
    assert float(safe_calculate("(100 + 50) * 2 - 30")) == 270.0


def test_rejects_huge_exponent():
    out = safe_calculate("2**100000000")
    assert out.startswith("Error")


def test_rejects_deep_ast():
    expr = "+".join(["1"] * 200)
    out = safe_calculate(expr)
    assert out.startswith("Error")
