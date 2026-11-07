"""Mini factor expression engine using AST + wide-format panels."""

from alphalab.factor.expression.core import FactorExpression
from alphalab.factor.expression.registry import available_ops, register_op

# Trigger operator registration so available_ops() and evaluate() see all operators.
import alphalab.factor.ops  # noqa: F401

__all__ = ["FactorExpression", "available_ops", "register_op"]
