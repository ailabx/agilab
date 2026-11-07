"""Core of the mini factor expression engine."""

from __future__ import annotations

import ast
import operator
from typing import Any, Callable

import numpy as np
import pandas as pd

from alphalab.factor.expression.registry import _OPS


# --------------------------------------------------------------------------- #
# Allowed AST nodes
# --------------------------------------------------------------------------- #
_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Constant,
    ast.Load,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.UAdd,
    ast.USub,
    ast.Not,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.BoolOp,
    ast.And,
    ast.Or,
)

_BIN_OPS: dict[type, Callable] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}

_COMPARE_OPS: dict[type, Callable] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}

_BOOL_OPS: dict[type, Callable] = {
    ast.And: lambda a, b: a & b,
    ast.Or: lambda a, b: a | b,
}


# --------------------------------------------------------------------------- #
# Expression class
# --------------------------------------------------------------------------- #
class FactorExpression:
    """A simple, safe factor expression evaluator for wide-format panels."""

    def __init__(self, source: str):
        self.source = source
        self.tree = ast.parse(source, mode="eval")
        self._validate(self.tree)

    def evaluate(self, fields: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Evaluate the expression against a dictionary of wide-format fields."""
        return self._eval_node(self.tree.body, fields)

    def _validate(self, node: ast.AST) -> None:
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"unsupported AST node: {type(node).__name__}")
        for child in ast.iter_child_nodes(node):
            self._validate(child)

    def _eval_node(self, node: ast.AST, fields: dict[str, pd.DataFrame]) -> Any:
        if isinstance(node, ast.Constant):
            return node.value

        if isinstance(node, ast.Name):
            if node.id not in fields:
                raise NameError(f"unknown field or operator: {node.id}")
            return fields[node.id]

        if isinstance(node, ast.UnaryOp):
            value = self._eval_node(node.operand, fields)
            if isinstance(node.op, ast.UAdd):
                return value
            if isinstance(node.op, ast.USub):
                return -value
            if isinstance(node.op, ast.Not):
                return ~value.astype(bool)
            raise ValueError(f"unsupported unary op: {type(node.op).__name__}")

        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left, fields)
            right = self._eval_node(node.right, fields)
            return _BIN_OPS[type(node.op)](left, right)

        if isinstance(node, ast.Compare):
            left = self._eval_node(node.left, fields)
            result = None
            for op, comparator in zip(node.ops, node.comparators, strict=True):
                right = self._eval_node(comparator, fields)
                current = _COMPARE_OPS[type(op)](left, right)
                result = current if result is None else result & current
                left = right
            return result

        if isinstance(node, ast.BoolOp):
            values = [self._eval_node(v, fields) for v in node.values]
            result = values[0]
            for v in values[1:]:
                result = _BOOL_OPS[type(node.op)](result, v)
            return result

        if isinstance(node, ast.Call):
            return self._eval_call(node, fields)

        raise ValueError(f"unsupported node: {type(node).__name__}")

    def _eval_call(self, node: ast.Call, fields: dict[str, pd.DataFrame]) -> Any:
        if not isinstance(node.func, ast.Name):
            raise ValueError("only simple function calls are supported")

        func_name = node.func.id.upper()
        if func_name not in _OPS:
            raise NameError(f"unknown operator: {func_name}")

        args = [self._eval_node(arg, fields) for arg in node.args]
        return _OPS[func_name](*args)
