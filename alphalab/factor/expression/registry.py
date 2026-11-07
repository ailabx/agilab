"""Shared operator registry for the factor expression engine."""

from __future__ import annotations

from typing import Callable

_OPS: dict[str, Callable] = {}


def register_op(func: Callable) -> Callable:
    """Decorator to register a factor operator."""
    _OPS[func.__name__.upper()] = func
    return func


def available_ops() -> list[str]:
    """Return the list of supported factor operators."""
    return sorted(_OPS.keys())
