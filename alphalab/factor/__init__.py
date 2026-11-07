"""Factor analysis module."""

from alphalab.factor.engine import FactorEngine, FactorExpression
from alphalab.factor.analysis import FactorAnalyser
from alphalab.factor.utils import wide_from_long

__all__ = ["FactorEngine", "FactorExpression", "FactorAnalyser", "wide_from_long"]
