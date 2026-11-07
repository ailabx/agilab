"""Single-factor analysis: IC, Rank IC, quantile returns, turnover."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats


class FactorAnalyser:
    """Analyse a factor's predictive power.

    Parameters
    ----------
    factor : pd.DataFrame
        Wide-format factor values (date x symbol).
    forward_returns : pd.DataFrame
        Wide-format forward returns (date x symbol), e.g. next-day or 5-day returns.
    periods : int
        Forward holding period (used only for reporting).
    """

    def __init__(
        self,
        factor: pd.DataFrame,
        forward_returns: pd.DataFrame,
        periods: int = 1,
    ):
        self.factor = factor.sort_index()
        self.forward_returns = forward_returns.sort_index()
        self.periods = periods
        self._align()

    def _align(self) -> None:
        common_index = self.factor.index.intersection(self.forward_returns.index)
        common_cols = self.factor.columns.intersection(self.forward_returns.columns)
        self.factor = self.factor.loc[common_index, common_cols]
        self.forward_returns = self.forward_returns.loc[common_index, common_cols]

    # ------------------------------------------------------------------ #
    # IC analysis (vectorised per date)
    # ------------------------------------------------------------------ #
    def ic_series(self, method: Literal["pearson", "spearman"] = "pearson") -> pd.Series:
        """Cross-sectional IC for each date using vectorised computation."""
        f = self.factor
        r = self.forward_returns

        valid = f.notna() & r.notna()
        count = valid.sum(axis=1)

        # Standardise per row and compute Pearson on (f, r)
        def _row_pearson(x, y):
            count_ok = valid.sum(axis=1)
            x_mean = x.mean(axis=1)
            y_mean = y.mean(axis=1)
            xm = x.sub(x_mean, axis=0)
            ym = y.sub(y_mean, axis=0)
            num = (xm * ym).sum(axis=1)
            denom = np.sqrt((xm ** 2).sum(axis=1) * (ym ** 2).sum(axis=1))
            ic = num / denom.replace(0, np.nan)
            ic = ic.where(count_ok >= 10)
            return ic

        if method == "spearman":
            f = f.rank(axis=1)
            r = r.rank(axis=1)

        ic = _row_pearson(f, r)
        # Mask constant rows: std == 0 means no variation
        ic = ic.where((f.std(axis=1, ddof=0) > 0) & (r.std(axis=1, ddof=0) > 0))
        return ic

    def ic_summary(self, method: Literal["pearson", "spearman"] = "pearson") -> dict:
        """Summary statistics for IC series."""
        ic = self.ic_series(method=method)
        return {
            "mean": float(ic.mean()),
            "std": float(ic.std()),
            "ir": float(ic.mean() / ic.std()) if ic.std() != 0 else np.nan,
            "positive_ratio": float(ic.dropna().gt(0).mean()),
            "count": int(ic.notna().sum()),
        }

    # ------------------------------------------------------------------ #
    # Quantile returns (vectorised)
    # ------------------------------------------------------------------ #
    def quantile_returns(
        self,
        quantiles: int = 5,
        demean: bool = True,
    ) -> pd.DataFrame:
        """Average forward return per factor quantile group, per date.

        Returns
        -------
        mean_returns : pd.DataFrame
            DataFrame indexed by quantile label, columns are mean returns.
        """
        f = self.factor
        r = self.forward_returns
        valid = f.notna() & r.notna()

        # Stack to long format with (date, symbol) MultiIndex
        f_stack = f[valid].stack(future_stack=True)
        r_stack = r[valid].stack(future_stack=True)
        f_stack.index.names = ["date", "symbol"]
        r_stack.index.names = ["date", "symbol"]

        # Compute quantile labels per date
        def _qcut_group(g):
            if len(g) < quantiles:
                return pd.Series(np.nan, index=g.index)
            try:
                return pd.qcut(g, quantiles, labels=False, duplicates="drop")
            except Exception:
                return pd.Series(np.nan, index=g.index)

        labels = f_stack.groupby(level=0, group_keys=False).apply(_qcut_group)
        mask = labels.notna()
        labels = labels[mask]
        r_stack = r_stack[mask]

        if len(r_stack) == 0:
            return pd.DataFrame()

        group = r_stack.groupby(labels).mean()
        if demean:
            group = group - r_stack.mean()

        return group.to_frame("mean_return")

    # ------------------------------------------------------------------ #
    # Turnover / stability
    # ------------------------------------------------------------------ #
    def turnover(self, top_n: int = 50) -> pd.Series:
        """Average turnover of top-N factor portfolio across consecutive dates."""
        turnovers: list[float] = []
        prev_set: set[str] | None = None
        for date in self.factor.index:
            f = self.factor.loc[date]
            valid = f.dropna()
            if len(valid) < top_n:
                continue
            top = set(valid.nlargest(top_n).index.tolist())
            if prev_set is not None:
                union = top | prev_set
                intersection = top & prev_set
                if union:
                    turnovers.append(1.0 - len(intersection) / len(union))
            prev_set = top
        return pd.Series(turnovers, name="turnover")

    def summary(self) -> dict:
        """Full summary table."""
        ic = self.ic_summary("pearson")
        rank_ic = self.ic_summary("spearman")
        qret = self.quantile_returns()
        turnover = self.turnover()
        return {
            "IC": ic,
            "Rank_IC": rank_ic,
            "quantile_returns": qret.to_dict(),
            "turnover": {"mean": float(turnover.mean()) if not turnover.empty else None},
        }
