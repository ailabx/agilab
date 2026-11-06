"""Tushare data-source adapter."""

from __future__ import annotations

from typing import Optional

import pandas as pd


class TushareSource:
    """Thin tushare adapter for raw data access."""

    def __init__(self, token: Optional[str] = None) -> None:
        if token is None:
            from config import config

            token = config.TS_TOKEN
        import tushare as ts

        self._pro = ts.pro_api(token)

    def fetch_trade_cal(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        start = start or "20100101"
        end = end or pd.Timestamp.now().strftime("%Y%m%d")
        df = self._pro.trade_cal(
            exchange="SSE",
            start_date=start,
            end_date=end,
        )
        df["is_open"] = df["is_open"].astype(int)
        return df

    def fetch_stock_basic(self) -> pd.DataFrame:
        """Fetch full stock_basic from tushare and rename ts_code -> symbol."""
        fields = (
            "ts_code,name,area,industry,fullname,enname,cnspell,market,"
            "exchange,curr_type,list_status,list_date,delist_date,is_hs"
        )
        df = self._pro.stock_basic(exchange="", list_status="", fields=fields)
        df = df.rename(columns={"ts_code": "symbol"})
        return df

    def fetch_daily(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Fetch daily bars for one symbol from tushare."""
        df = self._pro.daily(ts_code=symbol, start_date=start, end_date=end)
        if df is None or df.empty:
            return pd.DataFrame(columns=self._daily_columns())
        df = df.rename(columns={"ts_code": "symbol"})
        df["symbol"] = df["symbol"].astype(str)
        return df

    def fetch_adj_factor(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Fetch adjustment factor for one symbol from tushare."""
        df = self._pro.adj_factor(ts_code=symbol, start_date=start, end_date=end)
        if df is None or df.empty:
            return pd.DataFrame(columns=["symbol", "trade_date", "adj_factor"])
        df = df.rename(columns={"ts_code": "symbol"})
        df["symbol"] = df["symbol"].astype(str)
        df["adj_factor"] = pd.to_numeric(df["adj_factor"], errors="coerce")
        return df

    def fetch_fina_indicator(
        self, symbol: str, start: str, end: str
    ) -> pd.DataFrame:
        """Fetch financial indicators for one symbol from tushare."""
        df = self._pro.fina_indicator(ts_code=symbol, start_date=start, end_date=end)
        if df is None or df.empty:
            return pd.DataFrame(columns=_FINA_COLUMNS)
        df = df.rename(columns={"ts_code": "symbol"})
        df["symbol"] = df["symbol"].astype(str)
        # Keep date columns as strings; numeric columns are left as-is.
        for col in _DATE_COLUMNS_FINA:
            if col in df.columns:
                df[col] = df[col].astype(str)
        return df

    def fetch_index_weight(
        self, index_code: str, start: str, end: str
    ) -> pd.DataFrame:
        """Fetch index constituent weights from tushare."""
        df = self._pro.index_weight(
            index_code=index_code, start_date=start, end_date=end
        )
        if df is None or df.empty:
            return pd.DataFrame(
                columns=["index_code", "con_code", "trade_date", "weight"]
            )
        df["index_code"] = df["index_code"].astype(str)
        df["con_code"] = df["con_code"].astype(str)
        df["trade_date"] = df["trade_date"].astype(str)
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
        return df

    @staticmethod
    def _daily_columns() -> list[str]:
        return [
            "symbol", "trade_date", "open", "high", "low", "close",
            "pre_close", "change", "pct_chg", "vol", "amount",
        ]


_DATE_COLUMNS_FINA = ["ann_date", "end_date"]

_FINN_COLS = [
    "symbol", "ann_date", "end_date", "eps", "dt_eps", "total_revenue_ps",
    "revenue_ps", "capital_rese_ps", "surplus_rese_ps", "undisturbed_profit_ps",
    "grossprofit_margin", "operate_profit_ps", "income_before_tax_ps",
    "netprofit_margin", "roe", "roe_waa", "roe_dt", "roa", "npta", "roic",
    "roe_yearly", "roa2_yearly", "roe_ltm", "grossprofit_yoy", "op_yoy",
    "ebt_yoy", "profit_dedt_yoy", "growth_yoy", "sales_yoy", "profit_yoy",
    "rd_exp", "rd_exp_yoy", "eps_yoy", "deductedeps_yoy", "grossprofit_qoq",
    "profit_qoq", "sales_qoq", "op_qoq", "roe_qoq", "or_yoy", "or_qoq",
    "tr_yoy", "tr_qoq", "equity_yoy", "equity_qoq", "assets_yoy",
    "assets_qoq", "st_borrow", "lt_borrow", "total_liab", "total_assets",
    "total_ncl", "fixed_assets", "int_assets", "goodwill", "inventories",
    "money_cap", "notes_receiv", "accounts_receiv", "oth_receiv",
    "prepayment", "div_receiv", "interest_receiv", "lt_receiv", "fa_trade",
    "fa_available_for_sale", "held_for_sale", "defer_tax_assets",
    "deposit", " Derivative_assets", "lt_equity_invest", "invest_real_estate",
    "construction_in_process", "contract_assets", "produc_bio_assets",
    "oil_gas_assets", "use_rights_assets", "total_ca", "total_cl",
    "total_nca", "total_nca_lt", "deferred_income", "oth_assets",
    "settlement_provision", "loans_to_cb", "premium_receiv", "defer_inc",
    "cb_valuation", "loan_otb", "trading_fa", "noncurrent_assets_one_yr",
    "oth_cur_assets", "total_liab_equity", "lt_payable", "bonds_payable",
    "lt_payable_to_advance", "deferred_tax_liab", "deferred_income_lt",
    "oth_noncur_liabilities", "st_bonds_payable", "bond_payable",
    "minority_int", "preferred_stock", "perpetual_bond", "capital_rese",
    "undistributed_profit", "surplus_rese", "goodwill_lt_amort",
    "accumulated_oth_compre_income", "treasury_stock", "oth_equity_tools",
    "oth_equity_interest", "total_hldr_eqy_exc_min_int",
    "total_hldr_eqy_inc_min_int", "total_liab_hldr_eqy",
    "lt_payroll_payable", "accounts_payable", "adv_receipts",
    "st_oth_payable", "lt_oth_payable", "proceeds_from_disposal_long",
    "dividend_payable", "notes_payable", "oth_payable", "accrued_expense",
    "payroll_payable", "int_payable", "taxes_payable", "bonds_payable2",
    "deferred_rev", "oth_cl_lt", "total_cur_liab", "total_non_cur_liab",
    "lt_borrow2", "lt_payable2", "deferred_income2", "minority_int2",
    "total_equity", "total_liab_equity2", "tangible_assets",
    "net_debt", "working_capital", "networking_capital", "retained_earnings",
    "eps_basic", "eps_diluted", "bps", "mgjyxjje", "total_profit",
    "income_tax", "expenses", "op_profit", "non_oper_income", "non_oper_exp",
    "minority_profit", "oth_compr_income", "total_compr_income",
    "profit_dedt", "ebit", "ebitda", "fcff", "fcfe", "current_ratio",
    "quick_ratio", "cash_ratio", "ar_turn", "inv_turn", "ca_turn",
    "fa_turn", "total_asset_turn", "op_income", "valuechange_income",
    "interst_income", "interst_exp", "admin_exp", "sellexp", "rd_expense",
    "financial_expense", "asset_impairment", "credit_impairment",
    "operate_profit", "invest_income", "assoc_invest_income",
    "exchange_income", "operate_expense", "insurance_premium",
    "reinsurance_income", "reinsurance", "compensate_payout",
    "compensate_refund", "reserve_purchase_refund", "policy_div_paid",
    "reinsur_exp", "commission_expense", "prem_received", "n_sec_tc_income",
    "n_reinsur_premium", "n_commis_income", "n_oth_income",
    "premium_earned", "invest_income_minority", "oth_income", "income_before_tax",
    "net_profit", "profit_after_tax", "net_profit_dedt", "unrecognized_tax",
    "op_cf", "cf_from_op", "cf_from_inv_act", "cf_from_fina_act",
    "net_incr_cash_cheque", "cash_at_equi_end_period", "cash_end_period",
    "cash_beg_period", "equiv_end_period", "total_notes_receiv", "acct_payable",
    "notes_payable2", "st_oth_receiv", "lt_oth_receiv",
]


# fina_indicator fields known to be available in tushare pro
_FINN_FIELDS = [
    "ann_date", "end_date", "eps", "dt_eps", "total_revenue_ps",
    "revenue_ps", "capital_rese_ps", "surplus_rese_ps", "undisturbed_profit_ps",
    "grossprofit_margin", "operate_profit_ps", "income_before_tax_ps",
    "netprofit_margin", "roe", "roe_waa", "roe_dt", "roa", "npta", "roic",
    "roe_yearly", "roa2_yearly", "roe_ltm", "grossprofit_yoy", "op_yoy",
    "ebt_yoy", "profit_dedt_yoy", "growth_yoy", "sales_yoy", "profit_yoy",
    "rd_exp", "rd_exp_yoy", "eps_yoy", "deductedeps_yoy", "grossprofit_qoq",
    "profit_qoq", "sales_qoq", "op_qoq", "roe_qoq", "or_yoy", "or_qoq",
    "tr_yoy", "tr_qoq", "equity_yoy", "equity_qoq", "assets_yoy",
    "assets_qoq", "st_borrow", "lt_borrow", "total_liab", "total_assets",
    "total_ncl", "fixed_assets", "int_assets", "goodwill", "inventories",
    "money_cap", "notes_receiv", "accounts_receiv", "oth_receiv",
    "prepayment", "div_receiv", "interest_receiv", "lt_receiv", "fa_trade",
    "fa_available_for_sale", "held_for_sale", "defer_tax_assets",
    "deposit", "derivative_assets", "lt_equity_invest", "invest_real_estate",
    "construction_in_process", "contract_assets", "produc_bio_assets",
    "oil_gas_assets", "use_rights_assets", "total_ca", "total_cl",
    "total_nca", "total_nca_lt", "deferred_income", "oth_assets",
    "settlement_provision", "loans_to_cb", "premium_receiv", "defer_inc",
    "cb_valuation", "loan_otb", "trading_fa", "noncurrent_assets_one_yr",
    "oth_cur_assets", "total_liab_equity", "lt_payable", "bonds_payable",
    "lt_payable_to_advance", "deferred_tax_liab", "deferred_income_lt",
    "oth_noncur_liabilities", "st_bonds_payable", "bond_payable",
    "minority_int", "preferred_stock", "perpetual_bond", "capital_rese",
    "undistributed_profit", "surplus_rese", "goodwill_lt_amort",
    "accumulated_oth_compre_income", "treasury_stock", "oth_equity_tools",
    "oth_equity_interest", "total_hldr_eqy_exc_min_int",
    "total_hldr_eqy_inc_min_int", "total_liab_hldr_eqy",
    "lt_payroll_payable", "accounts_payable", "adv_receipts",
    "st_oth_payable", "lt_oth_payable", "proceeds_from_disposal_long",
    "dividend_payable", "notes_payable", "oth_payable", "accrued_expense",
    "payroll_payable", "int_payable", "taxes_payable", "bonds_payable2",
    "deferred_rev", "oth_cl_lt", "total_cur_liab", "total_non_cur_liab",
    "lt_borrow2", "lt_payable2", "deferred_income2", "minority_int2",
    "total_equity", "total_liab_equity2", "tangible_assets",
    "net_debt", "working_capital", "networking_capital", "retained_earnings",
    "eps_basic", "eps_diluted", "bps", "mgjyxjje", "total_profit",
    "income_tax", "expenses", "op_profit", "non_oper_income", "non_oper_exp",
    "minority_profit", "oth_compr_income", "total_compr_income",
    "profit_dedt", "ebit", "ebitda", "fcff", "fcfe", "current_ratio",
    "quick_ratio", "cash_ratio", "ar_turn", "inv_turn", "ca_turn",
    "fa_turn", "total_asset_turn", "op_income", "valuechange_income",
    "interst_income", "interst_exp", "admin_exp", "sellexp", "rd_expense",
    "financial_expense", "asset_impairment", "credit_impairment",
    "operate_profit", "invest_income", "assoc_invest_income",
    "exchange_income", "operate_expense", "insurance_premium",
    "reinsurance_income", "reinsurance", "compensate_payout",
    "compensate_refund", "reserve_purchase_refund", "policy_div_paid",
    "reinsur_exp", "commission_expense", "prem_received", "n_sec_tc_income",
    "n_reinsur_premium", "n_commis_income", "n_oth_income",
    "premium_earned", "invest_income_minority", "oth_income", "income_before_tax",
    "net_profit", "profit_after_tax", "net_profit_dedt", "unrecognized_tax",
    "op_cf", "cf_from_op", "cf_from_inv_act", "cf_from_fina_act",
    "net_incr_cash_cheque", "cash_at_equi_end_period", "cash_end_period",
    "cash_beg_period", "equiv_end_period", "total_notes_receiv", "acct_payable",
    "notes_payable2", "st_oth_receiv", "lt_oth_receiv",
]

_FINA_COLUMNS = ["symbol", "ann_date", "end_date"] + _FINN_FIELDS
