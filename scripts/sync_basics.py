"""从 Tushare 同步基础信息到 SQLite。

同步 5 类品种：A股、ETF、期货、可转债、场外基金
存储到 4 个表：
    - basic_astock: A股
    - basic_etf: ETF
    - basic_future: 期货
    - basic_bond: 可转债
    - basic_fund: 场外基金（排除货币基金和 C 类，保留银华日利）

用法：
    python scripts/sync_basics.py
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from sqlite_utils import Database

from config import BASE_DIR, config

# 数据库路径
_DB_PATH = BASE_DIR / "data" / "market_data.db"


def _get_db():
    """获取数据库实例。"""
    return Database(str(_DB_PATH))


def _get_api():
    """获取 Tushare API。"""
    import tushare as ts
    token = config.TS_TOKEN
    if not token:
        raise RuntimeError("TS_TOKEN 未配置，请在 ~/.agilab/config.toml 中设置")
    return ts.pro_api(token)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """统一字段：把 ts_code 重命名为 symbol，删除旧的 symbol 列。"""
    if df.empty:
        return df
    if "ts_code" in df.columns:
        # 如果原表有 symbol 列，先删除
        if "symbol" in df.columns:
            df = df.drop(columns=["symbol"])
        # 把 ts_code 改名为 symbol
        df = df.rename(columns={"ts_code": "symbol"})
    return df


def _save(df: pd.DataFrame, table: str, pk: str | list) -> None:
    """保存 DataFrame 到 SQLite，先 drop 再写入。"""
    if df.empty:
        print(f"[跳过] {table}: 无数据")
        return
    # 统一 symbol
    df = _normalize(df)
    if "symbol" in df.columns:
        pk = "symbol"

    db = _get_db()
    # 先删除旧表，避免历史数据残留
    if table in db.table_names():
        db[table].drop()
    db[table].insert_all(df.to_dict("records"), pk=pk)
    print(f"[保存] {table}: {len(df)} 条")


def sync_astock():
    """同步A股基础信息。"""
    print("\n=== 同步 A股 ===")
    api = _get_api()
    df = api.stock_basic(
        exchange="",
        list_status="L",
        fields="ts_code,symbol,name,area,industry,market,list_date,list_status,delist_date,is_hs"
    )
    _save(df, "basic_astock", "ts_code")


def sync_etf():
    """同步ETF基础信息。"""
    print("\n=== 同步 ETF ===")
    api = _get_api()
    df = api.fund_basic(market="E", status="L")
    _save(df, "basic_etf", "ts_code")


def sync_future():
    """同步期货基础信息（主连合约，用于日频分析）。"""
    print("\n=== 同步 期货主连 ===")
    api = _get_api()
    # fut_type="2" 为连续/主力合约（如 CU.SHF 沪铜主力）
    df1 = api.fut_basic(exchange="DCE", fut_type="2", fields="ts_code,symbol,name,list_date,delist_date,exchange,pi_tip,price_unit,trade_unit,contract_unit,min_price_chg,last_ddate")
    df2 = api.fut_basic(exchange="CZCE", fut_type="2", fields="ts_code,symbol,name,list_date,delist_date,exchange,pi_tip,price_unit,trade_unit,contract_unit,min_price_chg,last_ddate")
    df3 = api.fut_basic(exchange="SHFE", fut_type="2", fields="ts_code,symbol,name,list_date,delist_date,exchange,pi_tip,price_unit,trade_unit,contract_unit,min_price_chg,last_ddate")
    df4 = api.fut_basic(exchange="GFEX", fut_type="2", fields="ts_code,symbol,name,list_date,delist_date,exchange,pi_tip,price_unit,trade_unit,contract_unit,min_price_chg,last_ddate")
    df5 = api.fut_basic(exchange="INE", fut_type="2", fields="ts_code,symbol,name,list_date,delist_date,exchange,pi_tip,price_unit,trade_unit,contract_unit,min_price_chg,last_ddate")
    df6 = api.fut_basic(exchange="CFFEX", fut_type="2", fields="ts_code,symbol,name,list_date,delist_date,exchange,pi_tip,price_unit,trade_unit,contract_unit,min_price_chg,last_ddate")

    df = pd.concat([df1, df2, df3, df4, df5, df6], ignore_index=True)
    _save(df, "basic_future", "ts_code")


def sync_bond():
    """同步可转债基础信息。"""
    print("\n=== 同步 可转债 ===")
    api = _get_api()
    df = api.cb_basic(fields="ts_code,bond_short_name,stk_code,stk_name,list_date,delist_date,issue_size,conversion_price")
    _save(df, "basic_bond", "ts_code")


def sync_fund():
    """同步场外基金基础信息，排除货币基金和 C 类，保留银华日利。"""
    print("\n=== 同步 场外基金 ===")
    api = _get_api()
    # 场外基金
    df = api.fund_basic(market="O", status="L")
    if df.empty:
        print("[跳过] basic_fund: 无数据")
        return

    print(f"[原始] 场外基金共 {len(df)} 条")

    # 保留银华日利
    silver_fund = df[df["name"].str.contains("银华日利", na=False)]

    # 排除货币基金（fund_type == 'E' 通常表示货币基金，这里按名称关键字双重保险）
    df = df[df["fund_type"] != "E"]
    df = df[~df["name"].str.contains("货币|现金|余额宝|理财", na=False)]

    # 排除 C 类（名称以 C 结尾，且 fund_type 不是货币基金）
    df = df[~df["name"].str.endswith("C", na=False)]

    # 把银华日利加回来
    df = pd.concat([df, silver_fund], ignore_index=True)
    df = df.drop_duplicates(subset=["ts_code"])

    _save(df, "basic_fund", "ts_code")


def main():
    """执行全部同步。"""
    print(f"数据库路径: {_DB_PATH}")
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    sync_astock()
    sync_etf()
    sync_future()
    sync_bond()
    sync_fund()

    print("\n[完成] 基础信息同步完成")


if __name__ == "__main__":
    main()
