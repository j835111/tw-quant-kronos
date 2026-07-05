import logging
import sqlite3
import numpy as np
import pandas as pd

logger = logging.getLogger("xreg")

def apply_xreg_adjustment(
    db_path: str,
    symbols: list[str],
    rebal_date_str: str,
    scores_gbdt: dict[str, float],
    trading_days: list[str],
    lookback: int = 60,
    purging_gap: int = 5,
    hold_days: int = 5,
    mult: float = 2.0,
) -> dict[str, float]:
    """Calculate rolling individual stock momentum (XReg Intercept-only) and adjust scores.
    
    This replaces the defunct Ridge regression with a robust, simplified rolling return mean,
    while enforcing look-ahead leak guards and dynamic hold_days alignment.
    """
    # 🔴 安全防線：防止未來數據洩漏
    assert purging_gap >= hold_days, f"purging_gap ({purging_gap}) must be >= hold_days ({hold_days}) to prevent look-ahead leak!"
    
    if rebal_date_str not in trading_days:
        logger.warning(f"Rebalance date {rebal_date_str} not in trading calendar. Skipping XReg adjustment.")
        return scores_gbdt.copy()
        
    curr_idx = trading_days.index(rebal_date_str)
    if curr_idx < lookback + purging_gap:
        logger.warning(f"Insufficient history for date {rebal_date_str} (index {curr_idx} < {lookback + purging_gap}). Skipping XReg.")
        return scores_gbdt.copy()
        
    start_idx = curr_idx - lookback - purging_gap
    load_dates = trading_days[start_idx : curr_idx + 1]
    history_dates = trading_days[start_idx : curr_idx - purging_gap]
    
    # 僅讀取 symbols 列表中有分數的個股歷史價格
    conn = sqlite3.connect(db_path)
    df_prices = pd.read_sql(
        "SELECT date, symbol, open FROM daily_prices WHERE date BETWEEN ? AND ?",
        conn,
        params=[load_dates[0], load_dates[-1]]
    )
    conn.close()
    
    if df_prices.empty:
        logger.warning(f"No price data retrieved for window {load_dates[0]} to {load_dates[-1]}. Skipping XReg.")
        return scores_gbdt.copy()
        
    df_prices['date'] = pd.to_datetime(df_prices['date']).dt.strftime('%Y-%m-%d')
    df_prices = df_prices[df_prices['date'].isin(load_dates)]
    
    # 構建開盤價矩陣
    price_matrix = df_prices.pivot(index='date', columns='symbol', values='open').reindex(load_dates)
    
    # 🔴 標籤對齊：動態根據策略的 hold_days 設定 y 標籤持有期
    returns_matrix = price_matrix.shift(-(hold_days + 1)) / price_matrix.shift(-1) - 1.0
    
    adjusted_scores = scores_gbdt.copy()
    
    for sym in scores_gbdt:
        if sym not in returns_matrix.columns:
            continue
            
        y_train = returns_matrix.loc[history_dates, sym].values
        
        # 安全處理 NaN
        valid = ~np.isnan(y_train)
        if valid.sum() < 15:  # 至少需要 15 天的有效交易數據
            continue
            
        # 計算個股自身的時序滾動回報均值（動能因子）
        stock_momentum = float(np.nanmean(y_train[valid]))
        
        adjusted_scores[sym] = scores_gbdt[sym] + mult * stock_momentum
        
    return adjusted_scores
