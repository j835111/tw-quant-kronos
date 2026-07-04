from __future__ import annotations

import sqlite3


def twse_trading_days(db_path, benchmark_symbol: str = "^TWII", min_symbols: int = 500) -> set[str]:
    """Real TWSE trading days = dates with a benchmark row AND >= min_symbols stock rows."""
    conn = sqlite3.connect(db_path)
    try:
        bench = {
            r[0] for r in conn.execute(
                "SELECT DISTINCT date FROM daily_prices WHERE symbol = ?", (benchmark_symbol,)
            )
        }
        busy = {
            r[0] for r in conn.execute(
                "SELECT date FROM daily_prices GROUP BY date HAVING COUNT(*) >= ?", (min_symbols,)
            )
        }
    finally:
        conn.close()
    return bench & busy
