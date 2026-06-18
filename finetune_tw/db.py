import sqlite3
from pathlib import Path
import pandas as pd


def init_db(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS stocks (
                symbol     TEXT PRIMARY KEY,
                name       TEXT,
                first_date TEXT,
                last_date  TEXT
            );
            CREATE TABLE IF NOT EXISTS daily_prices (
                symbol TEXT,
                date   TEXT,
                open   REAL,
                high   REAL,
                low    REAL,
                close  REAL,
                volume REAL,
                amount REAL,
                PRIMARY KEY (symbol, date)
            );
            CREATE INDEX IF NOT EXISTS idx_date ON daily_prices(date);
        """)


def upsert_prices(db_path: str, symbol: str, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    rows = df.assign(symbol=symbol)[
        ["symbol", "date", "open", "high", "low", "close", "volume", "amount"]
    ].values.tolist()
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO daily_prices VALUES (?,?,?,?,?,?,?,?)", rows
        )
        # Create stock record only if it doesn't exist (preserves first_date)
        conn.execute(
            "INSERT OR IGNORE INTO stocks (symbol, first_date, last_date) VALUES (?,?,?)",
            (symbol, df["date"].min(), df["date"].max()),
        )
        # Always update last_date to the latest date seen
        conn.execute(
            "UPDATE stocks SET last_date = ? WHERE symbol = ? AND last_date < ?",
            (df["date"].max(), symbol, df["date"].max()),
        )
    return len(rows)


def query_symbol(
    db_path: str, symbol: str, start: str = None, end: str = None
) -> pd.DataFrame:
    q = "SELECT date,open,high,low,close,volume,amount FROM daily_prices WHERE symbol=?"
    params: list = [symbol]
    if start:
        q += " AND date>=?"; params.append(start)
    if end:
        q += " AND date<=?"; params.append(end)
    q += " ORDER BY date"
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql(q, conn, params=params)


def list_symbols(db_path: str) -> list:
    with sqlite3.connect(db_path) as conn:
        return [r[0] for r in conn.execute("SELECT symbol FROM stocks ORDER BY symbol")]


def get_last_date(db_path: str, symbol: str) -> "str | None":
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT last_date FROM stocks WHERE symbol=?", (symbol,)
        ).fetchone()
    return row[0] if row else None
