from __future__ import annotations
import re
import requests
import pandas as pd
import yfinance as yf

TWSE_LISTING_URL = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_AVG_ALL"

# 4-digit ETF codes worth including alongside common stocks
_CORE_ETFS = {"0050", "0051", "0052", "0053", "0055", "0056", "0057", "0061"}


def get_twse_symbol_list() -> list[str]:
    resp = requests.get(TWSE_LISTING_URL, timeout=15)
    resp.raise_for_status()
    symbols = []
    for item in resp.json():
        code = item.get("Code", "")
        if not code:
            continue
        # Keep regular stocks (4-digit, 1–9 prefix) and a curated ETF set.
        # Exclude warrants/structured products (6-digit 0-prefix) and
        # anything with letter suffixes (subscription-period ETFs, etc.).
        if re.fullmatch(r"[0-9]+", code):
            if re.fullmatch(r"[1-9][0-9]{3}", code) or code in _CORE_ETFS:
                symbols.append(f"{code}.TW")
    return symbols


def fetch_symbol(
    symbol: str, start: str = "2015-01-01", end: str | None = None
) -> pd.DataFrame | None:
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=start, end=end, auto_adjust=True)
        if hist.empty:
            return None
        # Extract date from index regardless of its name
        dates = pd.to_datetime(hist.index)
        if dates.tz is not None:
            dates = dates.tz_localize(None)
        date_strs = dates.strftime("%Y-%m-%d")

        avg_price = (hist["Open"] + hist["High"] + hist["Low"] + hist["Close"]) / 4.0
        df = pd.DataFrame({
            "date": date_strs,
            "open": hist["Open"].values,
            "high": hist["High"].values,
            "low": hist["Low"].values,
            "close": hist["Close"].values,
            "volume": hist["Volume"].values,
            "amount": (hist["Volume"] * avg_price).values,
        })
        return df.reset_index(drop=True)
    except Exception:
        return None
