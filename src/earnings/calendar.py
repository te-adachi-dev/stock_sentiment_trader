import time
from pathlib import Path

import finnhub
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def get_sp500_symbols(fallback_path: str = "configs/sp500_symbols.txt") -> list[str]:
    try:
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        )
        if tables:
            df = tables[0]
            symbols = df["Symbol"].str.replace(".", "-", regex=False).tolist()
            logger.info(f"Fetched {len(symbols)} S&P 500 symbols from Wikipedia")
            return symbols
    except Exception as e:
        logger.warning(f"Wikipedia scraping failed: {e}")

    path = Path(fallback_path)
    if path.exists():
        symbols = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        logger.info(f"Loaded {len(symbols)} symbols from fallback file {fallback_path}")
        return symbols

    logger.error("No symbol source available")
    return []


def fetch_earnings_calendar(
    start_date: str,
    end_date: str,
    api_key: str,
) -> pd.DataFrame:
    client = finnhub.Client(api_key=api_key)

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    all_records: list[dict] = []

    current = start
    while current < end:
        chunk_end = min(current + pd.Timedelta(days=30), end)
        from_str = current.strftime("%Y-%m-%d")
        to_str = chunk_end.strftime("%Y-%m-%d")

        try:
            result = client.earnings_calendar(
                _from=from_str, to=to_str, symbol="", international=False
            )
            earnings = result.get("earningsCalendar", [])
            for e in earnings:
                if e.get("epsActual") is None:
                    continue
                all_records.append({
                    "date": e.get("date", ""),
                    "symbol": e.get("symbol", ""),
                    "eps_actual": e.get("epsActual"),
                    "eps_estimate": e.get("epsEstimate"),
                    "revenue_actual": e.get("revenueActual"),
                    "revenue_estimate": e.get("revenueEstimate"),
                    "hour": e.get("hour", ""),
                })
            logger.info(f"Earnings calendar {from_str} to {to_str}: {len(earnings)} records")
        except Exception as e:
            logger.error(f"Failed to fetch earnings calendar {from_str}-{to_str}: {e}")

        time.sleep(1.1)
        current = chunk_end + pd.Timedelta(days=1)

    df = pd.DataFrame(all_records)
    if not df.empty:
        save_dir = Path("data/raw/earnings")
        save_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_dir / "earnings_calendar.csv", index=False, encoding="utf-8")
        logger.info(f"Saved earnings calendar: {len(df)} records")
    return df


def fetch_company_earnings(
    symbol: str,
    api_key: str,
    limit: int = 8,
) -> pd.DataFrame:
    cache_path = Path(f"data/raw/earnings/company_earnings_{symbol}.csv")
    if cache_path.exists():
        df = pd.read_csv(cache_path)
        logger.debug(f"Using cached earnings for {symbol}")
        return df

    client = finnhub.Client(api_key=api_key)
    try:
        records = client.company_earnings(symbol, limit=limit)
    except Exception as e:
        logger.error(f"Failed to fetch earnings for {symbol}: {e}")
        return pd.DataFrame()

    if not records:
        return pd.DataFrame()

    rows: list[dict] = []
    for r in records:
        rows.append({
            "date": r.get("period", ""),
            "eps_actual": r.get("actual"),
            "eps_estimate": r.get("estimate"),
            "surprise": r.get("surprise"),
            "surprise_pct": r.get("surprisePercent"),
            "symbol": symbol,
        })

    df = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False, encoding="utf-8")
    return df
