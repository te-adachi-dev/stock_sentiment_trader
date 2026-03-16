import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger("yahoo_earnings")


def fetch_yahoo_earnings(
    symbols: list[str],
    lookback_quarters: int = 8,
    save_dir: str = "data/raw/earnings",
) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed, skipping Yahoo earnings fetch")
        return pd.DataFrame()

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    all_dfs: list[pd.DataFrame] = []
    total = len(symbols)

    for i, symbol in enumerate(symbols):
        if (i + 1) % 20 == 0 or i == 0:
            logger.info(f"Fetching Yahoo earnings: {i + 1}/{total}")

        cache_file = save_path / f"yahoo_earnings_{symbol}.csv"
        if cache_file.exists():
            df = pd.read_csv(cache_file)
            if not df.empty:
                all_dfs.append(df)
                continue

        try:
            ticker = yf.Ticker(symbol)
            earnings_dates = ticker.get_earnings_dates(limit=lookback_quarters * 3)

            if earnings_dates is None or earnings_dates.empty:
                continue

            earnings_dates = earnings_dates.reset_index()

            date_col = None
            for col_name in ["Earnings Date", "index"]:
                if col_name in earnings_dates.columns:
                    date_col = col_name
                    break

            if date_col is None:
                date_col = earnings_dates.columns[0]

            reported_col = None
            estimate_col = None
            for col in earnings_dates.columns:
                col_lower = str(col).lower()
                if "reported" in col_lower or "actual" in col_lower:
                    reported_col = col
                elif "estimate" in col_lower:
                    estimate_col = col

            if reported_col is None or estimate_col is None:
                logger.warning(f"{symbol}: Could not identify EPS columns: {list(earnings_dates.columns)}")
                continue

            mask = earnings_dates[reported_col].notna()
            confirmed = earnings_dates[mask].copy()

            if confirmed.empty:
                continue

            confirmed = confirmed.head(lookback_quarters)

            records: list[dict[str, object]] = []
            for _, row in confirmed.iterrows():
                eps_actual = float(row[reported_col]) if pd.notna(row[reported_col]) else np.nan
                eps_estimate = float(row[estimate_col]) if pd.notna(row[estimate_col]) else np.nan

                if pd.isna(eps_actual):
                    continue

                surprise = eps_actual - eps_estimate if pd.notna(eps_estimate) else np.nan

                if pd.notna(eps_estimate) and abs(eps_estimate) > 0:
                    surprise_pct = (eps_actual - eps_estimate) / abs(eps_estimate) * 100
                else:
                    surprise_pct = np.nan

                earn_date = pd.Timestamp(row[date_col])
                records.append({
                    "date": earn_date.strftime("%Y-%m-%d"),
                    "symbol": symbol,
                    "eps_actual": round(eps_actual, 4),
                    "eps_estimate": round(eps_estimate, 4) if pd.notna(eps_estimate) else np.nan,
                    "surprise": round(surprise, 4) if pd.notna(surprise) else np.nan,
                    "surprise_pct": round(surprise_pct, 2) if pd.notna(surprise_pct) else np.nan,
                })

            if records:
                df = pd.DataFrame(records)
                df.to_csv(cache_file, index=False, encoding="utf-8")
                all_dfs.append(df)

        except Exception as e:
            logger.warning(f"{symbol}: Yahoo earnings fetch failed: {e}")

        time.sleep(0.5)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"Total Yahoo earnings records: {len(combined)} ({combined['symbol'].nunique()} symbols)")
        return combined

    return pd.DataFrame()
