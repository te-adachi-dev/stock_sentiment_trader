from pathlib import Path

import pandas as pd
import yfinance as yf

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def fetch_stock_data(
    symbols: list[str],
    period: str = "1y",
    interval: str = "1d",
    save_dir: str = "data/raw/stock_prices",
) -> dict[str, pd.DataFrame]:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    results: dict[str, pd.DataFrame] = {}
    failed: list[str] = []

    for symbol in symbols:
        try:
            logger.info(f"Fetching stock data for {symbol}...")
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)

            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                failed.append(symbol)
                continue

            df = df.reset_index()
            df = df.rename(columns={"Date": "Date"})
            df["Returns"] = df["Close"].pct_change()
            df["Symbol"] = symbol

            csv_path = save_path / f"{symbol}.csv"
            df.to_csv(csv_path, index=False, encoding="utf-8")

            results[symbol] = df
            logger.info(f"Fetched {len(df)} rows for {symbol}")
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            failed.append(symbol)

    logger.info(
        f"Stock data fetch complete. Success: {len(results)}, Failed: {len(failed)}"
    )
    if failed:
        logger.info(f"Failed symbols: {failed}")

    return results
