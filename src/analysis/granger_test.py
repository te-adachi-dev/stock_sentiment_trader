from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def prepare_daily_data(
    stock_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
) -> pd.DataFrame:
    stock = stock_df.copy()
    if "Date" in stock.columns:
        stock["Date"] = pd.to_datetime(stock["Date"]).dt.tz_localize(None)
    stock["date"] = pd.to_datetime(stock["Date"]).dt.date
    daily_returns = stock.groupby("date")["Returns"].mean().reset_index()
    daily_returns.columns = ["date", "returns"]

    sent = sentiment_df.copy()
    sent["date"] = pd.to_datetime(sent["datetime"]).dt.date
    daily_sentiment = sent.groupby("date").agg(
        sentiment_score=("positive", "mean"),
    ).reset_index()
    # Use net sentiment: positive - negative average
    neg_avg = sent.groupby("date")["negative"].mean().reset_index()
    daily_sentiment["sentiment_score"] = daily_sentiment["sentiment_score"] - neg_avg["negative"]

    merged = pd.merge(daily_returns, daily_sentiment, on="date", how="left")
    merged["sentiment_score"] = merged["sentiment_score"].fillna(0.0)
    merged = merged.dropna(subset=["returns"])
    merged = merged.sort_values("date").reset_index(drop=True)

    return merged


def run_granger_test(
    data: pd.DataFrame,
    max_lag: int = 10,
    significance_level: float = 0.05,
) -> dict:
    results = {
        "sentiment_to_returns": {},
        "returns_to_sentiment": {},
    }

    n_obs = len(data)
    effective_max_lag = min(max_lag, n_obs // 3 - 1)
    if effective_max_lag < 1:
        logger.warning(f"Not enough data for Granger test (n={n_obs})")
        return results

    # Sentiment -> Returns
    try:
        test_data_sr = data[["returns", "sentiment_score"]].values
        gc_results = grangercausalitytests(test_data_sr, maxlag=effective_max_lag, verbose=False)
        for lag, result in gc_results.items():
            f_test = result[0]["ssr_ftest"]
            results["sentiment_to_returns"][lag] = {
                "f_stat": f_test[0],
                "p_value": f_test[1],
                "significant": f_test[1] < significance_level,
            }
    except Exception as e:
        logger.error(f"Granger test (sentiment->returns) failed: {e}")

    # Returns -> Sentiment
    try:
        test_data_rs = data[["sentiment_score", "returns"]].values
        gc_results = grangercausalitytests(test_data_rs, maxlag=effective_max_lag, verbose=False)
        for lag, result in gc_results.items():
            f_test = result[0]["ssr_ftest"]
            results["returns_to_sentiment"][lag] = {
                "f_stat": f_test[0],
                "p_value": f_test[1],
                "significant": f_test[1] < significance_level,
            }
    except Exception as e:
        logger.error(f"Granger test (returns->sentiment) failed: {e}")

    return results


def run_granger_for_all_symbols(
    stock_data: dict[str, pd.DataFrame],
    sentiment_data: dict[str, pd.DataFrame],
    max_lag: int = 10,
    significance_level: float = 0.05,
    save_dir: str = "data/results",
) -> pd.DataFrame:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    summary_lines: list[str] = []
    significant_symbols: list[str] = []

    summary_lines.append("=" * 70)
    summary_lines.append("Granger Causality Test Results Summary")
    summary_lines.append("=" * 70)
    summary_lines.append("")

    for symbol in stock_data:
        if symbol not in sentiment_data or sentiment_data[symbol].empty:
            logger.warning(f"No sentiment data for {symbol}, skipping Granger test")
            continue

        logger.info(f"Running Granger test for {symbol}...")
        daily_data = prepare_daily_data(stock_data[symbol], sentiment_data[symbol])

        if len(daily_data) < 10:
            logger.warning(f"Insufficient data for {symbol} ({len(daily_data)} rows)")
            continue

        results = run_granger_test(daily_data, max_lag=max_lag, significance_level=significance_level)

        summary_lines.append(f"--- {symbol} ---")
        symbol_has_significant = False

        for direction, label in [
            ("sentiment_to_returns", "Sentiment -> Returns"),
            ("returns_to_sentiment", "Returns -> Sentiment"),
        ]:
            summary_lines.append(f"  {label}:")
            for lag, vals in results.get(direction, {}).items():
                sig_marker = " *" if vals["significant"] else ""
                summary_lines.append(
                    f"    Lag {lag}: F={vals['f_stat']:.4f}, p={vals['p_value']:.4f}{sig_marker}"
                )
                all_rows.append({
                    "symbol": symbol,
                    "direction": direction,
                    "lag": lag,
                    "f_stat": vals["f_stat"],
                    "p_value": vals["p_value"],
                    "significant": vals["significant"],
                })
                if direction == "sentiment_to_returns" and vals["significant"]:
                    symbol_has_significant = True

        if symbol_has_significant:
            significant_symbols.append(symbol)
        summary_lines.append("")

    # Go/No-Go judgment
    summary_lines.append("=" * 70)
    summary_lines.append("Go / No-Go Judgment")
    summary_lines.append("=" * 70)
    summary_lines.append(
        f"Symbols with significant Sentiment -> Returns causality: "
        f"{len(significant_symbols)} ({', '.join(significant_symbols) if significant_symbols else 'None'})"
    )

    if len(significant_symbols) >= 3:
        summary_lines.append("Judgment: GO - Proceed to Phase 2")
        summary_lines.append(
            "Rationale: 3 or more symbols show statistically significant "
            "Granger causality from sentiment to returns."
        )
    else:
        summary_lines.append("Judgment: NO-GO - Reconsider approach")
        summary_lines.append(
            "Rationale: Fewer than 3 symbols show significant causality. "
            "Consider alternative data sources or analysis methods."
        )

    results_df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame(
        columns=["symbol", "direction", "lag", "f_stat", "p_value", "significant"]
    )
    results_df.to_csv(save_path / "granger_results.csv", index=False, encoding="utf-8")

    summary_text = "\n".join(summary_lines)
    with open(save_path / "granger_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    logger.info("Granger test results saved")
    return results_df
