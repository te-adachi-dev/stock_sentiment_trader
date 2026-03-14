from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src.analysis.granger_test import prepare_daily_data
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def compute_lagged_correlations(
    daily_data: pd.DataFrame,
    max_lag: int = 5,
) -> dict[str, dict[int, float]]:
    results: dict[str, dict[int, float]] = {"pearson": {}, "spearman": {}}

    sentiment = daily_data["sentiment_score"].values
    returns = daily_data["returns"].values

    for lag in range(0, max_lag + 1):
        if lag == 0:
            s = sentiment
            r = returns
        else:
            s = sentiment[:-lag]
            r = returns[lag:]

        if len(s) < 5:
            continue

        pearson_r, _ = stats.pearsonr(s, r)
        spearman_r, _ = stats.spearmanr(s, r)
        results["pearson"][lag] = pearson_r
        results["spearman"][lag] = spearman_r

    return results


def run_correlation_analysis(
    stock_data: dict[str, pd.DataFrame],
    sentiment_data: dict[str, pd.DataFrame],
    max_lag: int = 5,
    save_dir: str = "data/results/figures",
) -> dict[str, dict]:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    all_correlations: dict[str, dict] = {}

    for symbol in stock_data:
        if symbol not in sentiment_data or sentiment_data[symbol].empty:
            continue

        daily_data = prepare_daily_data(stock_data[symbol], sentiment_data[symbol])
        if len(daily_data) < 10:
            logger.warning(f"Insufficient data for {symbol} correlation analysis")
            continue

        corrs = compute_lagged_correlations(daily_data, max_lag=max_lag)
        all_correlations[symbol] = corrs

    if not all_correlations:
        logger.warning("No data available for correlation analysis")
        return all_correlations

    _plot_correlation_heatmap(all_correlations, max_lag, save_path)
    _plot_scatter(stock_data, sentiment_data, all_correlations, save_path)
    _plot_lagged_correlation_lines(all_correlations, max_lag, save_path)

    logger.info(f"Correlation analysis figures saved to {save_path}")
    return all_correlations


def _plot_correlation_heatmap(
    all_correlations: dict[str, dict],
    max_lag: int,
    save_path: Path,
) -> None:
    symbols = sorted(all_correlations.keys())
    lags = list(range(0, max_lag + 1))

    heatmap_data = []
    for symbol in symbols:
        row = []
        for lag in lags:
            val = all_correlations[symbol]["pearson"].get(lag, np.nan)
            row.append(val)
        heatmap_data.append(row)

    df_heat = pd.DataFrame(heatmap_data, index=symbols, columns=[f"Lag {l}" for l in lags])

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df_heat, annot=True, fmt=".3f", cmap="RdYlGn", center=0, ax=ax)
    ax.set_title("Pearson Correlation: Sentiment Score vs Future Returns")
    ax.set_ylabel("Symbol")
    ax.set_xlabel("Lag (days)")
    plt.tight_layout()
    fig.savefig(save_path / "correlation_heatmap.png", dpi=150)
    plt.close(fig)


def _plot_scatter(
    stock_data: dict[str, pd.DataFrame],
    sentiment_data: dict[str, pd.DataFrame],
    all_correlations: dict[str, dict],
    save_path: Path,
) -> None:
    lag1_corrs = {}
    for symbol, corrs in all_correlations.items():
        if 1 in corrs["pearson"]:
            lag1_corrs[symbol] = abs(corrs["pearson"][1])

    top_symbols = sorted(lag1_corrs, key=lag1_corrs.get, reverse=True)[:5]

    if not top_symbols:
        return

    fig, axes = plt.subplots(1, len(top_symbols), figsize=(4 * len(top_symbols), 4))
    if len(top_symbols) == 1:
        axes = [axes]

    for ax, symbol in zip(axes, top_symbols):
        daily_data = prepare_daily_data(stock_data[symbol], sentiment_data[symbol])
        sentiment = daily_data["sentiment_score"].values[:-1]
        next_returns = daily_data["returns"].values[1:]

        ax.scatter(sentiment, next_returns, alpha=0.5, s=10)
        ax.set_title(f"{symbol} (r={all_correlations[symbol]['pearson'].get(1, 0):.3f})")
        ax.set_xlabel("Sentiment Score (t)")
        ax.set_ylabel("Return (t+1)")
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
        ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    fig.savefig(save_path / "sentiment_return_scatter.png", dpi=150)
    plt.close(fig)


def _plot_lagged_correlation_lines(
    all_correlations: dict[str, dict],
    max_lag: int,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    lags = list(range(0, max_lag + 1))

    for symbol, corrs in all_correlations.items():
        values = [corrs["pearson"].get(lag, np.nan) for lag in lags]
        ax.plot(lags, values, marker="o", label=symbol)

    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Pearson Correlation")
    ax.set_title("Lagged Correlation: Sentiment Score vs Returns")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(save_path / "lagged_correlation.png", dpi=150)
    plt.close(fig)
