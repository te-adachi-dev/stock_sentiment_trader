#!/usr/bin/env python3
"""Phase 1 validation pipeline: sentiment-price causality analysis."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.correlation import run_correlation_analysis
from src.analysis.granger_test import run_granger_for_all_symbols
from src.data_collection.news_fetcher import fetch_all_news
from src.data_collection.stock_price import fetch_stock_data
from src.indicators.technical import calculate_indicators
from src.sentiment.finbert_analyzer import analyze_news_sentiment
from src.utils.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger("phase1_validation")


def main() -> None:
    logger.info("Starting Phase 1 Validation Pipeline")
    logger.info("=" * 70)

    # 1. Load config
    logger.info("Step 1: Loading configuration...")
    config = load_config()
    symbols = config["target_symbols"]
    logger.info(f"Target symbols: {symbols}")

    # 2. Fetch stock data
    logger.info("Step 2: Fetching stock price data...")
    stock_config = config["data_collection"]["stock_price"]
    stock_data = fetch_stock_data(
        symbols=symbols,
        period=stock_config["period"],
        interval=stock_config["interval"],
    )

    if not stock_data:
        logger.error("No stock data fetched. Aborting.")
        sys.exit(1)

    # Calculate technical indicators
    for symbol in stock_data:
        stock_data[symbol] = calculate_indicators(stock_data[symbol])

    # 3. Fetch news data
    logger.info("Step 3: Fetching news data...")
    news_config = config["data_collection"]["news"]
    api_keys = config.get("api_keys", {})
    finnhub_key = api_keys.get("finnhub")
    newsapi_key = api_keys.get("newsapi")

    news_data: dict[str, list[dict]] = {}
    if finnhub_key or newsapi_key:
        news_data = fetch_all_news(
            symbols=symbols,
            finnhub_api_key=finnhub_key,
            newsapi_key=newsapi_key,
            lookback_days=news_config["lookback_days"],
            max_articles_per_symbol=news_config["max_articles_per_symbol"],
            calls_per_minute=config["api"]["finnhub"]["calls_per_minute"],
        )
    else:
        logger.warning(
            "No API keys available. Loading existing news data if present..."
        )
        news_dir = Path("data/raw/news")
        if news_dir.exists():
            for jsonl_file in news_dir.glob("*.jsonl"):
                symbol = jsonl_file.stem
                if symbol in symbols:
                    articles = []
                    with open(jsonl_file, encoding="utf-8") as f:
                        for line in f:
                            articles.append(json.loads(line.strip()))
                    news_data[symbol] = articles
                    logger.info(f"Loaded {len(articles)} cached articles for {symbol}")

    if not news_data:
        logger.error(
            "No news data available. Set FINNHUB_API_KEY in .env or provide cached data in data/raw/news/."
        )
        sys.exit(1)

    # 4. Run sentiment analysis
    logger.info("Step 4: Running FinBERT sentiment analysis...")
    sentiment_config = config["sentiment"]
    sentiment_data = analyze_news_sentiment(
        news_data=news_data,
        model_name=sentiment_config["model_name"],
        device=sentiment_config["device"],
        batch_size=sentiment_config["batch_size"],
        max_length=sentiment_config["max_length"],
    )

    if not sentiment_data:
        logger.error("No sentiment data produced. Aborting.")
        sys.exit(1)

    # 5. Granger causality test
    logger.info("Step 5: Running Granger causality tests...")
    analysis_config = config["analysis"]
    granger_df = run_granger_for_all_symbols(
        stock_data=stock_data,
        sentiment_data=sentiment_data,
        max_lag=analysis_config["granger_max_lag"],
        significance_level=analysis_config["significance_level"],
    )

    # 6. Correlation analysis
    logger.info("Step 6: Running correlation analysis and generating figures...")
    correlations = run_correlation_analysis(
        stock_data=stock_data,
        sentiment_data=sentiment_data,
        max_lag=5,
    )

    # 7. Print summary
    logger.info("=" * 70)
    logger.info("Phase 1 Validation Complete")
    logger.info("=" * 70)

    summary_path = Path("data/results/granger_summary.txt")
    if summary_path.exists():
        print("\n")
        print(summary_path.read_text(encoding="utf-8"))
        print("\n")

    logger.info("Results saved to:")
    logger.info("  - data/results/granger_results.csv")
    logger.info("  - data/results/granger_summary.txt")
    logger.info("  - data/results/figures/correlation_heatmap.png")
    logger.info("  - data/results/figures/sentiment_return_scatter.png")
    logger.info("  - data/results/figures/lagged_correlation.png")

    # Print correlation summary
    if correlations:
        print("\nCorrelation Summary (Pearson, Lag 0-5):")
        print("-" * 50)
        for symbol, corrs in sorted(correlations.items()):
            lag_values = [
                f"L{lag}={val:.3f}" for lag, val in sorted(corrs["pearson"].items())
            ]
            print(f"  {symbol}: {', '.join(lag_values)}")


if __name__ == "__main__":
    main()
