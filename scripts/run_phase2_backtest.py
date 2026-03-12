#!/usr/bin/env python3

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.backtest.engine import BacktestEngine
from src.backtest.metrics import calculate_metrics
from src.backtest.report import generate_report
from src.data_collection.stock_price import fetch_stock_data
from src.data_collection.news_fetcher import fetch_all_news
from src.indicators.technical import calculate_indicators
from src.sentiment.finbert_analyzer import analyze_news_sentiment
from src.utils.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger("phase2_backtest")


def load_or_fetch_stock_data(
    symbols: list[str], config: dict,
) -> dict[str, pd.DataFrame]:
    stock_data: dict[str, pd.DataFrame] = {}
    raw_dir = Path("data/raw/stock_prices")

    for symbol in symbols:
        csv_path = raw_dir / f"{symbol}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            stock_data[symbol] = df
            logger.info(f"Loaded cached stock data for {symbol}: {len(df)} rows")
        else:
            logger.info(f"No cached data for {symbol}, fetching...")

    missing = [s for s in symbols if s not in stock_data]
    if missing:
        stock_config = config["data_collection"]["stock_price"]
        fetched = fetch_stock_data(
            symbols=missing,
            period=stock_config["period"],
            interval=stock_config["interval"],
        )
        stock_data.update(fetched)

    return stock_data


def load_or_fetch_sentiment_data(
    symbols: list[str], config: dict,
) -> dict[str, pd.DataFrame]:
    sentiment_data: dict[str, pd.DataFrame] = {}
    processed_dir = Path("data/processed/sentiment")

    for symbol in symbols:
        csv_path = processed_dir / f"{symbol}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            sentiment_data[symbol] = df
            logger.info(f"Loaded cached sentiment data for {symbol}: {len(df)} rows")

    missing = [s for s in symbols if s not in sentiment_data]
    if missing:
        logger.info(f"Missing sentiment data for {missing}, running pipeline...")
        api_keys = config.get("api_keys", {})
        finnhub_key = api_keys.get("finnhub")
        newsapi_key = api_keys.get("newsapi")
        news_config = config["data_collection"]["news"]

        news_data: dict[str, list[dict]] = {}
        if finnhub_key or newsapi_key:
            news_data = fetch_all_news(
                symbols=missing,
                finnhub_api_key=finnhub_key,
                newsapi_key=newsapi_key,
                lookback_days=news_config["lookback_days"],
                max_articles_per_symbol=news_config["max_articles_per_symbol"],
                calls_per_minute=config["api"]["finnhub"]["calls_per_minute"],
            )
        else:
            news_dir = Path("data/raw/news")
            if news_dir.exists():
                for jsonl_file in news_dir.glob("*.jsonl"):
                    sym = jsonl_file.stem
                    if sym in missing:
                        articles: list[dict] = []
                        with open(jsonl_file, encoding="utf-8") as f:
                            for line in f:
                                articles.append(json.loads(line.strip()))
                        news_data[sym] = articles

        if news_data:
            sentiment_config = config["sentiment"]
            fetched = analyze_news_sentiment(
                news_data=news_data,
                model_name=sentiment_config["model_name"],
                device=sentiment_config["device"],
                batch_size=sentiment_config["batch_size"],
                max_length=sentiment_config["max_length"],
            )
            sentiment_data.update(fetched)

    return sentiment_data


def main() -> None:
    logger.info("Starting Phase 2 Backtest Pipeline")
    logger.info("=" * 70)

    config = load_config()

    backtest_config = config.get("backtest", {})
    strategy_config = config.get("strategy", {})
    tier_a_symbols = backtest_config.get("tier_a_symbols", ["GOOGL", "TSLA", "AAPL"])

    logger.info(f"Tier A symbols: {tier_a_symbols}")

    logger.info("Step 1: Loading/fetching stock data...")
    stock_data = load_or_fetch_stock_data(tier_a_symbols, config)

    technical_data: dict[str, pd.DataFrame] = {}
    for symbol in stock_data:
        technical_data[symbol] = calculate_indicators(stock_data[symbol])

    logger.info("Step 2: Loading/fetching sentiment data...")
    sentiment_data = load_or_fetch_sentiment_data(tier_a_symbols, config)

    if not sentiment_data:
        logger.error("No sentiment data available. Cannot run backtest.")
        sys.exit(1)

    available_symbols = [
        s for s in tier_a_symbols
        if s in stock_data and s in sentiment_data
    ]
    logger.info(f"Available symbols for backtest: {available_symbols}")

    strategies = {
        "threshold": strategy_config.get("threshold", {
            "buy_threshold": 0.3, "sell_threshold": -0.3,
        }),
        "combo": strategy_config.get("combo", {
            "sentiment_buy": 0.2, "sentiment_sell": -0.2,
            "rsi_buy": 40, "rsi_sell": 60,
        }),
        "mean_reversion": strategy_config.get("mean_reversion", {
            "extreme_positive": 0.5, "extreme_negative": -0.5,
        }),
    }

    initial_capital = backtest_config.get("initial_capital", 100_000.0)
    commission = backtest_config.get("commission_per_share", 0.005)
    slippage = backtest_config.get("slippage_pct", 0.0005)
    risk_free_rate = backtest_config.get("risk_free_rate", 0.045)
    risk_per_trade = backtest_config.get("max_position_risk", 0.02)
    atr_stop = backtest_config.get("atr_stop_multiplier", 2.0)
    atr_profit = backtest_config.get("atr_profit_multiplier", 3.0)
    max_positions = backtest_config.get("max_concurrent_positions", 3)
    daily_loss = backtest_config.get("daily_loss_limit", 0.05)

    all_results: list = []
    buy_hold_returns: dict[str, pd.Series] = {}

    for symbol in available_symbols:
        sym_stock = stock_data[symbol]
        if "Returns" in sym_stock.columns:
            returns = sym_stock["Returns"].dropna()
            if "Date" in sym_stock.columns:
                dates = pd.to_datetime(sym_stock["Date"]).dt.tz_localize(None).dt.date
                buy_hold_returns[symbol] = pd.Series(
                    returns.values, index=dates.values[: len(returns)]
                )

    logger.info("Step 3: Running backtests...")
    for strategy_name, params in strategies.items():
        logger.info(f"Running backtest: {strategy_name}")

        price_subset = {s: stock_data[s] for s in available_symbols if s in stock_data}
        sentiment_subset = {s: sentiment_data[s] for s in available_symbols if s in sentiment_data}
        technical_subset = {s: technical_data[s] for s in available_symbols if s in technical_data}

        engine = BacktestEngine(
            price_data=price_subset,
            sentiment_data=sentiment_subset,
            technical_data=technical_subset,
            strategy=strategy_name,
            params=params,
            initial_capital=initial_capital,
            commission_per_share=commission,
            slippage_pct=slippage,
            risk_per_trade=risk_per_trade,
            atr_stop_multiplier=atr_stop,
            atr_profit_multiplier=atr_profit,
            max_concurrent_positions=max_positions,
            daily_loss_limit=daily_loss,
        )

        result = engine.run()

        metrics = calculate_metrics(
            equity_curve=result.equity_curve,
            daily_returns=result.daily_returns,
            trades=result.trades,
            risk_free_rate=risk_free_rate,
        )
        result.metrics = metrics
        all_results.append(result)

        logger.info(
            f"  {strategy_name}: "
            f"Return={metrics.get('total_return_pct', 0)}%, "
            f"Sharpe={metrics.get('sharpe_ratio', 0)}, "
            f"MaxDD={metrics.get('max_drawdown_pct', 0)}%, "
            f"Trades={metrics.get('total_trades', 0)}"
        )

    logger.info("Step 4: Generating report and visualizations...")
    report_text = generate_report(
        results=all_results,
        buy_hold_returns=buy_hold_returns,
    )

    logger.info("=" * 70)
    logger.info("Phase 2 Backtest Complete")
    logger.info("=" * 70)
    print("\n")
    print(report_text)
    print("\n")

    logger.info("Results saved to:")
    logger.info("  - data/results/backtest/backtest_report.txt")
    logger.info("  - data/results/backtest/*.png")


if __name__ == "__main__":
    main()
