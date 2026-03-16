#!/usr/bin/env python3

import argparse
import csv
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tabulate import tabulate

from src.backtest.engine import run_pead_backtest
from src.backtest.metrics import calculate_metrics
from src.data_collection.stock_price import fetch_stock_data
from src.earnings.calendar import (
    fetch_company_earnings,
    get_sp500_symbols,
)
from src.earnings.screener import screen_pead_candidates
from src.earnings.yahoo_earnings import fetch_yahoo_earnings
from src.earnings.surprise import calculate_ear, calculate_sue
from src.strategy.pead import generate_pead_trades
from src.utils.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger("phase3_pead")


def _load_cached_earnings() -> pd.DataFrame:
    cache_dir = Path("data/raw/earnings")
    if not cache_dir.exists():
        return pd.DataFrame()

    calendar_path = cache_dir / "earnings_calendar.csv"
    if calendar_path.exists():
        df = pd.read_csv(calendar_path)
        logger.info(f"Loaded cached earnings calendar: {len(df)} records")
        return df

    csvs = list(cache_dir.glob("company_earnings_*.csv"))
    if csvs:
        dfs = [pd.read_csv(p) for p in csvs]
        combined = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded cached company earnings: {len(combined)} records from {len(csvs)} files")
        return combined

    return pd.DataFrame()


def _fetch_all_company_earnings(
    symbols: list[str],
    api_key: str,
    lookback_quarters: int,
) -> pd.DataFrame:
    all_dfs: list[pd.DataFrame] = []
    total = len(symbols)

    for i, symbol in enumerate(symbols):
        if (i + 1) % 20 == 0 or i == 0:
            logger.info(f"Fetching company earnings: {i + 1}/{total}")

        df = fetch_company_earnings(symbol, api_key=api_key, limit=lookback_quarters)
        if not df.empty:
            all_dfs.append(df)

        time.sleep(1.1)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"Total company earnings records: {len(combined)}")
        return combined
    return pd.DataFrame()


def _generate_synthetic_earnings(
    symbols: list[str],
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    rng = np.random.RandomState(42)
    records: list[dict] = []

    quarter_months = [(1, 25), (4, 25), (7, 25), (10, 25)]

    for symbol in symbols:
        base_eps = rng.uniform(0.5, 5.0)
        for year in range(start_year, end_year + 1):
            for month, day in quarter_months:
                try:
                    earn_date = pd.Timestamp(year=year, month=month, day=day)
                except ValueError:
                    continue

                if earn_date > pd.Timestamp.now():
                    continue

                eps_growth = rng.normal(0.05, 0.15)
                base_eps *= (1 + eps_growth)
                eps_actual = round(base_eps, 2)

                estimate_error = rng.normal(0, 0.1)
                eps_estimate = round(eps_actual * (1 - estimate_error), 2)

                if eps_estimate == 0:
                    eps_estimate = 0.01

                surprise_pct = (eps_actual - eps_estimate) / abs(eps_estimate) * 100

                records.append({
                    "date": earn_date.strftime("%Y-%m-%d"),
                    "symbol": symbol,
                    "eps_actual": eps_actual,
                    "eps_estimate": eps_estimate,
                    "revenue_actual": None,
                    "revenue_estimate": None,
                    "hour": "amc",
                    "surprise_pct": round(surprise_pct, 2),
                })

    df = pd.DataFrame(records)
    save_dir = Path("data/raw/earnings")
    save_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_dir / "earnings_calendar.csv", index=False, encoding="utf-8")
    logger.info(f"Generated synthetic earnings data: {len(df)} records for {len(symbols)} symbols")
    return df


def _save_pead_figures(
    all_results: dict[int, dict],
    candidates: pd.DataFrame,
    save_dir: Path,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    for hp, data in all_results.items():
        result = data["result"]
        if result.equity_curve.empty:
            continue

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(result.equity_curve.index, result.equity_curve.values, linewidth=1.5)
        ax.set_title(f"PEAD Equity Curve - {hp} Day Holding Period")
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value ($)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_dir / f"pead_equity_curve_{hp}d.png", dpi=150)
        plt.close(fig)

    all_trades: list[dict] = []
    for hp, data in all_results.items():
        for t in data["trades"]:
            t_copy = dict(t)
            t_copy["holding_period_config"] = hp
            all_trades.append(t_copy)

    if all_trades:
        returns = [t["return_pct"] for t in all_trades]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(returns, bins=40, edgecolor="black", alpha=0.7)
        ax.axvline(x=0, color="red", linestyle="--")
        ax.set_title("PEAD Trade Return Distribution (All Holding Periods)")
        ax.set_xlabel("Return (%)")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_dir / "pead_trade_distribution.png", dpi=150)
        plt.close(fig)

    if not candidates.empty and "date" in candidates.columns:
        dates = pd.to_datetime(candidates["date"])
        monthly = dates.dt.to_period("M").value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.bar(range(len(monthly)), monthly.values, alpha=0.7)
        labels = [str(p) for p in monthly.index]
        step = max(1, len(labels) // 20)
        ax.set_xticks(range(0, len(labels), step))
        ax.set_xticklabels(labels[::step], rotation=45, ha="right")
        ax.set_title("PEAD Monthly Trade Candidates")
        ax.set_xlabel("Month")
        ax.set_ylabel("Number of Candidates")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_dir / "pead_monthly_trades.png", dpi=150)
        plt.close(fig)

    if not candidates.empty and "surprise_pct" in candidates.columns:
        best_hp = max(all_results.keys(), key=lambda k: len(all_results[k]["trades"]))
        best_trades = all_results[best_hp]["trades"]
        if best_trades:
            fig, ax = plt.subplots(figsize=(10, 6))
            surprises = [t["surprise_pct"] for t in best_trades]
            returns = [t["return_pct"] for t in best_trades]
            ax.scatter(surprises, returns, alpha=0.5, s=20)
            ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
            ax.set_title(f"EPS Surprise % vs Trade Return % ({best_hp}d)")
            ax.set_xlabel("EPS Surprise (%)")
            ax.set_ylabel("Trade Return (%)")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(save_dir / "pead_surprise_vs_return.png", dpi=150)
            plt.close(fig)

    logger.info(f"Saved PEAD figures to {save_dir}")


def _run_single_backtest(
    earnings_with_ear: pd.DataFrame,
    price_data: dict[str, pd.DataFrame],
    surprise_threshold: float,
    volume_spike: float,
    stop_loss: float,
    position_size: float,
    holding_period: int,
    entry_delay: int,
    max_positions: int,
    initial_capital: float,
    commission: float,
    slippage: float,
    risk_free_rate: float,
) -> dict[str, float]:
    candidates = screen_pead_candidates(
        earnings_with_ear, price_data,
        {
            "surprise_threshold_pct": surprise_threshold,
            "volume_spike_multiplier": volume_spike,
            "check_volume": volume_spike > 0 and bool(price_data),
        },
    )

    if candidates.empty:
        return {
            "surprise_threshold": surprise_threshold,
            "stop_loss": stop_loss,
            "position_size": position_size,
            "holding_period": holding_period,
            "total_return": 0.0,
            "sharpe": 0.0,
            "max_dd": 0.0,
            "win_rate": 0.0,
            "trades": 0,
            "profit_factor": 0.0,
        }

    pead_params = {
        "entry_delay_days": entry_delay,
        "stop_loss_pct": stop_loss,
        "max_positions": max_positions,
    }

    trades = generate_pead_trades(candidates, price_data, holding_period, pead_params)

    result = run_pead_backtest(
        trades=trades,
        initial_capital=initial_capital,
        position_size_pct=position_size,
        commission_per_share=commission,
        slippage_pct=slippage,
    )

    metrics = calculate_metrics(
        equity_curve=result.equity_curve,
        daily_returns=result.daily_returns,
        trades=result.trades,
        risk_free_rate=risk_free_rate,
    )

    pf = metrics.get("profit_factor", 0)
    if pf == "inf":
        pf = 999.99

    return {
        "surprise_threshold": surprise_threshold,
        "stop_loss": stop_loss,
        "position_size": position_size,
        "holding_period": holding_period,
        "total_return": metrics["total_return_pct"],
        "sharpe": metrics["sharpe_ratio"],
        "max_dd": metrics["max_drawdown_pct"],
        "win_rate": metrics["win_rate_pct"],
        "trades": metrics["total_trades"],
        "profit_factor": pf,
    }


def _run_sweep(
    earnings_with_ear: pd.DataFrame,
    price_data: dict[str, pd.DataFrame],
    config: dict,
) -> None:
    pead_config = config.get("pead", {})
    entry_delay = pead_config.get("entry_delay_days", 1)
    max_positions = pead_config.get("max_positions", 10)
    initial_capital = pead_config.get("backtest", {}).get("initial_capital", 100_000)
    commission = pead_config.get("backtest", {}).get("commission_per_share", 0.005)
    slippage = pead_config.get("backtest", {}).get("slippage_pct", 0.0005)
    risk_free_rate = config.get("backtest", {}).get("risk_free_rate", 0.045)

    surprise_values = [2.0, 3.0, 5.0, 7.0, 10.0]
    stop_loss_values = [0.03, 0.05, 0.08, 0.10]
    position_size_values = [0.03, 0.05, 0.10]
    holding_periods = [5, 10, 20, 40, 60]

    combinations = list(itertools.product(
        surprise_values, stop_loss_values, position_size_values, holding_periods
    ))
    total = len(combinations)
    logger.info(f"Starting parameter sweep: {total} combinations")

    results: list[dict[str, float]] = []

    for idx, (surprise, sl, ps, hp) in enumerate(combinations):
        if (idx + 1) % 50 == 0 or idx == 0:
            logger.info(f"Sweep progress: {idx + 1}/{total}")

        row = _run_single_backtest(
            earnings_with_ear=earnings_with_ear,
            price_data=price_data,
            surprise_threshold=surprise,
            volume_spike=0.0,
            stop_loss=sl,
            position_size=ps,
            holding_period=hp,
            entry_delay=entry_delay,
            max_positions=max_positions,
            initial_capital=initial_capital,
            commission=commission,
            slippage=slippage,
            risk_free_rate=risk_free_rate,
        )
        results.append(row)

    save_dir = Path("data/results/pead")
    save_dir.mkdir(parents=True, exist_ok=True)

    csv_path = save_dir / "sweep_results.csv"
    fieldnames = [
        "surprise_threshold", "stop_loss", "position_size", "holding_period",
        "total_return", "sharpe", "max_dd", "win_rate", "trades", "profit_factor",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"Saved sweep results: {csv_path}")

    go_results = [
        r for r in results
        if r["sharpe"] >= 0.8
        and r["max_dd"] <= 30.0
        and r["win_rate"] >= 55.0
        and r["trades"] >= 30
    ]

    best_path = save_dir / "sweep_best.txt"
    lines: list[str] = []

    if go_results:
        go_sorted = sorted(go_results, key=lambda r: r["sharpe"], reverse=True)
        lines.append(f"Go/No-Go criteria met: {len(go_sorted)} patterns")
        lines.append("")
        for r in go_sorted:
            lines.append(
                f"surprise={r['surprise_threshold']}, stop_loss={r['stop_loss']}, "
                f"pos_size={r['position_size']}, hp={r['holding_period']}d | "
                f"Sharpe={r['sharpe']}, MaxDD={r['max_dd']}%, "
                f"WinRate={r['win_rate']}%, Trades={int(r['trades'])}, "
                f"PF={r['profit_factor']}"
            )
    else:
        lines.append("No patterns met all Go/No-Go criteria.")
        lines.append("Top 10 by Sharpe ratio:")
        lines.append("")
        top10 = sorted(results, key=lambda r: r["sharpe"], reverse=True)[:10]
        for r in top10:
            lines.append(
                f"surprise={r['surprise_threshold']}, stop_loss={r['stop_loss']}, "
                f"pos_size={r['position_size']}, hp={r['holding_period']}d | "
                f"Sharpe={r['sharpe']}, MaxDD={r['max_dd']}%, "
                f"WinRate={r['win_rate']}%, Trades={int(r['trades'])}, "
                f"PF={r['profit_factor']}"
            )

    with open(best_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved sweep best: {best_path}")

    print("\n" + "=" * 70)
    print("Parameter Sweep - Top 5 by Sharpe")
    print("=" * 70)
    top5 = sorted(results, key=lambda r: r["sharpe"], reverse=True)[:5]
    headers = [
        "Surprise%", "StopLoss", "PosSize", "HP", "Return%",
        "Sharpe", "MaxDD%", "WinRate%", "Trades", "PF",
    ]
    rows = [
        [
            r["surprise_threshold"], r["stop_loss"], r["position_size"],
            int(r["holding_period"]), r["total_return"], r["sharpe"],
            r["max_dd"], r["win_rate"], int(r["trades"]), r["profit_factor"],
        ]
        for r in top5
    ]
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    if go_results:
        print(f"\nGo/No-Go: {len(go_results)} patterns meet all criteria.")
    else:
        print("\nGo/No-Go: No patterns meet all criteria.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 PEAD Strategy")
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    args = parser.parse_args()

    logger.info("Starting Phase 3 PEAD Strategy Pipeline")
    logger.info("=" * 70)

    config = load_config()
    pead_config = config.get("pead", {})
    api_keys = config.get("api_keys", {})
    finnhub_key = api_keys.get("finnhub")

    start_year = pead_config.get("backtest", {}).get("start_year", 2022)
    end_year = pead_config.get("backtest", {}).get("end_year", 2025)
    holding_periods = pead_config.get("holding_periods", [5, 10, 20, 40, 60])
    surprise_threshold = pead_config.get("surprise_threshold_pct", 3.0)
    volume_spike = pead_config.get("volume_spike_multiplier", 0.0)
    max_positions = pead_config.get("max_positions", 10)
    position_size = pead_config.get("position_size_pct", 0.05)
    stop_loss = pead_config.get("stop_loss_pct", 0.05)
    entry_delay = pead_config.get("entry_delay_days", 1)
    initial_capital = pead_config.get("backtest", {}).get("initial_capital", 100_000)
    commission = pead_config.get("backtest", {}).get("commission_per_share", 0.005)
    slippage = pead_config.get("backtest", {}).get("slippage_pct", 0.0005)
    risk_free_rate = config.get("backtest", {}).get("risk_free_rate", 0.045)

    logger.info("Step 1: Getting S&P 500 symbol universe...")
    symbols = get_sp500_symbols()
    if not symbols:
        logger.error("Failed to get symbol list")
        sys.exit(1)
    logger.info(f"Universe: {len(symbols)} symbols")

    logger.info("Step 2: Fetching earnings data...")
    earnings_df = _load_cached_earnings()

    if earnings_df.empty:
        source_dfs: list[pd.DataFrame] = []

        if finnhub_key:
            logger.info("Fetching per-company earnings data (Finnhub)...")
            finnhub_df = _fetch_all_company_earnings(
                symbols, finnhub_key,
                pead_config.get("lookback_quarters", 8),
            )
            if not finnhub_df.empty:
                finnhub_df["_source"] = "finnhub"
                source_dfs.append(finnhub_df)

        data_sources = pead_config.get("data_sources", ["finnhub", "yahoo"])
        if "yahoo" in data_sources:
            logger.info("Fetching per-company earnings data (Yahoo Finance)...")
            yahoo_df = fetch_yahoo_earnings(
                symbols,
                lookback_quarters=pead_config.get("lookback_quarters", 8),
            )
            if not yahoo_df.empty:
                yahoo_df["_source"] = "yahoo"
                source_dfs.append(yahoo_df)

        if source_dfs:
            merged = pd.concat(source_dfs, ignore_index=True)
            merged["_date_key"] = pd.to_datetime(merged["date"]).dt.date.astype(str)
            merged["_sort"] = merged["_source"].map({"finnhub": 0, "yahoo": 1}).fillna(2)
            merged = merged.sort_values("_sort").drop_duplicates(
                subset=["symbol", "_date_key"], keep="first"
            )
            merged = merged.drop(columns=["_source", "_date_key", "_sort"], errors="ignore")
            earnings_df = merged.reset_index(drop=True)
            logger.info(f"Merged earnings data: {len(earnings_df)} records")
        elif not finnhub_key:
            logger.warning("No API keys available. Generating synthetic earnings data for testing.")
            earnings_df = _generate_synthetic_earnings(symbols[:100], start_year, end_year)

    if earnings_df.empty:
        logger.error("No earnings data available. Cannot proceed.")
        sys.exit(1)

    earnings_in_universe = earnings_df[earnings_df["symbol"].isin(symbols)]
    logger.info(
        f"Earnings records in universe: {len(earnings_in_universe)} "
        f"({earnings_in_universe['symbol'].nunique()} symbols)"
    )

    logger.info("Step 3: Fetching stock price data for earnings symbols...")
    earn_symbols = earnings_in_universe["symbol"].unique().tolist()

    price_data: dict[str, pd.DataFrame] = {}
    raw_dir = Path("data/raw/stock_prices")
    cached_symbols = []
    fetch_symbols = []
    for sym in earn_symbols:
        csv_path = raw_dir / f"{sym}.csv"
        if csv_path.exists():
            price_data[sym] = pd.read_csv(csv_path)
            cached_symbols.append(sym)
        else:
            fetch_symbols.append(sym)

    if cached_symbols:
        logger.info(f"Loaded cached price data for {len(cached_symbols)} symbols")

    if fetch_symbols:
        batch_size = 50
        for i in range(0, len(fetch_symbols), batch_size):
            batch = fetch_symbols[i : i + batch_size]
            logger.info(f"Fetching prices: batch {i // batch_size + 1}, {len(batch)} symbols")
            fetched = fetch_stock_data(
                symbols=batch,
                period=f"{end_year - start_year + 2}y",
                interval="1d",
            )
            price_data.update(fetched)

    logger.info(f"Total price data: {len(price_data)} symbols")

    logger.info("Step 4: Calculating SUE and EAR...")
    earnings_with_sue = calculate_sue(earnings_in_universe)
    earnings_with_ear = calculate_ear(earnings_with_sue, price_data)

    if args.sweep:
        logger.info("Running parameter sweep mode...")
        _run_sweep(earnings_with_ear, price_data, config)
        return

    logger.info("Step 5: Screening PEAD candidates...")
    candidates = screen_pead_candidates(
        earnings_with_ear, price_data,
        {
            "surprise_threshold_pct": surprise_threshold,
            "volume_spike_multiplier": volume_spike,
            "check_volume": volume_spike > 0 and bool(price_data),
        },
    )
    logger.info(f"PEAD candidates: {len(candidates)}")

    if candidates.empty:
        logger.warning("No PEAD candidates found. Relaxing filters...")
        candidates = screen_pead_candidates(
            earnings_with_ear, price_data,
            {
                "surprise_threshold_pct": 0.0,
                "volume_spike_multiplier": 0.0,
                "check_volume": False,
            },
        )
        logger.info(f"Relaxed screening candidates: {len(candidates)}")

    if candidates.empty:
        logger.error("No candidates even with relaxed filters. Cannot proceed.")
        sys.exit(1)

    logger.info("Step 6: Running PEAD backtests...")
    all_results: dict[int, dict] = {}

    pead_params = {
        "entry_delay_days": entry_delay,
        "stop_loss_pct": stop_loss,
        "max_positions": max_positions,
    }

    for hp in holding_periods:
        logger.info(f"  Backtest: {hp}-day holding period...")
        trades = generate_pead_trades(candidates, price_data, hp, pead_params)

        result = run_pead_backtest(
            trades=trades,
            initial_capital=initial_capital,
            position_size_pct=position_size,
            commission_per_share=commission,
            slippage_pct=slippage,
        )

        metrics = calculate_metrics(
            equity_curve=result.equity_curve,
            daily_returns=result.daily_returns,
            trades=result.trades,
            risk_free_rate=risk_free_rate,
        )
        result.metrics = metrics

        all_results[hp] = {"result": result, "trades": trades, "metrics": metrics}

        logger.info(
            f"    {hp}d: Return={metrics['total_return_pct']}%, "
            f"Sharpe={metrics['sharpe_ratio']}, "
            f"MaxDD={metrics['max_drawdown_pct']}%, "
            f"WinRate={metrics['win_rate_pct']}%, "
            f"Trades={metrics['total_trades']}"
        )

    logger.info("Step 7: Generating report and figures...")
    save_dir = Path("data/results/pead")
    _save_pead_figures(all_results, candidates, save_dir)

    report_text = _generate_pead_report(all_results, save_dir)

    logger.info("=" * 70)
    logger.info("Phase 3 PEAD Backtest Complete")
    logger.info("=" * 70)
    print("\n")
    print(report_text)
    print("\n")

    logger.info("Results saved to:")
    logger.info("  - data/results/pead/pead_report.txt")
    logger.info("  - data/results/pead/*.png")


def _generate_pead_report(
    all_results: dict[int, dict],
    save_dir: Path,
) -> str:
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("PEAD Strategy Backtest Results")
    lines.append("=" * 70)
    lines.append("")

    headers = [
        "Holding Period", "Total Return%", "Sharpe", "Max DD%",
        "Win Rate%", "Trades", "Avg Return%", "Profit Factor",
    ]
    rows: list[list] = []
    go_candidates: list[dict] = []

    for hp in sorted(all_results.keys()):
        m = all_results[hp]["metrics"]
        pf = m.get("profit_factor", 0)
        pf_str = str(pf) if pf != "inf" else "inf"
        rows.append([
            f"{hp} day",
            m["total_return_pct"],
            m["sharpe_ratio"],
            m["max_drawdown_pct"],
            m["win_rate_pct"],
            m["total_trades"],
            m["avg_trade_return_pct"],
            pf_str,
        ])

        if (
            m["sharpe_ratio"] >= 0.8
            and m["max_drawdown_pct"] <= 30.0
            and m["win_rate_pct"] >= 55.0
            and m["total_trades"] >= 30
        ):
            go_candidates.append({"hp": hp, **m})

    lines.append(tabulate(rows, headers=headers, tablefmt="grid"))
    lines.append("")
    lines.append("=" * 70)
    lines.append("Go / No-Go Judgment")
    lines.append("=" * 70)

    if go_candidates:
        lines.append("Judgment: GO - Proceed to live paper trading")
        lines.append("")
        lines.append("Qualifying configurations:")
        for gc in go_candidates:
            lines.append(
                f"  - {gc['hp']}d: Sharpe={gc['sharpe_ratio']}, "
                f"MaxDD={gc['max_drawdown_pct']}%, "
                f"WinRate={gc['win_rate_pct']}%, "
                f"Trades={gc['total_trades']}"
            )
        lines.append("")
        lines.append(
            "Rationale: At least one holding period meets all criteria "
            "(Sharpe >= 0.8, Max DD <= 30%, Win Rate >= 55%, Trades >= 30)."
        )
    else:
        lines.append("Judgment: NO-GO - Parameter adjustment recommended")
        lines.append("")
        lines.append(
            "No holding period met all criteria "
            "(Sharpe >= 0.8, Max DD <= 30%, Win Rate >= 55%, Trades >= 30)."
        )
        lines.append("")
        best_hp = max(
            all_results.keys(),
            key=lambda k: all_results[k]["metrics"].get("sharpe_ratio", 0),
        )
        best_m = all_results[best_hp]["metrics"]
        lines.append("Closest configuration:")
        lines.append(
            f"  - {best_hp}d: Sharpe={best_m['sharpe_ratio']}, "
            f"MaxDD={best_m['max_drawdown_pct']}%, "
            f"WinRate={best_m['win_rate_pct']}%, "
            f"Trades={best_m['total_trades']}"
        )
        lines.append("")
        lines.append("Recommendations:")
        if best_m["total_trades"] < 30:
            lines.append("  - Lower surprise_threshold_pct to increase trade count")
        if best_m["win_rate_pct"] < 55:
            lines.append("  - Increase surprise_threshold_pct for higher quality signals")
        if best_m["max_drawdown_pct"] > 30:
            lines.append("  - Tighten stop_loss_pct to reduce drawdown")
        if best_m["sharpe_ratio"] < 0.8:
            lines.append("  - Test intermediate holding periods (e.g. 15, 30 days)")

    report_text = "\n".join(lines)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "pead_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    logger.info(f"Saved PEAD report: {save_dir / 'pead_report.txt'}")
    return report_text


if __name__ == "__main__":
    main()
