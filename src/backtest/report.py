from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tabulate import tabulate

from src.backtest.engine import BacktestResult
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def generate_report(
    results: list[BacktestResult],
    buy_hold_returns: dict[str, pd.Series],
    save_dir: str = "data/results/backtest",
) -> str:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    figures_path = save_path

    for result in results:
        if result.equity_curve.empty:
            continue

        symbols_in_result = list({t["symbol"] for t in result.trades})
        symbol_label = "_".join(sorted(symbols_in_result)) if symbols_in_result else "none"

        _plot_equity_curve(
            result, buy_hold_returns, symbol_label,
            figures_path / f"equity_curve_{result.strategy_name}_{symbol_label}.png",
        )
        _plot_drawdown(
            result,
            figures_path / f"drawdown_{result.strategy_name}_{symbol_label}.png",
        )
        _plot_monthly_returns(
            result,
            figures_path / f"monthly_returns_{result.strategy_name}_{symbol_label}.png",
        )
        _plot_trade_distribution(
            result,
            figures_path / f"trade_distribution_{result.strategy_name}_{symbol_label}.png",
        )

    _plot_strategy_comparison(results, figures_path / "strategy_comparison.png")

    report_text = _generate_text_report(results, save_path / "backtest_report.txt")
    return report_text


def _plot_equity_curve(
    result: BacktestResult,
    buy_hold_returns: dict[str, pd.Series],
    symbol_label: str,
    filepath: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(result.equity_curve.index, result.equity_curve.values, label=result.strategy_name, linewidth=1.5)

    for symbol, bh_returns in buy_hold_returns.items():
        if bh_returns.empty:
            continue
        initial_val = result.equity_curve.iloc[0] if not result.equity_curve.empty else 100000
        bh_equity = initial_val * (1 + bh_returns).cumprod()
        common_dates = [d for d in result.equity_curve.index if d in bh_equity.index]
        if common_dates:
            ax.plot(common_dates, [bh_equity[d] for d in common_dates],
                    label=f"Buy&Hold ({symbol})", linestyle="--", alpha=0.7)

    ax.set_title(f"Equity Curve - {result.strategy_name}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    logger.info(f"Saved equity curve: {filepath}")


def _plot_drawdown(result: BacktestResult, filepath: Path) -> None:
    if result.daily_returns.empty:
        return
    cumulative = (1 + result.daily_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max * 100

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.5, color="red")
    ax.set_title(f"Drawdown - {result.strategy_name}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    logger.info(f"Saved drawdown chart: {filepath}")


def _plot_monthly_returns(result: BacktestResult, filepath: Path) -> None:
    if result.daily_returns.empty:
        return

    dr = result.daily_returns.copy()
    dr.index = pd.to_datetime(dr.index)
    monthly = dr.resample("ME").apply(lambda x: (1 + x).prod() - 1) * 100

    if monthly.empty:
        return

    years = sorted(set(monthly.index.year))
    months = list(range(1, 13))
    data = np.full((len(years), 12), np.nan)
    for i, year in enumerate(years):
        for j, month in enumerate(months):
            vals = monthly[(monthly.index.year == year) & (monthly.index.month == month)]
            if not vals.empty:
                data[i, j] = vals.iloc[0]

    fig, ax = plt.subplots(figsize=(14, max(3, len(years) + 1)))
    sns.heatmap(
        data, annot=True, fmt=".1f", center=0, cmap="RdYlGn",
        xticklabels=["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        yticklabels=years, ax=ax,
    )
    ax.set_title(f"Monthly Returns (%) - {result.strategy_name}")
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    logger.info(f"Saved monthly returns heatmap: {filepath}")


def _plot_trade_distribution(result: BacktestResult, filepath: Path) -> None:
    if not result.trades:
        return
    trade_returns = [
        (t["exit_price"] - t["entry_price"]) / t["entry_price"] * 100
        for t in result.trades
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(trade_returns, bins=30, edgecolor="black", alpha=0.7)
    ax.axvline(x=0, color="red", linestyle="--")
    ax.set_title(f"Trade Return Distribution - {result.strategy_name}")
    ax.set_xlabel("Return (%)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    logger.info(f"Saved trade distribution: {filepath}")


def _plot_strategy_comparison(results: list[BacktestResult], filepath: Path) -> None:
    valid_results = [r for r in results if r.metrics]
    if not valid_results:
        return

    labels = [r.strategy_name for r in valid_results]
    sharpe_vals = [r.metrics.get("sharpe_ratio", 0) for r in valid_results]
    max_dd_vals = [r.metrics.get("max_drawdown_pct", 0) for r in valid_results]
    total_ret_vals = [r.metrics.get("total_return_pct", 0) for r in valid_results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].bar(labels, sharpe_vals, color="steelblue")
    axes[0].set_title("Sharpe Ratio")
    axes[0].axhline(y=1.0, color="green", linestyle="--", label="Target (1.0)")
    axes[0].legend()

    axes[1].bar(labels, max_dd_vals, color="indianred")
    axes[1].set_title("Max Drawdown (%)")
    axes[1].axhline(y=20, color="green", linestyle="--", label="Limit (20%)")
    axes[1].legend()

    axes[2].bar(labels, total_ret_vals, color="seagreen")
    axes[2].set_title("Total Return (%)")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    logger.info(f"Saved strategy comparison: {filepath}")


def _generate_text_report(
    results: list[BacktestResult],
    filepath: Path,
) -> str:
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("Phase 2 Backtest Report")
    lines.append("=" * 80)
    lines.append("")

    table_rows: list[list] = []
    headers = [
        "Strategy", "Total Return%", "Annual Return%", "Sharpe",
        "Sortino", "Max DD%", "DD Days", "Win Rate%",
        "Profit Factor", "Trades", "Avg Return%", "Avg Hold Days",
    ]

    go_candidates: list[dict] = []

    for result in results:
        m = result.metrics
        if not m:
            continue
        pf = m["profit_factor"]
        pf_str = str(pf) if pf != "inf" else "inf"
        table_rows.append([
            result.strategy_name,
            m["total_return_pct"],
            m["annualized_return_pct"],
            m["sharpe_ratio"],
            m["sortino_ratio"],
            m["max_drawdown_pct"],
            m["max_drawdown_duration_days"],
            m["win_rate_pct"],
            pf_str,
            m["total_trades"],
            m["avg_trade_return_pct"],
            m["avg_holding_period_days"],
        ])

        sharpe = m["sharpe_ratio"]
        max_dd = m["max_drawdown_pct"]
        total_trades = m["total_trades"]
        if sharpe >= 1.0 and max_dd <= 20.0 and total_trades >= 20:
            go_candidates.append({
                "strategy": result.strategy_name,
                "sharpe": sharpe,
                "max_dd": max_dd,
                "trades": total_trades,
            })

    lines.append(tabulate(table_rows, headers=headers, tablefmt="grid"))
    lines.append("")

    lines.append("=" * 80)
    lines.append("Go / No-Go Judgment")
    lines.append("=" * 80)

    if go_candidates:
        lines.append("Judgment: GO - Proceed to Phase 3")
        lines.append("")
        lines.append("Qualifying strategies:")
        for gc in go_candidates:
            lines.append(
                f"  - {gc['strategy']}: Sharpe={gc['sharpe']}, "
                f"MaxDD={gc['max_dd']}%, Trades={gc['trades']}"
            )
        lines.append("")
        lines.append(
            "Rationale: At least one strategy meets the criteria "
            "(Sharpe >= 1.0, Max Drawdown <= 20%, Trades >= 20)."
        )
    else:
        lines.append("Judgment: NO-GO - Parameter adjustment recommended")
        lines.append("")
        lines.append("No strategy met all criteria (Sharpe >= 1.0, Max DD <= 20%, Trades >= 20).")
        lines.append("")
        lines.append("Recommendations:")
        best_sharpe = max(
            (r for r in results if r.metrics),
            key=lambda r: r.metrics.get("sharpe_ratio", 0),
            default=None,
        )
        if best_sharpe and best_sharpe.metrics:
            lines.append(
                f"  - Best Sharpe: {best_sharpe.strategy_name} "
                f"({best_sharpe.metrics['sharpe_ratio']})"
            )
            if best_sharpe.metrics["sharpe_ratio"] > 0:
                lines.append("  - Consider tightening entry thresholds")
                lines.append("  - Consider adjusting stop-loss/take-profit multipliers")
                lines.append("  - Consider adding more signal filters")

    report_text = "\n".join(lines)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report_text)

    logger.info(f"Saved backtest report: {filepath}")
    return report_text
