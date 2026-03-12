import numpy as np
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def calculate_metrics(
    equity_curve: pd.Series,
    daily_returns: pd.Series,
    trades: list[dict],
    risk_free_rate: float = 0.045,
) -> dict:
    if equity_curve.empty or daily_returns.empty:
        return _empty_metrics()

    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
    trading_days = len(equity_curve)
    annualized_return = ((1 + total_return / 100) ** (252 / max(trading_days, 1)) - 1) * 100

    daily_rf = risk_free_rate / 252
    excess_returns = daily_returns - daily_rf
    if daily_returns.std() > 0:
        sharpe_ratio = float(np.sqrt(252) * excess_returns.mean() / daily_returns.std())
    else:
        sharpe_ratio = 0.0

    downside_returns = daily_returns[daily_returns < 0]
    if len(downside_returns) > 0 and downside_returns.std() > 0:
        sortino_ratio = float(np.sqrt(252) * excess_returns.mean() / downside_returns.std())
    else:
        sortino_ratio = 0.0

    cumulative = (1 + daily_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = float(abs(drawdown.min()) * 100) if len(drawdown) > 0 else 0.0

    dd_duration = 0
    max_dd_duration = 0
    for dd_val in drawdown:
        if dd_val < 0:
            dd_duration += 1
            max_dd_duration = max(max_dd_duration, dd_duration)
        else:
            dd_duration = 0

    if trades:
        winning_trades = [t for t in trades if t["pnl"] > 0]
        losing_trades = [t for t in trades if t["pnl"] <= 0]
        win_rate = len(winning_trades) / len(trades) * 100

        gross_profit = sum(t["pnl"] for t in winning_trades) if winning_trades else 0.0
        gross_loss = abs(sum(t["pnl"] for t in losing_trades)) if losing_trades else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

        trade_returns = []
        holding_periods = []
        for t in trades:
            trade_ret = (t["exit_price"] - t["entry_price"]) / t["entry_price"] * 100
            trade_returns.append(trade_ret)
            entry_date = pd.Timestamp(t["entry_date"])
            exit_date = pd.Timestamp(t["exit_date"])
            holding_days = (exit_date - entry_date).days
            holding_periods.append(max(holding_days, 1))

        avg_trade_return = float(np.mean(trade_returns))
        avg_holding_period = float(np.mean(holding_periods))
    else:
        win_rate = 0.0
        profit_factor = 0.0
        avg_trade_return = 0.0
        avg_holding_period = 0.0

    metrics = {
        "total_return_pct": round(total_return, 2),
        "annualized_return_pct": round(annualized_return, 2),
        "sharpe_ratio": round(sharpe_ratio, 2),
        "sortino_ratio": round(sortino_ratio, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "max_drawdown_duration_days": max_dd_duration,
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "inf",
        "total_trades": len(trades),
        "avg_trade_return_pct": round(avg_trade_return, 2),
        "avg_holding_period_days": round(avg_holding_period, 1),
    }

    logger.info(
        f"Metrics: Sharpe={metrics['sharpe_ratio']}, "
        f"MaxDD={metrics['max_drawdown_pct']}%, "
        f"WinRate={metrics['win_rate_pct']}%"
    )
    return metrics


def _empty_metrics() -> dict:
    return {
        "total_return_pct": 0.0,
        "annualized_return_pct": 0.0,
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
        "max_drawdown_pct": 0.0,
        "max_drawdown_duration_days": 0,
        "win_rate_pct": 0.0,
        "profit_factor": 0.0,
        "total_trades": 0,
        "avg_trade_return_pct": 0.0,
        "avg_holding_period_days": 0.0,
    }
