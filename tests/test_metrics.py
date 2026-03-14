import numpy as np
import pandas as pd
import pytest

from src.backtest.metrics import calculate_metrics


class TestMetrics:
    def test_sharpe_ratio_calculation(self) -> None:
        np.random.seed(42)
        daily_returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        equity_curve = pd.Series(
            100000 * (1 + daily_returns).cumprod(),
            index=pd.date_range("2024-01-01", periods=252, freq="D"),
        )
        trades = [
            {"entry_price": 100, "exit_price": 105, "shares": 10,
             "pnl": 50, "entry_date": "2024-01-05", "exit_date": "2024-01-10",
             "symbol": "ITEM", "exit_reason": "signal_sell"},
        ]
        metrics = calculate_metrics(equity_curve, daily_returns, trades, risk_free_rate=0.045)

        assert isinstance(metrics["sharpe_ratio"], float)
        assert metrics["sharpe_ratio"] != 0.0

    def test_max_drawdown_known_values(self) -> None:
        values = [100, 110, 105, 95, 90, 100, 105]
        daily_returns = pd.Series([
            (values[i] - values[i - 1]) / values[i - 1]
            for i in range(1, len(values))
        ])
        equity_curve = pd.Series(
            values,
            index=pd.date_range("2024-01-01", periods=len(values), freq="D"),
        )
        trades: list[dict] = []
        metrics = calculate_metrics(equity_curve, daily_returns, trades)

        expected_dd = (110 - 90) / 110 * 100
        assert abs(metrics["max_drawdown_pct"] - expected_dd) < 1.0

    def test_win_rate_calculation(self) -> None:
        trades = [
            {"entry_price": 100, "exit_price": 110, "shares": 10,
             "pnl": 100, "entry_date": "2024-01-01", "exit_date": "2024-01-05",
             "symbol": "ITEM_A", "exit_reason": "take_profit"},
            {"entry_price": 100, "exit_price": 95, "shares": 10,
             "pnl": -50, "entry_date": "2024-01-06", "exit_date": "2024-01-10",
             "symbol": "ITEM_B", "exit_reason": "stop_loss"},
            {"entry_price": 100, "exit_price": 108, "shares": 10,
             "pnl": 80, "entry_date": "2024-01-11", "exit_date": "2024-01-15",
             "symbol": "ITEM_C", "exit_reason": "signal_sell"},
        ]
        equity_curve = pd.Series(
            [100000, 100100, 100050, 100130],
            index=pd.date_range("2024-01-01", periods=4, freq="D"),
        )
        daily_returns = equity_curve.pct_change().dropna()

        metrics = calculate_metrics(equity_curve, daily_returns, trades)
        assert abs(metrics["win_rate_pct"] - 66.67) < 1.0

    def test_empty_data_returns_zero_metrics(self) -> None:
        metrics = calculate_metrics(
            pd.Series(dtype=float),
            pd.Series(dtype=float),
            [],
        )
        assert metrics["total_return_pct"] == 0.0
        assert metrics["sharpe_ratio"] == 0.0
        assert metrics["total_trades"] == 0

    def test_profit_factor(self) -> None:
        trades = [
            {"entry_price": 100, "exit_price": 120, "shares": 10,
             "pnl": 200, "entry_date": "2024-01-01", "exit_date": "2024-01-05",
             "symbol": "ITEM_A", "exit_reason": "take_profit"},
            {"entry_price": 100, "exit_price": 90, "shares": 10,
             "pnl": -100, "entry_date": "2024-01-06", "exit_date": "2024-01-10",
             "symbol": "ITEM_B", "exit_reason": "stop_loss"},
        ]
        equity_curve = pd.Series(
            [100000, 100200, 100100],
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )
        daily_returns = equity_curve.pct_change().dropna()
        metrics = calculate_metrics(equity_curve, daily_returns, trades)
        assert metrics["profit_factor"] == 2.0
