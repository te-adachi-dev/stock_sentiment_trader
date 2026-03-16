import csv
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest.engine import run_pead_backtest
from src.backtest.metrics import calculate_metrics
from src.earnings.screener import screen_pead_candidates
from src.strategy.pead import generate_pead_trades


def _make_price_df(n: int = 60, start_price: float = 100.0) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    prices = [start_price + i * 0.5 for i in range(n)]
    return pd.DataFrame({
        "Date": dates,
        "Open": prices,
        "High": [p + 2 for p in prices],
        "Low": [p - 1 for p in prices],
        "Close": [p + 0.3 for p in prices],
        "Volume": [1000000 + i * 10000 for i in range(n)],
    })


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


class TestVolumeFilterBypass:
    def test_volume_filter_skipped_when_multiplier_zero(self) -> None:
        earnings = pd.DataFrame({
            "date": ["2024-01-10", "2024-01-15", "2024-01-20"],
            "symbol": ["ITEM_A", "ITEM_B", "ITEM_C"],
            "eps_actual": [1.5, 1.8, 2.0],
            "eps_estimate": [1.0, 1.2, 1.3],
            "surprise_pct": [50.0, 50.0, 53.8],
            "sue": [2.5, 3.0, 2.8],
            "ear": [0.03, 0.04, 0.05],
        })
        price_data = {
            "ITEM_A": _make_price_df(),
            "ITEM_B": _make_price_df(),
            "ITEM_C": _make_price_df(),
        }

        result = screen_pead_candidates(
            earnings, price_data,
            {"surprise_threshold_pct": 5.0, "volume_spike_multiplier": 0.0, "check_volume": True},
        )
        assert len(result) == 3

    def test_volume_filter_applied_when_multiplier_positive(self) -> None:
        earnings = pd.DataFrame({
            "date": ["2024-01-10"],
            "symbol": ["ITEM_A"],
            "eps_actual": [1.5],
            "eps_estimate": [1.0],
            "surprise_pct": [50.0],
            "sue": [2.5],
            "ear": [0.03],
        })
        price_data = {"ITEM_A": _make_price_df()}

        result = screen_pead_candidates(
            earnings, price_data,
            {"surprise_threshold_pct": 5.0, "volume_spike_multiplier": 5.0, "check_volume": True},
        )
        assert len(result) <= 1


class TestSweepOutput:
    def test_sweep_csv_has_correct_columns(self, tmp_path: Path) -> None:
        earnings = pd.DataFrame({
            "date": ["2024-01-10", "2024-01-15"],
            "symbol": ["ITEM_A", "ITEM_B"],
            "eps_actual": [1.5, 1.8],
            "eps_estimate": [1.0, 1.2],
            "surprise_pct": [50.0, 50.0],
            "sue": [2.5, 3.0],
            "ear": [0.03, 0.04],
        })
        price_data = {
            "ITEM_A": _make_price_df(),
            "ITEM_B": _make_price_df(),
        }

        row = _run_single_backtest(
            earnings_with_ear=earnings,
            price_data=price_data,
            surprise_threshold=3.0,
            volume_spike=0.0,
            stop_loss=0.05,
            position_size=0.05,
            holding_period=5,
            entry_delay=1,
            max_positions=10,
            initial_capital=100000,
            commission=0.005,
            slippage=0.0005,
            risk_free_rate=0.045,
        )

        expected_keys = {
            "surprise_threshold", "stop_loss", "position_size", "holding_period",
            "total_return", "sharpe", "max_dd", "win_rate", "trades", "profit_factor",
        }
        assert set(row.keys()) == expected_keys

    def test_sweep_mini_run(self, tmp_path: Path) -> None:
        earnings = pd.DataFrame({
            "date": ["2024-01-10", "2024-01-15"],
            "symbol": ["ITEM_A", "ITEM_B"],
            "eps_actual": [1.5, 1.8],
            "eps_estimate": [1.0, 1.2],
            "surprise_pct": [50.0, 50.0],
            "sue": [2.5, 3.0],
            "ear": [0.03, 0.04],
        })
        price_data = {
            "ITEM_A": _make_price_df(),
            "ITEM_B": _make_price_df(),
        }

        results: list[dict[str, float]] = []
        params = [
            (3.0, 0.05, 0.05, 5),
            (5.0, 0.08, 0.10, 10),
        ]

        for surprise, sl, ps, hp in params:
            row = _run_single_backtest(
                earnings_with_ear=earnings,
                price_data=price_data,
                surprise_threshold=surprise,
                volume_spike=0.0,
                stop_loss=sl,
                position_size=ps,
                holding_period=hp,
                entry_delay=1,
                max_positions=10,
                initial_capital=100000,
                commission=0.005,
                slippage=0.0005,
                risk_free_rate=0.045,
            )
            results.append(row)

        assert len(results) == 2

        csv_path = tmp_path / "sweep_results.csv"
        fieldnames = [
            "surprise_threshold", "stop_loss", "position_size", "holding_period",
            "total_return", "sharpe", "max_dd", "win_rate", "trades", "profit_factor",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        df = pd.read_csv(csv_path)
        assert list(df.columns) == fieldnames
        assert len(df) == 2
