import pandas as pd
import pytest

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


class TestScreener:
    def test_filters_negative_sue(self) -> None:
        earnings = pd.DataFrame({
            "date": ["2024-01-10", "2024-01-15"],
            "symbol": ["ITEM_A", "ITEM_B"],
            "eps_actual": [1.5, 0.8],
            "eps_estimate": [1.0, 1.0],
            "surprise_pct": [50.0, -20.0],
            "sue": [2.5, -1.5],
            "ear": [0.03, -0.02],
        })
        price_data = {
            "ITEM_A": _make_price_df(),
            "ITEM_B": _make_price_df(),
        }
        result = screen_pead_candidates(
            earnings, price_data,
            {"surprise_threshold_pct": 5.0, "volume_spike_multiplier": 0.0, "check_volume": False},
        )
        assert len(result) == 1
        assert result.iloc[0]["symbol"] == "ITEM_A"

    def test_filters_below_surprise_threshold(self) -> None:
        earnings = pd.DataFrame({
            "date": ["2024-01-10", "2024-01-15"],
            "symbol": ["ITEM_A", "ITEM_B"],
            "eps_actual": [1.06, 1.5],
            "eps_estimate": [1.05, 1.0],
            "surprise_pct": [1.0, 50.0],
            "sue": [0.5, 3.0],
            "ear": [0.01, 0.05],
        })
        result = screen_pead_candidates(
            earnings, {},
            {"surprise_threshold_pct": 5.0, "check_volume": False},
        )
        assert len(result) == 1
        assert result.iloc[0]["symbol"] == "ITEM_B"


class TestPeadStrategy:
    def test_entry_after_exit(self) -> None:
        candidates = pd.DataFrame({
            "date": ["2024-01-10"],
            "symbol": ["ITEM"],
            "eps_actual": [1.5],
            "eps_estimate": [1.0],
            "surprise_pct": [50.0],
            "sue": [3.0],
            "ear": [0.03],
            "volume_ratio": [2.0],
        })
        price_data = {"ITEM": _make_price_df()}

        trades = generate_pead_trades(
            candidates, price_data, holding_period=5,
            params={"entry_delay_days": 1, "stop_loss_pct": 0.08, "max_positions": 10},
        )
        assert len(trades) > 0
        for t in trades:
            assert t["entry_date"] < t["exit_date"]

    def test_stop_loss_triggers(self) -> None:
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        prices = [100.0] * 12 + [100.0 - i * 3.0 for i in range(18)]
        price_data = {
            "ITEM": pd.DataFrame({
                "Date": dates,
                "Open": prices,
                "High": [p + 1 for p in prices],
                "Low": [p - 1 for p in prices],
                "Close": prices,
                "Volume": [1000000] * 30,
            })
        }

        candidates = pd.DataFrame({
            "date": ["2024-01-10"],
            "symbol": ["ITEM"],
            "eps_actual": [1.5],
            "eps_estimate": [1.0],
            "surprise_pct": [50.0],
            "sue": [3.0],
            "ear": [0.03],
            "volume_ratio": [2.0],
        })

        trades = generate_pead_trades(
            candidates, price_data, holding_period=20,
            params={"entry_delay_days": 1, "stop_loss_pct": 0.08, "max_positions": 10},
        )
        assert len(trades) == 1
        assert trades[0]["exit_reason"] == "stop_loss"

    def test_holding_period_respected(self) -> None:
        candidates = pd.DataFrame({
            "date": ["2024-01-05"],
            "symbol": ["ITEM"],
            "eps_actual": [1.5],
            "eps_estimate": [1.0],
            "surprise_pct": [50.0],
            "sue": [3.0],
            "ear": [0.03],
            "volume_ratio": [2.0],
        })
        price_data = {"ITEM": _make_price_df(n=60)}

        for hp in [5, 10, 20]:
            trades = generate_pead_trades(
                candidates, price_data, holding_period=hp,
                params={"entry_delay_days": 1, "stop_loss_pct": 0.50, "max_positions": 10},
            )
            if trades and trades[0]["exit_reason"] == "holding_period":
                assert trades[0]["holding_days"] <= hp + 5

    def test_max_positions_respected(self) -> None:
        candidates = pd.DataFrame({
            "date": ["2024-01-05"] * 5,
            "symbol": [f"ITEM_{i}" for i in range(5)],
            "eps_actual": [1.5] * 5,
            "eps_estimate": [1.0] * 5,
            "surprise_pct": [50.0] * 5,
            "sue": [3.0] * 5,
            "ear": [0.03] * 5,
            "volume_ratio": [2.0] * 5,
        })
        price_data = {f"ITEM_{i}": _make_price_df(n=60) for i in range(5)}

        trades = generate_pead_trades(
            candidates, price_data, holding_period=20,
            params={"entry_delay_days": 1, "stop_loss_pct": 0.50, "max_positions": 3},
        )
        same_day_entries = [t for t in trades if str(t["entry_date"]) == "2024-01-06"]
        assert len(same_day_entries) <= 3
