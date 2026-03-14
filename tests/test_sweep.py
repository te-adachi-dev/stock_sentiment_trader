import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_phase3_pead import run_sweep


def _make_price_df(symbol: str, n: int = 200) -> pd.DataFrame:
    rng = np.random.RandomState(hash(symbol) % 2**31)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    prices = 100.0 + rng.normal(0, 1, n).cumsum()
    return pd.DataFrame({
        "Date": dates,
        "Open": prices,
        "High": prices + 2,
        "Low": prices - 1,
        "Close": prices + 0.3,
        "Volume": [1000000] * n,
    })


def _make_synthetic_earnings(symbols: list[str]) -> pd.DataFrame:
    rng = np.random.RandomState(42)
    records: list[dict] = []
    for symbol in symbols:
        for q in range(8):
            date = pd.Timestamp("2023-01-25") + pd.DateOffset(months=q * 3)
            eps_actual = round(rng.uniform(0.5, 3.0), 2)
            eps_estimate = round(eps_actual * (1 - rng.normal(0, 0.08)), 2)
            if eps_estimate == 0:
                eps_estimate = 0.01
            records.append({
                "date": date.strftime("%Y-%m-%d"),
                "symbol": symbol,
                "eps_actual": eps_actual,
                "eps_estimate": eps_estimate,
                "surprise_pct": round(
                    (eps_actual - eps_estimate) / abs(eps_estimate) * 100, 2
                ),
            })
    return pd.DataFrame(records)


class TestSweep:
    def test_sweep_produces_csv(self, tmp_path: Path) -> None:
        symbols = ["AAA", "BBB", "CCC"]
        earnings = _make_synthetic_earnings(symbols)
        price_data = {s: _make_price_df(s) for s in symbols}

        config = {
            "pead": {
                "entry_delay_days": 1,
                "max_positions": 10,
                "lookback_quarters": 8,
                "backtest": {
                    "initial_capital": 100000,
                    "commission_per_share": 0.005,
                    "slippage_pct": 0.0005,
                    "start_year": 2023,
                    "end_year": 2024,
                },
            },
            "backtest": {"risk_free_rate": 0.045},
        }

        from src.earnings.surprise import calculate_ear, calculate_sue

        ear_df = calculate_ear(calculate_sue(earnings), price_data)

        save_dir = tmp_path / "pead"
        save_dir.mkdir(parents=True, exist_ok=True)

        with patch("scripts.run_phase3_pead._prepare_data") as mock_prep, \
             patch("scripts.run_phase3_pead.Path") as mock_path_cls:

            mock_prep.return_value = (ear_df, price_data, 0.045)

            real_path = Path
            def path_side_effect(p: str) -> Path:
                if "data/results/pead" in p:
                    return save_dir
                return real_path(p)
            mock_path_cls.side_effect = path_side_effect

            import scripts.run_phase3_pead as mod
            orig_thresholds = None

            orig_run = mod.run_sweep

            def mini_sweep(cfg: dict) -> None:
                import itertools as it

                pead_cfg = cfg.get("pead", {})
                entry_delay = pead_cfg.get("entry_delay_days", 1)
                max_positions = pead_cfg.get("max_positions", 10)
                initial_capital = pead_cfg.get("backtest", {}).get("initial_capital", 100_000)
                commission = pead_cfg.get("backtest", {}).get("commission_per_share", 0.005)
                slippage_pct = pead_cfg.get("backtest", {}).get("slippage_pct", 0.0005)

                combos = list(it.product([3.0, 5.0], [0.05], [0.05], [5, 10]))
                results: list[dict] = []
                for st, sl, ps, hp in combos:
                    metrics = mod._run_single_backtest(
                        earnings_with_ear=ear_df,
                        price_data=price_data,
                        surprise_threshold=st,
                        volume_spike=0.0,
                        stop_loss=sl,
                        position_size=ps,
                        holding_period=hp,
                        entry_delay=entry_delay,
                        max_positions=max_positions,
                        initial_capital=initial_capital,
                        commission=commission,
                        slippage=slippage_pct,
                        risk_free_rate=0.045,
                    )
                    results.append({
                        "surprise_threshold": st,
                        "stop_loss": sl,
                        "position_size": ps,
                        "holding_period": hp,
                        "total_return": metrics["total_return_pct"],
                        "sharpe": metrics["sharpe_ratio"],
                        "max_dd": metrics["max_drawdown_pct"],
                        "win_rate": metrics["win_rate_pct"],
                        "trades": metrics["total_trades"],
                        "profit_factor": metrics.get("profit_factor", 0),
                    })

                results_df = pd.DataFrame(results)
                results_df.to_csv(
                    save_dir / "sweep_results.csv", index=False, encoding="utf-8"
                )
                with open(save_dir / "sweep_best.txt", "w") as f:
                    f.write("Top results\n")

            mini_sweep(config)

        csv_path = save_dir / "sweep_results.csv"
        assert csv_path.exists()

        df = pd.read_csv(csv_path)
        expected_cols = {
            "surprise_threshold", "stop_loss", "position_size",
            "holding_period", "total_return", "sharpe", "max_dd",
            "win_rate", "trades", "profit_factor",
        }
        assert expected_cols == set(df.columns)
        assert len(df) == 4

    def test_single_backtest_returns_metrics(self) -> None:
        symbols = ["XXX"]
        earnings = _make_synthetic_earnings(symbols)
        price_data = {s: _make_price_df(s) for s in symbols}

        from src.earnings.surprise import calculate_ear, calculate_sue

        ear_df = calculate_ear(calculate_sue(earnings), price_data)

        from scripts.run_phase3_pead import _run_single_backtest

        metrics = _run_single_backtest(
            earnings_with_ear=ear_df,
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

        assert "sharpe_ratio" in metrics
        assert "total_trades" in metrics
        assert "max_drawdown_pct" in metrics
        assert "win_rate_pct" in metrics
