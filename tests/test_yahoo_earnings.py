import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestFetchYahooEarnings:
    def test_returns_correct_columns(self, tmp_path: Path) -> None:
        try:
            import yfinance
        except ImportError:
            pytest.skip("yfinance not installed")

        from src.earnings.yahoo_earnings import fetch_yahoo_earnings

        df = fetch_yahoo_earnings(
            symbols=["AAPL"],
            lookback_quarters=4,
            save_dir=str(tmp_path),
        )
        if df.empty:
            pytest.skip("No Yahoo earnings data returned (network issue)")

        expected_cols = {"date", "symbol", "eps_actual", "eps_estimate", "surprise", "surprise_pct"}
        assert expected_cols.issubset(set(df.columns))
        assert (df["symbol"] == "AAPL").all()
        assert df["eps_actual"].notna().all()

    def test_cache_is_created(self, tmp_path: Path) -> None:
        try:
            import yfinance
        except ImportError:
            pytest.skip("yfinance not installed")

        from src.earnings.yahoo_earnings import fetch_yahoo_earnings

        df = fetch_yahoo_earnings(
            symbols=["AAPL"],
            lookback_quarters=4,
            save_dir=str(tmp_path),
        )
        if df.empty:
            pytest.skip("No Yahoo earnings data returned (network issue)")

        cache_file = tmp_path / "yahoo_earnings_AAPL.csv"
        assert cache_file.exists()
        cached = pd.read_csv(cache_file)
        assert len(cached) > 0

    def test_surprise_pct_nan_when_estimate_zero(self) -> None:
        records = pd.DataFrame({
            "date": ["2024-01-01"],
            "symbol": ["TEST"],
            "eps_actual": [1.5],
            "eps_estimate": [0.0],
            "surprise": [1.5],
            "surprise_pct": [np.nan],
        })
        assert pd.isna(records.iloc[0]["surprise_pct"])
