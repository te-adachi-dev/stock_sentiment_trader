import os

import pandas as pd
import pytest

from src.earnings.surprise import calculate_sue, calculate_ear


class TestEarnings:
    def test_finnhub_earnings_fetch(self) -> None:
        api_key = os.getenv("FINNHUB_API_KEY")
        if not api_key:
            pytest.skip("FINNHUB_API_KEY not set")

        from src.earnings.calendar import fetch_company_earnings

        df = fetch_company_earnings("AAPL", api_key=api_key, limit=4)
        assert not df.empty
        assert "actual" in df.columns
        assert "estimate" in df.columns

    def test_sue_calculation(self) -> None:
        earnings = pd.DataFrame({
            "date": [
                "2023-01-01", "2023-04-01", "2023-07-01",
                "2023-10-01", "2024-01-01",
            ],
            "symbol": ["ITEM"] * 5,
            "eps_actual": [1.0, 1.1, 1.2, 1.3, 1.5],
            "eps_estimate": [0.9, 1.0, 1.1, 1.2, 1.2],
        })

        result = calculate_sue(earnings)
        assert "sue" in result.columns
        assert result["sue"].notna().sum() > 0
        assert result.iloc[-1]["sue"] > 0

    def test_sue_skips_zero_estimate(self) -> None:
        earnings = pd.DataFrame({
            "date": ["2023-01-01", "2023-04-01"],
            "symbol": ["ITEM", "ITEM"],
            "eps_actual": [1.0, 1.5],
            "eps_estimate": [0.0, 1.0],
        })
        result = calculate_sue(earnings)
        assert pd.isna(result.iloc[0]["sue"])

    def test_ear_calculation(self) -> None:
        earnings = pd.DataFrame({
            "date": ["2024-01-03"],
            "symbol": ["ITEM"],
            "eps_actual": [2.0],
            "eps_estimate": [1.5],
        })

        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        prices = [100, 101, 102, 105, 108, 107, 109, 110, 111, 112]
        price_data = {
            "ITEM": pd.DataFrame({
                "Date": dates,
                "Open": prices,
                "High": [p + 2 for p in prices],
                "Low": [p - 2 for p in prices],
                "Close": prices,
                "Volume": [1000000] * 10,
            })
        }

        result = calculate_ear(earnings, price_data)
        assert "ear" in result.columns
        assert result.iloc[0]["ear"] > 0
