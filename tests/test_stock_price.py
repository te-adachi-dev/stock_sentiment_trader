import pandas as pd
import pytest

from src.data_collection.stock_price import fetch_stock_data


class TestStockPrice:
    def test_fetch_single_symbol(self, tmp_path: str) -> None:
        result = fetch_stock_data(
            symbols=["AAPL"],
            period="1mo",
            interval="1d",
            save_dir=str(tmp_path),
        )
        assert "AAPL" in result
        df = result["AAPL"]
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_required_columns(self, tmp_path: str) -> None:
        result = fetch_stock_data(
            symbols=["AAPL"],
            period="1mo",
            interval="1d",
            save_dir=str(tmp_path),
        )
        df = result["AAPL"]
        required_columns = ["Date", "Open", "High", "Low", "Close", "Volume", "Returns"]
        for col in required_columns:
            assert col in df.columns, f"Missing column: {col}"

    def test_returns_calculation(self, tmp_path: str) -> None:
        result = fetch_stock_data(
            symbols=["AAPL"],
            period="1mo",
            interval="1d",
            save_dir=str(tmp_path),
        )
        df = result["AAPL"]
        assert df["Returns"].iloc[0] != df["Returns"].iloc[0] or pd.isna(df["Returns"].iloc[0])
        valid_returns = df["Returns"].dropna()
        assert len(valid_returns) > 0
        expected = df["Close"].pct_change().dropna()
        pd.testing.assert_series_equal(
            valid_returns.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
            atol=1e-10,
        )

    def test_invalid_symbol_skipped(self, tmp_path: str) -> None:
        result = fetch_stock_data(
            symbols=["INVALID_SYMBOL_XYZ123"],
            period="1mo",
            interval="1d",
            save_dir=str(tmp_path),
        )
        assert "INVALID_SYMBOL_XYZ123" not in result or result["INVALID_SYMBOL_XYZ123"].empty
