import pandas as pd
import pytest

from src.strategy.signals import generate_signals


def _make_sentiment_df(scores: list[tuple[str, float, float, float]]) -> pd.DataFrame:
    rows = []
    for i, (dt, pos, neg, neu) in enumerate(scores):
        rows.append({
            "datetime": dt,
            "symbol": "TEST",
            "headline": f"Item {i}",
            "sentiment_label": "positive" if pos > neg else "negative",
            "sentiment_score": max(pos, neg, neu),
            "positive": pos,
            "negative": neg,
            "neutral": neu,
        })
    return pd.DataFrame(rows)


def _make_technical_df(n: int = 5, rsi_values: list[float] | None = None) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "Date": dates,
        "Open": [100.0 + i for i in range(n)],
        "High": [105.0 + i for i in range(n)],
        "Low": [95.0 + i for i in range(n)],
        "Close": [102.0 + i for i in range(n)],
        "Volume": [1000000] * n,
        "Returns": [0.01] * n,
    })
    if rsi_values is not None:
        df["RSI_14"] = rsi_values
    else:
        df["RSI_14"] = [50.0] * n
    df["ATR_14"] = [2.0] * n
    return df


class TestSignals:
    def test_threshold_buy(self) -> None:
        sentiment_df = _make_sentiment_df([
            ("2024-01-01T10:00:00Z", 0.8, 0.1, 0.1),
        ])
        technical_df = _make_technical_df(n=1)
        result = generate_signals(
            sentiment_df, technical_df, "threshold",
            {"buy_threshold": 0.3, "sell_threshold": -0.3},
        )
        assert result["signal"].iloc[0] == 1

    def test_threshold_sell(self) -> None:
        sentiment_df = _make_sentiment_df([
            ("2024-01-01T10:00:00Z", 0.1, 0.8, 0.1),
        ])
        technical_df = _make_technical_df(n=1)
        result = generate_signals(
            sentiment_df, technical_df, "threshold",
            {"buy_threshold": 0.3, "sell_threshold": -0.3},
        )
        assert result["signal"].iloc[0] == -1

    def test_threshold_hold(self) -> None:
        sentiment_df = _make_sentiment_df([
            ("2024-01-01T10:00:00Z", 0.4, 0.3, 0.3),
        ])
        technical_df = _make_technical_df(n=1)
        result = generate_signals(
            sentiment_df, technical_df, "threshold",
            {"buy_threshold": 0.3, "sell_threshold": -0.3},
        )
        assert result["signal"].iloc[0] == 0

    def test_combo_requires_both_conditions(self) -> None:
        sentiment_df = _make_sentiment_df([
            ("2024-01-01T10:00:00Z", 0.7, 0.1, 0.2),
        ])
        technical_df = _make_technical_df(n=1, rsi_values=[55.0])
        result = generate_signals(
            sentiment_df, technical_df, "combo",
            {"sentiment_buy": 0.2, "sentiment_sell": -0.2,
             "rsi_buy": 40, "rsi_sell": 60},
        )
        assert result["signal"].iloc[0] == 0

    def test_combo_buy_when_both_met(self) -> None:
        sentiment_df = _make_sentiment_df([
            ("2024-01-01T10:00:00Z", 0.7, 0.1, 0.2),
        ])
        technical_df = _make_technical_df(n=1, rsi_values=[35.0])
        result = generate_signals(
            sentiment_df, technical_df, "combo",
            {"sentiment_buy": 0.2, "sentiment_sell": -0.2,
             "rsi_buy": 40, "rsi_sell": 60},
        )
        assert result["signal"].iloc[0] == 1

    def test_mean_reversion_extreme_positive_sells(self) -> None:
        sentiment_df = _make_sentiment_df([
            ("2024-01-01T10:00:00Z", 0.9, 0.05, 0.05),
        ])
        technical_df = _make_technical_df(n=1)
        result = generate_signals(
            sentiment_df, technical_df, "mean_reversion",
            {"extreme_positive": 0.5, "extreme_negative": -0.5},
        )
        assert result["signal"].iloc[0] == -1

    def test_mean_reversion_extreme_negative_buys(self) -> None:
        sentiment_df = _make_sentiment_df([
            ("2024-01-01T10:00:00Z", 0.05, 0.9, 0.05),
        ])
        technical_df = _make_technical_df(n=1)
        result = generate_signals(
            sentiment_df, technical_df, "mean_reversion",
            {"extreme_positive": 0.5, "extreme_negative": -0.5},
        )
        assert result["signal"].iloc[0] == 1

    def test_unknown_strategy_raises(self) -> None:
        sentiment_df = _make_sentiment_df([
            ("2024-01-01T10:00:00Z", 0.5, 0.3, 0.2),
        ])
        technical_df = _make_technical_df(n=1)
        with pytest.raises(ValueError, match="Unknown strategy"):
            generate_signals(sentiment_df, technical_df, "invalid", {})
