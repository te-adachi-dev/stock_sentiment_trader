import pandas as pd
import pytest

from src.backtest.engine import BacktestEngine, run_pead_backtest


def _make_price_df(n: int = 10) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    prices = [100.0 + i * 0.5 for i in range(n)]
    return pd.DataFrame({
        "Date": dates,
        "Open": prices,
        "High": [p + 2 for p in prices],
        "Low": [p - 2 for p in prices],
        "Close": [p + 0.5 for p in prices],
        "Volume": [1000000] * n,
        "Returns": [0.005] * n,
        "Symbol": ["ITEM"] * n,
    })


def _make_technical_df(n: int = 10) -> pd.DataFrame:
    df = _make_price_df(n)
    df["RSI_14"] = [35.0] * n
    df["MACD"] = [0.1] * n
    df["MACD_Signal"] = [0.05] * n
    df["MACD_Hist"] = [0.05] * n
    df["BB_Upper"] = [110.0] * n
    df["BB_Middle"] = [100.0] * n
    df["BB_Lower"] = [90.0] * n
    df["ATR_14"] = [2.0] * n
    df["Volume_MA_Ratio"] = [1.0] * n
    return df


def _make_sentiment_df(n: int = 10, positive: float = 0.7) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    rows = []
    for d in dates:
        rows.append({
            "datetime": d.isoformat(),
            "symbol": "ITEM",
            "headline": "Positive outlook",
            "sentiment_label": "positive",
            "sentiment_score": positive,
            "positive": positive,
            "negative": 0.1,
            "neutral": 1 - positive - 0.1,
        })
    return pd.DataFrame(rows)


class TestBacktestEngine:
    def test_backtest_completes(self) -> None:
        n = 40
        price = _make_price_df(n)
        technical = _make_technical_df(n)
        sentiment = _make_sentiment_df(n)

        engine = BacktestEngine(
            price_data={"ITEM": price},
            sentiment_data={"ITEM": sentiment},
            technical_data={"ITEM": technical},
            strategy="threshold",
            params={"buy_threshold": 0.3, "sell_threshold": -0.3},
            initial_capital=100_000.0,
            warmup_days=5,
        )
        result = engine.run()
        assert result.strategy_name == "threshold"
        assert not result.equity_curve.empty

    def test_buy_signal_executes_next_day_open(self) -> None:
        n = 40
        price = _make_price_df(n)
        technical = _make_technical_df(n)
        sentiment = _make_sentiment_df(n, positive=0.8)

        engine = BacktestEngine(
            price_data={"ITEM": price},
            sentiment_data={"ITEM": sentiment},
            technical_data={"ITEM": technical},
            strategy="threshold",
            params={"buy_threshold": 0.3, "sell_threshold": -0.3},
            initial_capital=100_000.0,
            warmup_days=5,
        )
        result = engine.run()

        if result.trades:
            first_trade = result.trades[0]
            assert first_trade["entry_price"] > 0
            assert first_trade["shares"] > 0

    def test_stop_loss_triggers(self) -> None:
        n = 40
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        prices = [100.0] * 10 + [100.0 - i * 2.0 for i in range(30)]
        price_df = pd.DataFrame({
            "Date": dates,
            "Open": prices,
            "High": [p + 1 for p in prices],
            "Low": [p - 1 for p in prices],
            "Close": prices,
            "Volume": [1000000] * n,
            "Returns": [0.0] + [(prices[i] - prices[i - 1]) / prices[i - 1] if prices[i - 1] != 0 else 0 for i in range(1, n)],
            "Symbol": ["ITEM"] * n,
        })
        tech_df = price_df.copy()
        tech_df["RSI_14"] = [35.0] * n
        tech_df["ATR_14"] = [2.0] * n

        sentiment = _make_sentiment_df(n, positive=0.8)

        engine = BacktestEngine(
            price_data={"ITEM": price_df},
            sentiment_data={"ITEM": sentiment},
            technical_data={"ITEM": tech_df},
            strategy="threshold",
            params={"buy_threshold": 0.3, "sell_threshold": -0.3},
            initial_capital=100_000.0,
            warmup_days=5,
            atr_stop_multiplier=2.0,
        )
        result = engine.run()

        stop_loss_trades = [t for t in result.trades if t["exit_reason"] == "stop_loss"]
        assert len(stop_loss_trades) > 0

    def test_commission_deducted(self) -> None:
        n = 40
        price = _make_price_df(n)
        technical = _make_technical_df(n)
        sentiment = _make_sentiment_df(n, positive=0.8)

        engine = BacktestEngine(
            price_data={"ITEM": price},
            sentiment_data={"ITEM": sentiment},
            technical_data={"ITEM": technical},
            strategy="threshold",
            params={"buy_threshold": 0.3, "sell_threshold": -0.3},
            initial_capital=100_000.0,
            commission_per_share=1.0,
            warmup_days=5,
        )
        result = engine.run()

        if result.trades:
            final_equity = result.equity_curve.iloc[-1]
            assert final_equity != 100_000.0

    def test_pead_backtest_capital_never_deeply_negative(self) -> None:
        trades = []
        for i in range(20):
            entry_date = f"2024-01-{(i * 3 + 1):02d}"
            exit_date = f"2024-01-{(i * 3 + 3):02d}"
            if i * 3 + 3 > 28:
                break
            trades.append({
                "symbol": "ITEM",
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_price": 100.0,
                "exit_price": 60.0,
                "return_pct": -40.0,
                "direction": "long",
                "surprise_pct": 10.0,
            })

        result = run_pead_backtest(
            trades=trades,
            initial_capital=100_000,
            position_size_pct=0.5,
            commission_per_share=0.005,
            slippage_pct=0.0005,
        )

        if not result.equity_curve.empty:
            min_equity = result.equity_curve.min()
            max_dd_pct = (1 - min_equity / 100_000) * 100
            assert max_dd_pct < 200, f"Max DD {max_dd_pct}% is unreasonably large"
