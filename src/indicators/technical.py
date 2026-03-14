import pandas as pd
import ta

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    if "Close" not in result.columns:
        logger.error("DataFrame must contain 'Close' column")
        return result

    # RSI (14)
    result["RSI_14"] = ta.momentum.RSIIndicator(
        close=result["Close"], window=14
    ).rsi()

    # MACD (12, 26, 9)
    macd = ta.trend.MACD(
        close=result["Close"], window_slow=26, window_fast=12, window_sign=9
    )
    result["MACD"] = macd.macd()
    result["MACD_Signal"] = macd.macd_signal()
    result["MACD_Hist"] = macd.macd_diff()

    # Bollinger Bands (20, 2sigma)
    bb = ta.volatility.BollingerBands(
        close=result["Close"], window=20, window_dev=2
    )
    result["BB_Upper"] = bb.bollinger_hband()
    result["BB_Middle"] = bb.bollinger_mavg()
    result["BB_Lower"] = bb.bollinger_lband()

    # ATR (14)
    if "High" in result.columns and "Low" in result.columns:
        result["ATR_14"] = ta.volatility.AverageTrueRange(
            high=result["High"],
            low=result["Low"],
            close=result["Close"],
            window=14,
        ).average_true_range()

    # Volume Moving Average Ratio (20)
    if "Volume" in result.columns:
        vol_ma = result["Volume"].rolling(window=20).mean()
        result["Volume_MA_Ratio"] = result["Volume"] / vol_ma

    logger.info(f"Technical indicators calculated: {len(result)} rows")
    return result
