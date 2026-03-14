import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def generate_ensemble_signal(
    sentiment_signal: pd.Series,
    technical_signal: pd.Series,
    sentiment_weight: float = 0.6,
    technical_weight: float = 0.4,
    threshold: float = 0.3,
) -> pd.Series:
    combined_score = (
        sentiment_signal * sentiment_weight + technical_signal * technical_weight
    )
    signals = pd.Series(0, index=combined_score.index)
    signals[combined_score > threshold] = 1
    signals[combined_score < -threshold] = -1

    buy_count = int((signals == 1).sum())
    sell_count = int((signals == -1).sum())
    hold_count = int((signals == 0).sum())
    logger.info(
        f"Ensemble signal: {buy_count} BUY, {sell_count} SELL, {hold_count} HOLD"
    )
    return signals
