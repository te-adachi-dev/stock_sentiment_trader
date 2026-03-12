import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def generate_signals(
    sentiment_df: pd.DataFrame,
    technical_df: pd.DataFrame,
    strategy: str,
    params: dict,
) -> pd.DataFrame:
    if strategy == "threshold":
        return _strategy_threshold(sentiment_df, technical_df, params)
    elif strategy == "combo":
        return _strategy_combo(sentiment_df, technical_df, params)
    elif strategy == "mean_reversion":
        return _strategy_mean_reversion(sentiment_df, technical_df, params)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _strategy_threshold(
    sentiment_df: pd.DataFrame,
    technical_df: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    buy_threshold = params.get("buy_threshold", 0.3)
    sell_threshold = params.get("sell_threshold", -0.3)

    merged = _merge_data(sentiment_df, technical_df)
    signals: list[int] = []
    for score in merged["sentiment_score"]:
        if score > buy_threshold:
            signals.append(1)
        elif score < sell_threshold:
            signals.append(-1)
        else:
            signals.append(0)
    merged["signal"] = signals
    logger.info(
        f"Threshold strategy: {sum(s == 1 for s in signals)} BUY, "
        f"{sum(s == -1 for s in signals)} SELL, "
        f"{sum(s == 0 for s in signals)} HOLD"
    )
    return merged


def _strategy_combo(
    sentiment_df: pd.DataFrame,
    technical_df: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    sentiment_buy = params.get("sentiment_buy", 0.2)
    sentiment_sell = params.get("sentiment_sell", -0.2)
    rsi_buy = params.get("rsi_buy", 40)
    rsi_sell = params.get("rsi_sell", 60)

    merged = _merge_data(sentiment_df, technical_df)
    signals: list[int] = []
    for _, row in merged.iterrows():
        score = row["sentiment_score"]
        rsi = row.get("RSI_14", 50)
        if pd.isna(rsi):
            rsi = 50
        if score > sentiment_buy and rsi < rsi_buy:
            signals.append(1)
        elif score < sentiment_sell and rsi > rsi_sell:
            signals.append(-1)
        else:
            signals.append(0)
    merged["signal"] = signals
    logger.info(
        f"Combo strategy: {sum(s == 1 for s in signals)} BUY, "
        f"{sum(s == -1 for s in signals)} SELL, "
        f"{sum(s == 0 for s in signals)} HOLD"
    )
    return merged


def _strategy_mean_reversion(
    sentiment_df: pd.DataFrame,
    technical_df: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    extreme_positive = params.get("extreme_positive", 0.5)
    extreme_negative = params.get("extreme_negative", -0.5)

    merged = _merge_data(sentiment_df, technical_df)
    signals: list[int] = []
    for score in merged["sentiment_score"]:
        if score > extreme_positive:
            signals.append(-1)
        elif score < extreme_negative:
            signals.append(1)
        else:
            signals.append(0)
    merged["signal"] = signals
    logger.info(
        f"Mean reversion strategy: {sum(s == 1 for s in signals)} BUY, "
        f"{sum(s == -1 for s in signals)} SELL, "
        f"{sum(s == 0 for s in signals)} HOLD"
    )
    return merged


def _merge_data(
    sentiment_df: pd.DataFrame,
    technical_df: pd.DataFrame,
) -> pd.DataFrame:
    sent = sentiment_df.copy()
    sent["date"] = pd.to_datetime(sent["datetime"]).dt.date
    daily_sentiment = sent.groupby("date").agg(
        sentiment_score_pos=("positive", "mean"),
        sentiment_score_neg=("negative", "mean"),
    ).reset_index()
    daily_sentiment["sentiment_score"] = (
        daily_sentiment["sentiment_score_pos"] - daily_sentiment["sentiment_score_neg"]
    )

    tech = technical_df.copy()
    if "Date" in tech.columns:
        tech["date"] = pd.to_datetime(tech["Date"], utc=True).dt.tz_convert(None).dt.date
    else:
        tech["date"] = pd.to_datetime(tech.index).date

    merged = pd.merge(tech, daily_sentiment[["date", "sentiment_score"]], on="date", how="left")
    merged["sentiment_score"] = merged["sentiment_score"].fillna(0.0)
    merged = merged.sort_values("date").reset_index(drop=True)
    return merged
