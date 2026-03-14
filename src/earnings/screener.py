import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def screen_pead_candidates(
    earnings_df: pd.DataFrame,
    price_data: dict[str, pd.DataFrame],
    params: dict,
) -> pd.DataFrame:
    surprise_threshold = params.get("surprise_threshold_pct", 5.0)
    volume_spike_mult = params.get("volume_spike_multiplier", 1.5)
    check_volume = params.get("check_volume", True)

    df = earnings_df.copy()

    if "sue" not in df.columns or "ear" not in df.columns:
        logger.warning("Missing sue or ear columns; run calculate_sue and calculate_ear first")
        return pd.DataFrame()

    initial_count = len(df)

    df = df[df["sue"].notna() & (df["sue"] > 0)]
    after_sue = len(df)

    if "surprise_pct" in df.columns:
        df = df[df["surprise_pct"].notna() & (df["surprise_pct"] > surprise_threshold)]
    elif "eps_actual" in df.columns and "eps_estimate" in df.columns:
        df = df[df["eps_estimate"] != 0]
        df = df.copy()
        df["surprise_pct"] = (
            (df["eps_actual"] - df["eps_estimate"]) / df["eps_estimate"].abs() * 100
        )
        df = df[df["surprise_pct"] > surprise_threshold]
    after_surprise = len(df)

    df = df[df["ear"].notna() & (df["ear"] > 0)]
    after_ear = len(df)

    if check_volume and price_data:
        volume_ratios: list[float] = []
        for _, row in df.iterrows():
            ratio = _get_volume_ratio(row["symbol"], row["date"], price_data)
            volume_ratios.append(ratio)
        df = df.copy()
        df["volume_ratio"] = volume_ratios
        df = df[df["volume_ratio"] >= volume_spike_mult]
    else:
        df = df.copy()
        df["volume_ratio"] = 0.0

    after_volume = len(df)

    logger.info(
        f"Screening: {initial_count} -> SUE>0: {after_sue} -> "
        f"surprise>{surprise_threshold}%: {after_surprise} -> "
        f"EAR>0: {after_ear} -> volume: {after_volume}"
    )

    return df.reset_index(drop=True)


def _get_volume_ratio(
    symbol: str,
    earn_date: str,
    price_data: dict[str, pd.DataFrame],
) -> float:
    if symbol not in price_data:
        return 0.0

    pdf = price_data[symbol].copy()
    if "Date" not in pdf.columns or "Volume" not in pdf.columns:
        return 0.0

    pdf["_date"] = pd.to_datetime(pdf["Date"], utc=True).dt.tz_convert(None).dt.date
    earn_dt = pd.Timestamp(earn_date).date()

    date_mask = pdf["_date"] == earn_dt
    if not date_mask.any():
        closest = pdf.loc[
            (pdf["_date"] >= earn_dt - pd.Timedelta(days=3))
            & (pdf["_date"] <= earn_dt + pd.Timedelta(days=3))
        ]
        if closest.empty:
            return 0.0
        earn_dt = closest.iloc[0]["_date"]
        date_mask = pdf["_date"] == earn_dt

    earn_volume = pdf.loc[date_mask, "Volume"].iloc[0]

    prior = pdf[pdf["_date"] < earn_dt].tail(30)
    if prior.empty:
        return 0.0

    avg_volume = prior["Volume"].mean()
    if avg_volume <= 0:
        return 0.0

    return earn_volume / avg_volume
