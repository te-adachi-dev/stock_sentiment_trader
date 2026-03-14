import numpy as np
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def calculate_sue(earnings_df: pd.DataFrame) -> pd.DataFrame:
    result = earnings_df.copy()
    result["sue"] = np.nan

    if "eps_actual" not in result.columns or "eps_estimate" not in result.columns:
        logger.warning("Missing eps_actual or eps_estimate columns")
        return result

    for symbol in result["symbol"].unique():
        mask = result["symbol"] == symbol
        sym_data = result.loc[mask].sort_values("date").copy()

        surprises = sym_data["eps_actual"] - sym_data["eps_estimate"]
        for i in range(len(sym_data)):
            actual = sym_data.iloc[i]["eps_actual"]
            estimate = sym_data.iloc[i]["eps_estimate"]

            if pd.isna(actual) or pd.isna(estimate) or estimate == 0:
                continue

            surprise = actual - estimate

            if i >= 2:
                past_surprises = surprises.iloc[:i]
                std = past_surprises.std()
                if std > 0:
                    sue_val = surprise / std
                else:
                    sue_val = surprise / abs(estimate)
            else:
                sue_val = surprise / abs(estimate)

            idx = sym_data.index[i]
            result.loc[idx, "sue"] = sue_val

    valid_count = result["sue"].notna().sum()
    logger.info(f"SUE calculated for {valid_count}/{len(result)} records")
    return result


def calculate_ear(
    earnings_df: pd.DataFrame,
    price_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    result = earnings_df.copy()
    result["ear"] = np.nan

    for idx, row in result.iterrows():
        symbol = row["symbol"]
        earn_date = row["date"]

        if symbol not in price_data:
            continue

        pdf = price_data[symbol].copy()
        if "Date" in pdf.columns:
            pdf["_date"] = pd.to_datetime(pdf["Date"], utc=True).dt.tz_convert(None).dt.date
        else:
            continue

        earn_dt = pd.Timestamp(earn_date).date()

        dates_sorted = sorted(pdf["_date"].unique())
        date_positions = {d: i for i, d in enumerate(dates_sorted)}

        if earn_dt not in date_positions:
            closest = min(dates_sorted, key=lambda d: abs((d - earn_dt).days), default=None)
            if closest is None or abs((closest - earn_dt).days) > 5:
                continue
            earn_dt = closest

        pos = date_positions[earn_dt]
        t_minus1 = pos - 1 if pos >= 1 else pos
        t_plus1 = pos + 1 if pos < len(dates_sorted) - 1 else pos

        window_dates = dates_sorted[t_minus1 : t_plus1 + 1]
        window_data = pdf[pdf["_date"].isin(window_dates)].sort_values("_date")

        if len(window_data) < 2 or "Close" not in window_data.columns:
            continue

        start_price = window_data.iloc[0]["Close"]
        end_price = window_data.iloc[-1]["Close"]

        if start_price > 0:
            ear_val = (end_price - start_price) / start_price
            result.loc[idx, "ear"] = ear_val

    valid_count = result["ear"].notna().sum()
    logger.info(f"EAR calculated for {valid_count}/{len(result)} records")
    return result
