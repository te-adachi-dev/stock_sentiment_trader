import datetime

import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def generate_pead_trades(
    candidates: pd.DataFrame,
    price_data: dict[str, pd.DataFrame],
    holding_period: int,
    params: dict,
) -> list[dict]:
    entry_delay = params.get("entry_delay_days", 1)
    stop_loss_pct = params.get("stop_loss_pct", 0.08)
    max_positions = params.get("max_positions", 10)

    trades: list[dict] = []
    candidates_sorted = candidates.sort_values("date").reset_index(drop=True)

    active_positions: list[dict] = []

    for _, row in candidates_sorted.iterrows():
        symbol = row["symbol"]
        earn_date = row["date"]

        if symbol not in price_data:
            continue

        pdf = price_data[symbol].copy()
        if "Date" not in pdf.columns:
            continue

        pdf["_date"] = pd.to_datetime(pdf["Date"], utc=True).dt.tz_convert(None).dt.date
        dates_sorted = sorted(pdf["_date"].unique())
        date_positions = {d: i for i, d in enumerate(dates_sorted)}

        earn_dt = pd.Timestamp(earn_date).date()
        if earn_dt not in date_positions:
            closest = [d for d in dates_sorted if d >= earn_dt]
            if not closest:
                continue
            earn_dt = closest[0]

        earn_pos = date_positions[earn_dt]
        entry_pos = earn_pos + entry_delay
        if entry_pos >= len(dates_sorted):
            continue

        entry_date = dates_sorted[entry_pos]

        active_positions = [
            p for p in active_positions
            if p["exit_date"] > entry_date
        ]
        if len(active_positions) >= max_positions:
            continue

        entry_row = pdf[pdf["_date"] == entry_date]
        if entry_row.empty:
            continue

        entry_price = float(entry_row.iloc[0]["Open"])
        if entry_price <= 0:
            continue

        stop_price = entry_price * (1 - stop_loss_pct)

        exit_pos = min(entry_pos + holding_period, len(dates_sorted) - 1)
        exit_date_target = dates_sorted[exit_pos]

        exit_price = entry_price
        exit_reason = "holding_period"
        actual_exit_date = exit_date_target

        for day_offset in range(1, exit_pos - entry_pos + 1):
            check_pos = entry_pos + day_offset
            if check_pos >= len(dates_sorted):
                break
            check_date = dates_sorted[check_pos]
            check_row = pdf[pdf["_date"] == check_date]
            if check_row.empty:
                continue

            low = float(check_row.iloc[0]["Low"])
            if low <= stop_price:
                exit_price = stop_price
                exit_reason = "stop_loss"
                actual_exit_date = check_date
                break

        if exit_reason == "holding_period":
            exit_row = pdf[pdf["_date"] == exit_date_target]
            if not exit_row.empty:
                exit_price = float(exit_row.iloc[0]["Close"])
            else:
                last_available = pdf[pdf["_date"] <= exit_date_target].tail(1)
                if not last_available.empty:
                    exit_price = float(last_available.iloc[0]["Close"])
                    actual_exit_date = last_available.iloc[0]["_date"]

        return_pct = (exit_price - entry_price) / entry_price * 100
        holding_days = (
            pd.Timestamp(actual_exit_date) - pd.Timestamp(entry_date)
        ).days

        trade = {
            "symbol": symbol,
            "entry_date": entry_date,
            "entry_price": entry_price,
            "exit_date": actual_exit_date,
            "exit_price": exit_price,
            "holding_days": max(holding_days, 1),
            "return_pct": return_pct,
            "exit_reason": exit_reason,
            "surprise_pct": row.get("surprise_pct", 0.0),
            "sue": row.get("sue", 0.0),
            "ear": row.get("ear", 0.0),
        }
        trades.append(trade)

        active_positions.append({
            "symbol": symbol,
            "entry_date": entry_date,
            "exit_date": actual_exit_date,
        })

    logger.info(
        f"PEAD trades generated: {len(trades)} trades, "
        f"holding_period={holding_period}d"
    )
    return trades
