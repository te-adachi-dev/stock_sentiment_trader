from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.strategy.position_manager import (
    calculate_position_size,
    calculate_stop_loss,
    calculate_take_profit,
)
from src.strategy.signals import generate_signals
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class BacktestResult:
    trades: list[dict]
    equity_curve: pd.Series
    daily_returns: pd.Series
    metrics: dict
    strategy_name: str
    params: dict


class BacktestEngine:
    def __init__(
        self,
        price_data: dict[str, pd.DataFrame],
        sentiment_data: dict[str, pd.DataFrame],
        technical_data: dict[str, pd.DataFrame],
        strategy: str,
        params: dict,
        initial_capital: float = 100_000.0,
        commission_per_share: float = 0.005,
        slippage_pct: float = 0.0005,
        risk_per_trade: float = 0.02,
        atr_stop_multiplier: float = 2.0,
        atr_profit_multiplier: float = 3.0,
        max_concurrent_positions: int = 3,
        daily_loss_limit: float = 0.05,
        warmup_days: int = 30,
    ) -> None:
        self.price_data = price_data
        self.sentiment_data = sentiment_data
        self.technical_data = technical_data
        self.strategy = strategy
        self.params = params
        self.initial_capital = initial_capital
        self.commission_per_share = commission_per_share
        self.slippage_pct = slippage_pct
        self.risk_per_trade = risk_per_trade
        self.atr_stop_multiplier = atr_stop_multiplier
        self.atr_profit_multiplier = atr_profit_multiplier
        self.max_concurrent_positions = max_concurrent_positions
        self.daily_loss_limit = daily_loss_limit
        self.warmup_days = warmup_days

    def run(self) -> BacktestResult:
        capital = self.initial_capital
        start_of_day_capital = capital
        trades: list[dict] = []
        positions: dict[str, dict] = {}
        equity_records: list[dict] = []

        all_signal_dfs: dict[str, pd.DataFrame] = {}
        for symbol in self.price_data:
            if symbol not in self.sentiment_data:
                continue
            signal_df = generate_signals(
                sentiment_df=self.sentiment_data[symbol],
                technical_df=self.technical_data[symbol],
                strategy=self.strategy,
                params=self.params,
            )
            all_signal_dfs[symbol] = signal_df

        if not all_signal_dfs:
            logger.warning("No signal data generated")
            empty_equity = pd.Series(dtype=float)
            empty_returns = pd.Series(dtype=float)
            return BacktestResult(
                trades=[], equity_curve=empty_equity,
                daily_returns=empty_returns, metrics={},
                strategy_name=self.strategy, params=self.params,
            )

        all_dates: set = set()
        for df in all_signal_dfs.values():
            all_dates.update(df["date"].tolist())
        sorted_dates = sorted(all_dates)

        if len(sorted_dates) <= self.warmup_days:
            logger.warning("Not enough data after warmup period")
            empty_equity = pd.Series(dtype=float)
            empty_returns = pd.Series(dtype=float)
            return BacktestResult(
                trades=[], equity_curve=empty_equity,
                daily_returns=empty_returns, metrics={},
                strategy_name=self.strategy, params=self.params,
            )

        trading_dates = sorted_dates[self.warmup_days:]

        for i, current_date in enumerate(trading_dates):
            start_of_day_capital = capital + self._unrealized_pnl(positions, all_signal_dfs, current_date)
            daily_loss_threshold = self.initial_capital * self.daily_loss_limit
            daily_pnl = start_of_day_capital - (
                capital + self._unrealized_pnl(positions, all_signal_dfs, current_date)
            )

            symbols_to_close: list[str] = []
            for sym, pos in positions.items():
                current_price = self._get_price(all_signal_dfs, sym, current_date, "Open")
                if current_price is None:
                    continue
                if current_price <= pos["stop_loss"]:
                    pnl = self._close_position(pos, current_price)
                    capital += pnl
                    trades.append({
                        "symbol": sym, "entry_date": pos["entry_date"],
                        "exit_date": current_date, "entry_price": pos["entry_price"],
                        "exit_price": current_price, "shares": pos["shares"],
                        "pnl": pnl, "exit_reason": "stop_loss",
                    })
                    symbols_to_close.append(sym)
                elif current_price >= pos["take_profit"]:
                    pnl = self._close_position(pos, current_price)
                    capital += pnl
                    trades.append({
                        "symbol": sym, "entry_date": pos["entry_date"],
                        "exit_date": current_date, "entry_price": pos["entry_price"],
                        "exit_price": current_price, "shares": pos["shares"],
                        "pnl": pnl, "exit_reason": "take_profit",
                    })
                    symbols_to_close.append(sym)
            for sym in symbols_to_close:
                del positions[sym]

            if i > 0:
                prev_date = trading_dates[i - 1]
                realized_today = sum(
                    t["pnl"] for t in trades if t["exit_date"] == current_date
                )
                if realized_today < -daily_loss_threshold:
                    equity_val = capital + self._unrealized_pnl(positions, all_signal_dfs, current_date)
                    equity_records.append({"date": current_date, "equity": equity_val})
                    continue

            for symbol, signal_df in all_signal_dfs.items():
                if symbol in positions:
                    continue
                if len(positions) >= self.max_concurrent_positions:
                    break

                prev_date_idx = None
                date_list = signal_df["date"].tolist()
                for idx, d in enumerate(date_list):
                    if d == current_date and idx > 0:
                        prev_date_idx = idx - 1
                        break

                if prev_date_idx is None:
                    continue

                prev_row = signal_df.iloc[prev_date_idx]
                signal = prev_row.get("signal", 0)
                if signal != 1:
                    continue

                current_row_matches = signal_df[signal_df["date"] == current_date]
                if current_row_matches.empty:
                    continue
                current_row = current_row_matches.iloc[0]

                entry_price = current_row.get("Open", None)
                if entry_price is None or pd.isna(entry_price):
                    continue

                entry_price = entry_price * (1 + self.slippage_pct)

                atr = current_row.get("ATR_14", None)
                if atr is None or pd.isna(atr) or atr <= 0:
                    continue

                shares = calculate_position_size(
                    capital=capital, atr=atr,
                    risk_per_trade=self.risk_per_trade,
                    atr_multiplier=self.atr_stop_multiplier,
                )
                shares = int(shares)
                if shares <= 0:
                    continue

                cost = shares * entry_price + shares * self.commission_per_share
                if cost > capital:
                    shares = int((capital * 0.95) / (entry_price + self.commission_per_share))
                    if shares <= 0:
                        continue
                    cost = shares * entry_price + shares * self.commission_per_share

                capital -= cost
                stop_loss = calculate_stop_loss(entry_price, atr, self.atr_stop_multiplier)
                take_profit = calculate_take_profit(entry_price, atr, self.atr_profit_multiplier)

                positions[symbol] = {
                    "entry_date": current_date,
                    "entry_price": entry_price,
                    "shares": shares,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "cost_basis": cost,
                }

            sell_signals: list[str] = []
            for symbol in list(positions.keys()):
                signal_df = all_signal_dfs.get(symbol)
                if signal_df is None:
                    continue

                prev_date_idx = None
                date_list = signal_df["date"].tolist()
                for idx, d in enumerate(date_list):
                    if d == current_date and idx > 0:
                        prev_date_idx = idx - 1
                        break
                if prev_date_idx is None:
                    continue

                prev_row = signal_df.iloc[prev_date_idx]
                if prev_row.get("signal", 0) == -1:
                    sell_signals.append(symbol)

            for sym in sell_signals:
                if sym not in positions:
                    continue
                pos = positions[sym]
                exit_price = self._get_price(all_signal_dfs, sym, current_date, "Open")
                if exit_price is None:
                    continue
                exit_price = exit_price * (1 - self.slippage_pct)
                pnl = self._close_position(pos, exit_price)
                capital += pnl
                trades.append({
                    "symbol": sym, "entry_date": pos["entry_date"],
                    "exit_date": current_date, "entry_price": pos["entry_price"],
                    "exit_price": exit_price, "shares": pos["shares"],
                    "pnl": pnl, "exit_reason": "signal_sell",
                })
                del positions[sym]

            equity_val = capital + self._unrealized_pnl(positions, all_signal_dfs, current_date)
            equity_records.append({"date": current_date, "equity": equity_val})

        for sym in list(positions.keys()):
            pos = positions[sym]
            last_date = trading_dates[-1]
            exit_price = self._get_price(all_signal_dfs, sym, last_date, "Close")
            if exit_price is None:
                exit_price = pos["entry_price"]
            pnl = self._close_position(pos, exit_price)
            capital += pnl
            trades.append({
                "symbol": sym, "entry_date": pos["entry_date"],
                "exit_date": last_date, "entry_price": pos["entry_price"],
                "exit_price": exit_price, "shares": pos["shares"],
                "pnl": pnl, "exit_reason": "end_of_backtest",
            })

        equity_df = pd.DataFrame(equity_records)
        if equity_df.empty:
            equity_curve = pd.Series(dtype=float)
            daily_returns = pd.Series(dtype=float)
        else:
            equity_curve = equity_df.set_index("date")["equity"]
            daily_returns = equity_curve.pct_change().dropna()

        logger.info(
            f"Backtest complete for {self.strategy}: "
            f"{len(trades)} trades, final equity {equity_curve.iloc[-1] if not equity_curve.empty else 0:.2f}"
        )

        return BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            daily_returns=daily_returns,
            metrics={},
            strategy_name=self.strategy,
            params=self.params,
        )

    def _close_position(self, pos: dict, exit_price: float) -> float:
        gross = pos["shares"] * exit_price
        commission = pos["shares"] * self.commission_per_share
        return gross - commission

    def _unrealized_pnl(
        self,
        positions: dict[str, dict],
        signal_dfs: dict[str, pd.DataFrame],
        current_date: object,
    ) -> float:
        total = 0.0
        for sym, pos in positions.items():
            current_price = self._get_price(signal_dfs, sym, current_date, "Close")
            if current_price is None:
                continue
            total += pos["shares"] * current_price - pos["cost_basis"]
        return total

    def _get_price(
        self,
        signal_dfs: dict[str, pd.DataFrame],
        symbol: str,
        target_date: object,
        price_col: str,
    ) -> float | None:
        df = signal_dfs.get(symbol)
        if df is None:
            return None
        matches = df[df["date"] == target_date]
        if matches.empty:
            return None
        val = matches.iloc[0].get(price_col)
        if val is None or pd.isna(val):
            return None
        return float(val)


def run_pead_backtest(
    trades: list[dict],
    initial_capital: float,
    position_size_pct: float,
    commission_per_share: float,
    slippage_pct: float,
) -> BacktestResult:
    if not trades:
        return BacktestResult(
            trades=[], equity_curve=pd.Series(dtype=float),
            daily_returns=pd.Series(dtype=float), metrics={},
            strategy_name="pead", params={},
        )

    sorted_trades = sorted(trades, key=lambda t: t["entry_date"])

    capital = initial_capital
    completed_trades: list[dict] = []

    all_dates: set = set()
    for t in sorted_trades:
        entry = pd.Timestamp(t["entry_date"])
        exit_d = pd.Timestamp(t["exit_date"])
        current = entry
        while current <= exit_d:
            if current.weekday() < 5:
                all_dates.add(current.date())
            current += pd.Timedelta(days=1)

    if not all_dates:
        return BacktestResult(
            trades=[], equity_curve=pd.Series(dtype=float),
            daily_returns=pd.Series(dtype=float), metrics={},
            strategy_name="pead", params={},
        )

    date_list = sorted(all_dates)
    active_positions: list[dict] = []
    equity_records: list[dict] = []

    trade_queue = list(sorted_trades)

    for current_date in date_list:
        closed_indices: list[int] = []
        for i, pos in enumerate(active_positions):
            exit_dt = pd.Timestamp(pos["exit_date"]).date()
            if current_date >= exit_dt:
                exit_price = pos["exit_price"]
                exit_price_adj = exit_price * (1 - slippage_pct)
                proceeds = pos["shares"] * exit_price_adj
                commission = pos["shares"] * commission_per_share
                capital += proceeds - commission

                pnl = (
                    pos["shares"] * exit_price_adj
                    - pos["shares"] * commission_per_share
                    - pos["cost_basis"]
                )

                completed_trades.append({
                    **pos["trade_info"],
                    "pnl": pnl,
                    "shares": pos["shares"],
                })
                closed_indices.append(i)

        for i in sorted(closed_indices, reverse=True):
            active_positions.pop(i)

        new_entries: list[dict] = []
        for t in trade_queue:
            entry_dt = pd.Timestamp(t["entry_date"]).date()
            if entry_dt == current_date:
                new_entries.append(t)

        for t in new_entries:
            trade_queue.remove(t)
            if capital <= 0:
                continue
            alloc = min(capital, initial_capital) * position_size_pct
            entry_price = t["entry_price"] * (1 + slippage_pct)
            shares = int(alloc / entry_price)
            if shares <= 0:
                continue
            cost = shares * entry_price + shares * commission_per_share
            if cost > capital:
                shares = int((capital * 0.95) / (entry_price + commission_per_share))
                if shares <= 0:
                    continue
                cost = shares * entry_price + shares * commission_per_share

            capital -= cost
            active_positions.append({
                "shares": shares,
                "entry_price": entry_price,
                "exit_price": t["exit_price"],
                "exit_date": t["exit_date"],
                "cost_basis": cost,
                "trade_info": t,
            })

        unrealized = sum(
            pos["shares"] * pos["exit_price"] - pos["cost_basis"]
            for pos in active_positions
        )
        equity_records.append({"date": current_date, "equity": capital + unrealized})

    equity_df = pd.DataFrame(equity_records)
    if equity_df.empty:
        equity_curve = pd.Series(dtype=float)
        daily_returns = pd.Series(dtype=float)
    else:
        equity_curve = equity_df.set_index("date")["equity"]
        daily_returns = equity_curve.pct_change().dropna()

    logger.info(
        f"PEAD backtest complete: {len(completed_trades)} trades, "
        f"final equity {equity_curve.iloc[-1] if not equity_curve.empty else 0:.2f}"
    )

    return BacktestResult(
        trades=completed_trades,
        equity_curve=equity_curve,
        daily_returns=daily_returns,
        metrics={},
        strategy_name="pead",
        params={"position_size_pct": position_size_pct},
    )
