from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def calculate_position_size(
    capital: float,
    atr: float,
    risk_per_trade: float,
    atr_multiplier: float,
) -> float:
    if atr <= 0 or capital <= 0:
        return 0.0
    risk_amount = capital * risk_per_trade
    position_size = risk_amount / (atr * atr_multiplier)
    return max(0.0, position_size)


def calculate_stop_loss(
    entry_price: float,
    atr: float,
    multiplier: float,
) -> float:
    return entry_price - (atr * multiplier)


def calculate_take_profit(
    entry_price: float,
    atr: float,
    multiplier: float,
) -> float:
    return entry_price + (atr * multiplier)
