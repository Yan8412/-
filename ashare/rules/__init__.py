"""A-share trading constraints shared by the backtest and the paper broker."""

from ashare.rules.adjust import apply_forward_adjust, ex_rights_reference
from ashare.rules.costs import buy_cash_out, sell_cash_in, stamp_rate
from ashare.rules.execution import evaluate_buy, evaluate_sell
from ashare.rules.limits import board_limit_ratio, classify_board, limit_prices
from ashare.rules.lots import board_lot, suggest_shares

__all__ = [
    "apply_forward_adjust",
    "board_limit_ratio",
    "board_lot",
    "buy_cash_out",
    "classify_board",
    "evaluate_buy",
    "evaluate_sell",
    "ex_rights_reference",
    "limit_prices",
    "sell_cash_in",
    "stamp_rate",
    "suggest_shares",
]
