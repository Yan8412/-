"""Corporate-action reference prices and a local forward-adjusted series.

Tencent raw bars quote the exchange print. Indicators that compare prices
across a dividend day use the adjusted series; fills and limit prices use
the raw print and the ex-rights reference price.
"""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal

from ashare.rules.money import CENT, D

_CASH = re.compile(r"派([0-9]+(?:\.[0-9]+)?)")
_BONUS = re.compile(r"送([0-9]+(?:\.[0-9]+)?)")
_TRANSFER = re.compile(r"转(?:增)?([0-9]+(?:\.[0-9]+)?)")


def parse_corporate_action(payload: dict | None) -> tuple[float, float]:
    """Return (cash per share, extra-share ratio) for one ex-rights day.

    ``10派2.49元`` → cash 0.249. ``10送5转5`` → ratio 1.0.
    An empty payload means the day is not an ex-rights day.
    """
    if not payload:
        return 0.0, 0.0
    text = str(payload.get("FHcontent") or "")
    cash = 0.0
    ratio = 0.0
    if text:
        matched = _CASH.search(text)
        if matched:
            cash = _per_ten(matched.group(1))
        matched = _BONUS.search(text)
        if matched:
            ratio += _per_ten(matched.group(1))
        matched = _TRANSFER.search(text)
        if matched:
            ratio += _per_ten(matched.group(1))
    elif payload.get("fh_sh") not in (None, ""):
        cash = _per_ten(str(payload["fh_sh"]))
    return cash, ratio


def _per_ten(value: str) -> float:
    """``10派2.49`` means 0.249 per share, without binary division noise."""
    return float(Decimal(value) / Decimal("10"))


def ex_rights_reference(prev_close: object, cash_per_share: object, bonus_ratio: object) -> Decimal:
    """Official-style next-day reference price, half-up to 0.01.

    reference = (previous close - cash dividend) / (1 + bonus + transfer)
    """
    numerator = D(prev_close) - D(cash_per_share)
    denominator = Decimal("1") + D(bonus_ratio)
    if denominator <= 0:
        raise ValueError("送转比例导致除权分母无效")
    return (numerator / denominator).quantize(CENT, rounding=ROUND_HALF_UP)


def apply_causal_adjust(
    open_: list[float],
    high: list[float],
    low: list[float],
    close: list[float],
    preclose: list[float],
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Total-return prices that only depend on bars up to each date.

    Broker-style 前复权 restates old prices when a later dividend is known.
    Indicators here use this causal scale instead, so a walk-forward signal
    cannot see a corporate action that has not happened yet.
    """
    n = len(close)
    if not (len(open_) == len(high) == len(low) == len(preclose) == n):
        raise ValueError("复权输入长度不一致")
    adj_open = [0.0] * n
    adj_high = [0.0] * n
    adj_low = [0.0] * n
    adj_close = [0.0] * n
    if n == 0:
        return adj_open, adj_high, adj_low, adj_close
    adj_open[0] = open_[0]
    adj_high[0] = high[0]
    adj_low[0] = low[0]
    adj_close[0] = close[0]
    for index in range(1, n):
        base = preclose[index]
        scale = (adj_close[index - 1] / base) if base else 1.0
        adj_open[index] = open_[index] * scale
        adj_high[index] = high[index] * scale
        adj_low[index] = low[index] * scale
        adj_close[index] = close[index] * scale
    return adj_open, adj_high, adj_low, adj_close


def apply_forward_adjust(
    open_: list[float],
    high: list[float],
    low: list[float],
    close: list[float],
    cash: list[float],
    bonus: list[float],
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Express every bar in the latest raw price scale (前复权).

    ``cash[i]`` / ``bonus[i]`` describe the corporate action that takes effect
    on bar ``i`` (the gap between close[i-1] and open[i]).
    """
    n = len(close)
    if not (len(open_) == len(high) == len(low) == len(cash) == len(bonus) == n):
        raise ValueError("复权输入长度不一致")
    q_open = [0.0] * n
    q_high = [0.0] * n
    q_low = [0.0] * n
    q_close = [0.0] * n
    if n == 0:
        return q_open, q_high, q_low, q_close
    q_open[-1] = open_[-1]
    q_high[-1] = high[-1]
    q_low[-1] = low[-1]
    q_close[-1] = close[-1]
    for i in range(n - 2, -1, -1):
        later = close[i + 1]
        scale = (q_close[i + 1] / later) if later else 1.0
        ratio = bonus[i + 1]
        dividend = cash[i + 1]
        factor = scale / (1.0 + ratio) if (1.0 + ratio) else scale

        def convert(price: float) -> float:
            return (price - dividend) * factor

        q_open[i] = convert(open_[i])
        q_high[i] = convert(high[i])
        q_low[i] = convert(low[i])
        q_close[i] = convert(close[i])
    return q_open, q_high, q_low, q_close
