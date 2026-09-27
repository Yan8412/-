"""Cent-accurate money helpers. Python's round() uses banker's rounding."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

CENT = Decimal("0.01")


def D(value: object) -> Decimal:
    """Convert a price-like value to Decimal without binary float surprises."""
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def money(value: object) -> Decimal:
    """Round half up to 0.01 CNY."""
    return D(value).quantize(CENT, rounding=ROUND_HALF_UP)


def to_cents(value: object) -> int:
    """Integer cents after half-up rounding."""
    return int((money(value) * 100).to_integral_value(rounding=ROUND_HALF_UP))
