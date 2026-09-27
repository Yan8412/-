"""Price-limit bands by board. ST names are excluded upstream, not re-priced here."""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

from ashare.rules.money import CENT, D, money, to_cents

MAIN_LIMIT = Decimal("0.10")
GROWTH_LIMIT = Decimal("0.20")
BEIJING_LIMIT = Decimal("0.30")


def normalize_code(code: str) -> str:
    """Return the 6-digit code, stripping an sh/sz/bj prefix or exchange suffix."""
    text = code.strip().lower()
    for prefix in ("sh", "sz", "bj"):
        if text.startswith(prefix) and len(text) >= 8:
            text = text[2:]
            break
    if "." in text:
        text = text.split(".", 1)[0]
    if not text.isdigit() or len(text) != 6:
        raise ValueError(f"无法识别的证券代码: {code}")
    return text


def classify_board(code: str) -> str:
    """Board id used for lot size and the daily limit ratio."""
    symbol = normalize_code(code)
    if symbol.startswith(("200", "900")):
        return "b_share"
    if symbol.startswith(("688", "689")):
        return "star"
    if symbol.startswith(("300", "301")):
        return "chinext"
    if symbol.startswith(("43", "82", "83", "87", "88", "92")):
        return "bj"
    if symbol.startswith(("000", "001", "002", "003", "600", "601", "603", "605")):
        return "main"
    return "other"


def board_limit_ratio(code: str) -> Decimal:
    """Daily up/down limit as a fraction of the previous reference price."""
    board = classify_board(code)
    if board in {"star", "chinext"}:
        return GROWTH_LIMIT
    if board == "bj":
        return BEIJING_LIMIT
    return MAIN_LIMIT


def limit_prices(preclose: object, code: str) -> tuple[Decimal, Decimal]:
    """Exchange limit-up and limit-down prices, rounded half-up to 0.01."""
    ratio = board_limit_ratio(code)
    base = D(preclose)
    up = (base * (Decimal("1") + ratio)).quantize(CENT, rounding=ROUND_HALF_UP)
    down = (base * (Decimal("1") - ratio)).quantize(CENT, rounding=ROUND_HALF_UP)
    return up, down


def is_limit_up(price: object, limit_up: object) -> bool:
    return to_cents(price) >= to_cents(limit_up)


def is_limit_down(price: object, limit_down: object) -> bool:
    return to_cents(price) <= to_cents(limit_down)


def is_risk_name(name: str) -> bool:
    """ST / *ST / delisting-risk names are out of the v1 universe."""
    text = (name or "").strip()
    upper = text.upper().replace(" ", "")
    if "ST" in upper:
        return True
    if text.startswith(("N", "C", "S")):
        return True
    if "退" in text:
        return True
    return False


def round_price(value: object) -> Decimal:
    return money(value)
