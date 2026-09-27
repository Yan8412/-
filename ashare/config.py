"""Runtime settings for a 20,000 CNY A-share account."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from datetime import date
from pathlib import Path

from ashare.paths import default_config_path


@dataclass
class Settings:
    """Fee rates follow the retail schedule used in mainland China.

    Commission is charged on both sides with a 5 CNY minimum.
    Stamp duty is sell-side only: 0.1% before 2023-08-28 and 0.05% after.
    Transfer fee (过户费) is 0.001% of notional on both sides.
    Slippage is adverse to the account (buys higher, sells lower).
    """

    initial_capital: float = 20_000.0
    max_positions: int = 4
    max_position_pct: float = 0.35
    commission_rate: float = 0.00025
    min_commission: float = 5.0
    stamp_duty_rate: float = 0.0005
    stamp_duty_rate_legacy: float = 0.001
    stamp_duty_change: date = date(2023, 8, 28)
    transfer_fee_rate: float = 0.00001
    slippage_rate: float = 0.001
    lot_size: int = 100
    star_lot_size: int = 200
    price_min: float = 3.0
    price_max: float = 50.0
    min_amount: float = 30_000_000.0
    min_history_bars: int = 60
    min_listed_bars: int = 120
    # 20-session high-low range / close. Short-term targets are several percent;
    # names that only drift a few percent (工商银行 in the previous sample) never
    # pay for the 5 CNY commission floor. 0 disables the check.
    min_swing_20d: float = 0.10
    top_n: int = 10
    universe_size: int = 50
    full_market: bool = True
    history_bars: int = 700
    train_days: int = 140
    test_days: int = 70
    min_trades_for_selection: int = 5
    # Frozen before the 2025-04-25 holdout was read. New entries require both
    # conditions, and at least regime_min_names names in the breadth count.
    regime_breadth_min: float = 0.40
    regime_ma_window: int = 20
    regime_min_names: int = 200
    ml_pool: int = 80

    def slot_budget(self, equity: float, cash: float) -> float:
        """Cash allocated to one new position, capped by the account rules."""
        if equity <= 0 or cash <= 0:
            return 0.0
        return min(cash, equity * self.max_position_pct, equity / self.max_positions)


def load_settings(path: Path | None = None) -> Settings:
    """Load optional JSON overrides. Unknown keys are ignored."""
    settings = Settings()
    candidate = path if path is not None else default_config_path()
    if not candidate.exists():
        return settings
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("config.json 必须是对象")
    known = {item.name for item in fields(settings)}
    data = asdict(settings)
    for key, value in payload.items():
        if key not in known or key == "stamp_duty_change":
            continue
        data[key] = value
    if "stamp_duty_change" in payload:
        data["stamp_duty_change"] = date.fromisoformat(str(payload["stamp_duty_change"]))
    return Settings(**data)
