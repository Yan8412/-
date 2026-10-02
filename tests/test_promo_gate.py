"""Paper-only 1进2 low-tercile gate. No network."""

from __future__ import annotations

import json
from datetime import date

import pytest

from ashare.broker.paper import PaperBroker
from ashare.config import Settings, load_settings
from ashare.pipeline import publish_daily
from ashare.sentiment import MarketDay, MarketPanel
from ashare.terciles import evaluate_promo_gate
from tests.helpers import loose_settings, trading_days


def _day(**overrides) -> MarketDay:
    payload = dict(
        breadth=0.5,
        breadth_count=300,
        limit_up_count=10,
        broken_count=2,
        broken_rate=0.2,
        max_height=2,
        prev_limit_return=0.01,
        index_level=100.0,
        index_vs_ma=0.01,
        risk_on=True,
        promo_1_2=0.3,
    )
    payload.update(overrides)
    return MarketDay(**payload)


def _panel(promos: list[float]) -> MarketPanel:
    days = trading_days(len(promos), date(2024, 1, 2))
    records = {day: _day(promo_1_2=promo) for day, promo in zip(days, promos, strict=True)}
    return MarketPanel(days=records, calendar=days, limit_up={}, height={})


def _settings(**overrides) -> Settings:
    settings = loose_settings()
    settings.strategies = ["ma_pullback"]
    settings.use_ranker = False
    settings.promo_gate_train_days = 10
    settings.promo_gate_test_days = 5
    for key, value in overrides.items():
        setattr(settings, key, value)
    return settings


def _promos(last: float, count: int = 15) -> list[float]:
    train = [0.20 + 0.03 * index for index in range(10)]
    test = [0.40] * (count - 11)
    return train + test + [last]


def _row(day: date) -> list[dict]:
    return [
        {
            "code": "600000",
            "name": "测试股份",
            "strategy": "均线回踩",
            "strategy_id": "ma_pullback",
            "reason": "测试",
            "score": 1.0,
            "close": 10.0,
            "entry_low": 9.5,
            "entry_high": 10.5,
            "take_profit": 11.0,
            "stop_loss": 9.0,
            "take_profit_pct": 0.08,
            "stop_loss_pct": 0.05,
            "max_hold_days": 5,
            "shares": 100,
            "budget": 1000.0,
            "signal_date": day.isoformat(),
        }
    ]


def _publish(tmp_path, panel: MarketPanel, settings: Settings, monkeypatch):
    day = panel.calendar[-1]
    monkeypatch.setattr("ashare.pipeline.build_recommendations", lambda *args, **kwargs: _row(day))
    monkeypatch.setattr("ashare.pipeline.load_ranker", lambda: (_ for _ in ()).throw(AssertionError("ranker")))
    path = publish_daily([], panel, settings, day, tmp_path, tmp_path / "ledger.json", [])
    broker = PaperBroker(tmp_path / "ledger.json", settings)
    snapshot = json.loads((tmp_path / "market_latest.json").read_text(encoding="utf-8"))
    return path.read_text(encoding="utf-8"), broker.snapshot().pending, snapshot


def test_gate_off_still_writes_paper_entries(tmp_path, monkeypatch):
    panel = _panel(_promos(0.05))
    text, pending, snapshot = _publish(tmp_path, panel, _settings(promo_gate_enabled=False), monkeypatch)
    assert len(pending) == 1
    assert pending[0]["code"] == "600000"
    assert snapshot["promo_gate"] == "off"
    assert snapshot["promo_gate_enabled"] is False
    assert "闸门未开启" in text
    assert snapshot["entry_note"] == "允许开新仓"


def test_low_promo_blocks_new_entries_only(tmp_path, monkeypatch):
    panel = _panel(_promos(0.05))
    text, pending, snapshot = _publish(tmp_path, panel, _settings(promo_gate_enabled=True), monkeypatch)
    assert pending == []
    assert snapshot["promo_gate"] == "closed"
    assert snapshot["promo_gate_tercile"] == 0
    assert snapshot["promo_1_2"] == pytest.approx(0.05)
    assert snapshot["promo_gate_cut"] == pytest.approx(0.26)
    assert "闸门关闭" in text
    assert "已有持仓仍按原规则卖出" in text
    assert "不写新委托" in snapshot["entry_note"]


def test_normal_promo_allows_entries(tmp_path, monkeypatch):
    panel = _panel(_promos(0.50))
    text, pending, snapshot = _publish(tmp_path, panel, _settings(promo_gate_enabled=True), monkeypatch)
    assert len(pending) == 1
    assert snapshot["promo_gate"] == "open"
    assert snapshot["promo_gate_tercile"] == 2
    assert snapshot["promo_gate_cut"] == pytest.approx(0.26)
    assert "闸门开启" in text


def test_gate_uses_only_data_up_to_the_signal_day():
    base = _promos(0.05)
    panel = _panel(base)
    day = panel.calendar[10]
    # The first test session is low. Later sessions must not move its cut.
    panel.days[day] = _day(promo_1_2=0.05)
    first = evaluate_promo_gate(panel, day, train_days=10, test_days=5)
    assert first.blocked
    assert first.train_end < day
    assert first.cut == pytest.approx(0.26)
    extended = _panel(base + [0.0] * 10)
    extended.days[day] = _day(promo_1_2=0.05)
    extended.days[extended.calendar[14]] = _day(promo_1_2=0.0)
    again = evaluate_promo_gate(extended, day, train_days=10, test_days=5)
    assert again.cut == first.cut
    assert again.tercile == first.tercile
    assert again.blocked
    assert again.train_start == first.train_start
    assert again.train_end == first.train_end


def test_missing_history_fails_open(tmp_path, monkeypatch, caplog):
    panel = _panel([0.05, 0.05, 0.05])
    text, pending, snapshot = _publish(tmp_path, panel, _settings(promo_gate_enabled=True), monkeypatch)
    assert len(pending) == 1
    assert snapshot["promo_gate"] == "unavailable"
    assert snapshot["promo_gate_warning"]
    assert "不拦截" in text
    assert any("不拦截" in record.message for record in caplog.records)


def test_promo_gate_config_defaults_off(tmp_path):
    missing = load_settings(tmp_path / "missing.json")
    assert missing.promo_gate_enabled is False
    assert missing.promo_gate_train_days == 140
    assert missing.promo_gate_test_days == 70
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "promo_gate_enabled": True,
                "promo_gate_train_days": 140,
                "promo_gate_test_days": 70,
            }
        ),
        encoding="utf-8",
    )
    loaded = load_settings(path)
    assert loaded.promo_gate_enabled is True
    assert loaded.promo_gate_train_days == 140
    path.write_text(json.dumps({"promo_gate_enabled": "true"}), encoding="utf-8")
    try:
        load_settings(path)
    except ValueError as exc:
        assert "promo_gate_enabled" in str(exc)
    else:
        raise AssertionError("non-bool gate flag was accepted")
