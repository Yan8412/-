"""Daily book selection and cache freshness. No network."""

import json
from datetime import date, datetime, timedelta

import pytest

from ashare.cli import apply_daily_overrides, build_parser
from ashare.config import load_settings
from ashare.data.cache import cache_is_fresh, load_bars, save_bars
from ashare.data.sources import TAIL_BARS, MarketData
from ashare.market import RawBar
from ashare.market_calendar import SHANGHAI, latest_completed_session
from ashare.pipeline import publish_daily, run_daily
from ashare.sentiment import MarketDay, MarketPanel
from tests.helpers import loose_settings


def _bar(day: date) -> RawBar:
    return RawBar(day, 10, 10.2, 9.8, 10, 1000, 10000)


def _bars_ending(last: date, count: int) -> list[RawBar]:
    bars: list[RawBar] = []
    day = last
    while len(bars) < count:
        if day.weekday() < 5:
            bars.append(_bar(day))
        day -= timedelta(days=1)
    bars.reverse()
    return bars


def _clock(moment: str) -> datetime:
    return datetime.fromisoformat(moment).replace(tzinfo=SHANGHAI)


def test_latest_completed_session_respects_the_close_and_holidays():
    assert latest_completed_session(_clock("2026-09-27T12:00:00")) == date(2026, 9, 24)
    assert latest_completed_session(_clock("2026-09-28T15:29:00")) == date(2026, 9, 24)
    assert latest_completed_session(_clock("2026-09-28T15:30:00")) == date(2026, 9, 28)
    assert latest_completed_session(_clock("2026-09-28T16:00:00")) == date(2026, 9, 28)
    assert latest_completed_session(_clock("2026-10-07T12:00:00")) == date(2026, 9, 30)
    assert latest_completed_session(_clock("2026-10-08T15:29:00")) == date(2026, 9, 30)
    assert latest_completed_session(_clock("2026-10-08T15:30:00")) == date(2026, 10, 8)


def test_cache_follows_the_completed_session_not_a_six_day_window():
    bars = _bars_ending(date(2026, 9, 24), 80)
    fetched = "2026-09-24T08:00:00+00:00"
    assert cache_is_fresh(bars, fetched, date(2026, 9, 27), 60, now=_clock("2026-09-27T12:00:00"))
    assert cache_is_fresh(bars, fetched, date(2026, 9, 28), 60, now=_clock("2026-09-28T15:29:00"))
    assert cache_is_fresh(bars, fetched, date(2026, 9, 28), 60, now=_clock("2026-09-28T16:00:00")) is False
    current = _bars_ending(date(2026, 9, 28), 80)
    assert cache_is_fresh(current, fetched, date(2026, 9, 28), 60, now=_clock("2026-09-28T16:00:00"))
    # A file written at the close is still stale while the new session is missing.
    just_fetched = "2026-09-28T08:00:00+00:00"
    assert cache_is_fresh(bars, just_fetched, date(2026, 9, 28), 60, now=_clock("2026-09-28T16:00:00")) is False


def test_get_bars_merges_a_short_tail(tmp_path, monkeypatch):
    cached = _bars_ending(date(2026, 9, 24), 75)
    save_bars(tmp_path, "600000", cached, "tencent")
    tail = _bars_ending(date(2026, 9, 28), 15)
    calls: list[int] = []

    def fake_tencent(self, code, count):
        calls.append(count)
        return tail

    monkeypatch.setattr(MarketData, "_tencent_bars", fake_tencent)
    source = MarketData(tmp_path)
    result = source.get_bars("600000", 700, date(2026, 9, 28), now=_clock("2026-09-28T16:00:00"))
    assert calls == [TAIL_BARS]
    assert TAIL_BARS == 40
    loaded, _fetched = load_bars(tmp_path, "600000")
    assert loaded[0].date == cached[0].date
    assert loaded[-1].date == date(2026, 9, 28)
    assert result[-1].date == date(2026, 9, 28)


def test_empty_cache_requests_the_full_count(tmp_path, monkeypatch):
    calls: list[int] = []
    history = _bars_ending(date(2026, 9, 28), 40)

    def fake_tencent(self, code, count):
        calls.append(count)
        return history

    monkeypatch.setattr(MarketData, "_tencent_bars", fake_tencent)
    source = MarketData(tmp_path)
    source.get_bars("600000", 700, date(2026, 9, 28), now=_clock("2026-09-28T16:00:00"))
    assert calls == [700]


def test_tail_without_overlap_falls_back_to_a_full_download(tmp_path, monkeypatch):
    cached = _bars_ending(date(2026, 9, 24), 75)
    save_bars(tmp_path, "600000", cached, "tencent")
    full = _bars_ending(date(2026, 9, 28), 50)
    calls: list[int] = []

    def fake_tencent(self, code, count):
        calls.append(count)
        if count == TAIL_BARS:
            return [_bar(date(2026, 9, 28))]
        return full

    monkeypatch.setattr(MarketData, "_tencent_bars", fake_tencent)
    source = MarketData(tmp_path)
    source.get_bars("600000", 700, date(2026, 9, 28), now=_clock("2026-09-28T16:00:00"))
    assert calls == [TAIL_BARS, 700]
    loaded, _fetched = load_bars(tmp_path, "600000")
    assert loaded[0].date == full[0].date
    assert loaded[-1].date == date(2026, 9, 28)


def test_refresh_many_does_not_download_a_fresh_name(tmp_path, monkeypatch):
    save_bars(tmp_path, "600000", _bars_ending(date(2026, 9, 28), 70), "tencent")

    def boom(self, code, count, today, now=None):
        raise AssertionError(code)

    monkeypatch.setattr(MarketData, "get_bars", boom)
    source = MarketData(tmp_path)
    errors = source.refresh_many(["600000"], 700, date(2026, 9, 28), now=_clock("2026-09-28T16:00:00"))
    assert errors == {}


def test_load_settings_reads_strategies_and_ranker(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"strategies": ["ma_pullback"], "use_ranker": False}), encoding="utf-8")
    settings = load_settings(path)
    assert settings.strategies == ["ma_pullback"]
    assert settings.use_ranker is False
    path.write_text(json.dumps({"strategies": "ma_pullback, first_board_follow"}), encoding="utf-8")
    assert load_settings(path).strategies == ["ma_pullback", "first_board_follow"]
    assert load_settings(tmp_path / "missing.json").strategies == ["first_board_follow", "ma_pullback"]
    assert load_settings(tmp_path / "missing.json").use_ranker is False


def test_unknown_strategy_and_non_bool_ranker_are_rejected(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"strategies": ["nope"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="未知策略"):
        load_settings(path)
    path.write_text(json.dumps({"strategies": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="不能为空"):
        load_settings(path)
    path.write_text(json.dumps({"use_ranker": "false"}), encoding="utf-8")
    with pytest.raises(ValueError, match="use_ranker"):
        load_settings(path)


def test_daily_cli_overrides_config_and_defaults_to_none():
    bare = build_parser().parse_args(["daily", "--no-paper"])
    assert bare.strategies is None
    assert bare.ranker is None
    args = build_parser().parse_args(
        ["daily", "--strategies", "ma_pullback", "--no-ranker", "--no-paper"]
    )
    assert args.strategies == "ma_pullback"
    assert args.ranker is False
    settings = loose_settings()
    settings.strategies = ["first_board_follow", "ma_pullback"]
    settings.use_ranker = True
    apply_daily_overrides(settings, bare)
    assert settings.strategies == ["first_board_follow", "ma_pullback"]
    assert settings.use_ranker is True
    apply_daily_overrides(settings, args)
    assert settings.strategies == ["ma_pullback"]
    assert settings.use_ranker is False


def _panel(day: date, risk_on: bool = False) -> MarketPanel:
    info = MarketDay(
        breadth=0.5,
        breadth_count=1000,
        limit_up_count=1,
        broken_count=1,
        broken_rate=0.5,
        max_height=1,
        prev_limit_return=0.0,
        index_level=1.0,
        index_vs_ma=0.01,
        risk_on=risk_on,
    )
    return MarketPanel(days={day: info}, calendar=[day], limit_up={}, height={})


def test_publish_uses_configured_strategies_and_skips_the_ranker(tmp_path, monkeypatch):
    seen: dict = {}

    def build(symbols, settings, day, strategies, cap=None):
        seen["ids"] = [item.id for item in strategies]
        seen["cap"] = cap
        return []

    def load():
        raise AssertionError("ranker should stay off")

    monkeypatch.setattr("ashare.pipeline.build_recommendations", build)
    monkeypatch.setattr("ashare.pipeline.load_ranker", load)
    settings = loose_settings()
    settings.strategies = ["ma_pullback"]
    settings.use_ranker = False
    settings.top_n = 10
    settings.ml_pool = 80
    day = date(2026, 9, 24)
    path = publish_daily([], _panel(day), settings, day, tmp_path, None, [])
    assert seen == {"ids": ["ma_pullback"], "cap": 10}
    text = path.read_text(encoding="utf-8")
    assert "均线回踩" in text
    assert "排序模型已关闭" in text
    snapshot = json.loads((tmp_path / "market_latest.json").read_text(encoding="utf-8"))
    assert snapshot["use_ranker"] is False
    assert snapshot["model_ready"] is False


def test_publish_loads_the_ranker_only_when_enabled(tmp_path, monkeypatch):
    seen: dict = {}

    def build(symbols, settings, day, strategies, cap=None):
        seen["cap"] = cap
        return []

    def load():
        seen["loaded"] = True
        return None

    monkeypatch.setattr("ashare.pipeline.build_recommendations", build)
    monkeypatch.setattr("ashare.pipeline.load_ranker", load)
    settings = loose_settings()
    settings.strategies = ["ma_pullback"]
    settings.use_ranker = True
    settings.top_n = 10
    settings.ml_pool = 80
    day = date(2026, 9, 24)
    path = publish_daily([], _panel(day), settings, day, tmp_path, None, [])
    assert seen["loaded"] is True
    assert seen["cap"] == 80
    assert "模型未训练" in path.read_text(encoding="utf-8")
    snapshot = json.loads((tmp_path / "market_latest.json").read_text(encoding="utf-8"))
    assert snapshot["use_ranker"] is True


def test_run_daily_settles_after_refresh_unless_paper_is_off(monkeypatch, tmp_path):
    order: list[str] = []

    def load_universe(settings, today, cache_dir):
        order.append("load")
        return [], [], [], {}

    def settle_paper(settings, cache_dir, paper_path, today):
        order.append("settle")
        return ["已结算"]

    def attach(symbols):
        order.append("attach")

    def panel(symbols, settings):
        order.append("panel")
        return object()

    def latest(symbols):
        order.append("day")
        return date(2026, 9, 24)

    def publish(*args, **kwargs):
        order.append("publish")
        return tmp_path / "daily.md"

    monkeypatch.setattr("ashare.pipeline.load_universe", load_universe)
    monkeypatch.setattr("ashare.pipeline.settle_paper", settle_paper)
    monkeypatch.setattr("ashare.pipeline.attach_inferred_st", attach)
    monkeypatch.setattr("ashare.pipeline.build_market_panel", panel)
    monkeypatch.setattr("ashare.pipeline.latest_common_day", latest)
    monkeypatch.setattr("ashare.pipeline.publish_daily", publish)
    settings = loose_settings()
    run_daily(settings, date(2026, 9, 28), tmp_path, tmp_path, tmp_path / "ledger.json")
    assert order == ["load", "settle", "attach", "panel", "day", "publish"]
    order.clear()
    run_daily(settings, date(2026, 9, 28), tmp_path, tmp_path, None)
    assert order == ["load", "attach", "panel", "day", "publish"]
