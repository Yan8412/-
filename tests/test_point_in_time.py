"""Point-in-time ST, seal stats, and the limit-up pool. No network."""

import json
import sys
import time
from datetime import date, datetime

import pytest
import requests

from ashare.cli import build_parser
from ashare.data.intraday import annotate_rows, connect_tdx, derive_seal
from ashare.data.limit_pool import snapshot_limit_pools
from ashare.data.st_history import _store, apply_trading_status, load_status, sync_st_history
from ashare.filters import rejection_reason
from ashare.market_calendar import SHANGHAI
from ashare.rules.limits import limit_prices
from tests.helpers import flat_symbol, loose_settings


def test_main_board_st_uses_five_percent_and_other_boards_do_not():
    up, down = limit_prices(10, "600000", st=True)
    assert float(up) == 10.5
    assert float(down) == 9.5
    growth, _ = limit_prices(10, "300001", st=True)
    assert float(growth) == 12.0
    plain, _ = limit_prices(10, "600000", st=False)
    assert float(plain) == 11.0


def test_official_st_day_changes_the_limit_and_blocks_the_buy():
    series = flat_symbol("600000", [10.0] * 8)
    day = series.dates[5]
    apply_trading_status(series, {day: (True, True)})
    assert bool(series.st_flags[5])
    assert abs(series.limit_up[5] - 10.5) < 1e-9
    settings = loose_settings(min_history_bars=3, min_listed_bars=1, min_amount=1, min_swing_20d=0)
    assert rejection_reason(series, 5, settings) == "ST或风险警示"


def test_current_name_is_not_used_once_a_daily_mask_exists():
    series = flat_symbol("000001", [10.0] * 8, name="ST测试")
    apply_trading_status(series, {})
    settings = loose_settings(min_history_bars=3, min_listed_bars=1, min_amount=1, min_swing_20d=0)
    assert rejection_reason(series, 5, settings) is None


def test_suspension_flag_blocks_a_bar_that_still_has_volume():
    series = flat_symbol("600000", [10.0] * 8)
    apply_trading_status(series, {series.dates[5]: (False, False)})
    settings = loose_settings(min_history_bars=3, min_listed_bars=1, min_amount=1, min_swing_20d=0)
    assert bool(series.halted[5])
    assert series.volume[5] > 0
    assert rejection_reason(series, 5, settings) == "停牌"


def test_st_sync_caches_and_skips_a_fresh_file(tmp_path, monkeypatch):
    calls: list[tuple[str, str]] = []

    class _Login:
        error_code = "0"
        error_msg = "success"

    class _Cursor:
        error_code = "0"
        fields = ["date", "code", "isST", "tradestatus"]

        def __init__(self) -> None:
            self._rows = [["2026-09-24", "sh.600000", "1", "0"]]
            self._index = 0

        def next(self) -> bool:
            if self._index >= len(self._rows):
                return False
            self._current = self._rows[self._index]
            self._index += 1
            return True

        def get_row_data(self) -> list[str]:
            return self._current

    class _Baostock:
        def login(self):
            return _Login()

        def logout(self):
            return _Login()

        def query_history_k_data_plus(self, code, fields, start_date, end_date, frequency, adjustflag):
            del fields, frequency, adjustflag
            calls.append((code, start_date, end_date))
            return _Cursor()

    monkeypatch.setitem(sys.modules, "baostock", _Baostock())
    now = datetime(2026, 9, 24, 16, 0, tzinfo=SHANGHAI)
    first = sync_st_history(tmp_path, ["600000"], date(2026, 9, 24), now=now)
    assert first["updated"] == 1
    assert calls == [("sh.600000", "2023-01-01", "2026-09-24")]
    status = load_status(tmp_path, "600000")
    assert status is not None
    assert status[date(2026, 9, 24)] == (True, False)
    second = sync_st_history(tmp_path, ["600000"], date(2026, 9, 24), now=now)
    assert second["fresh"] == 1
    assert second["updated"] == 0
    assert len(calls) == 1


def test_rejected_code_does_not_drop_the_rest_of_the_batch(tmp_path, monkeypatch):
    logins = {"n": 0}

    class _Login:
        error_code = "0"
        error_msg = "success"

    class _Cursor:
        def __init__(self, code: str) -> None:
            self.error_code = "0" if code != "bj.920000" else "10001001"
            self.error_msg = "success" if code != "bj.920000" else "股票代码未标识sh或sz"
            self.fields = ["date", "code", "isST", "tradestatus"]
            self._rows = [["2026-09-24", code, "0", "1"]]
            self._index = 0

        def next(self) -> bool:
            if self.error_code != "0" or self._index >= len(self._rows):
                return False
            self._current = self._rows[self._index]
            self._index += 1
            return True

        def get_row_data(self) -> list[str]:
            return self._current

    class _Baostock:
        def login(self):
            logins["n"] += 1
            return _Login()

        def logout(self):
            return _Login()

        def query_history_k_data_plus(self, code, fields, start_date, end_date, frequency, adjustflag):
            del fields, start_date, end_date, frequency, adjustflag
            return _Cursor(code)

    monkeypatch.setitem(sys.modules, "baostock", _Baostock())
    now = datetime(2026, 9, 24, 16, 0, tzinfo=SHANGHAI)
    summary = sync_st_history(tmp_path, ["920000", "600000"], date(2026, 9, 24), now=now)
    assert logins["n"] == 1
    assert summary["updated"] == 2
    assert summary["failed"] == 0
    assert load_status(tmp_path, "600000")[date(2026, 9, 24)] == (False, True)
    assert load_status(tmp_path, "920000") == {}
    again = sync_st_history(tmp_path, ["920000", "600000"], date(2026, 9, 24), now=now)
    assert again["fresh"] == 2
    assert logins["n"] == 1


def test_daily_gap_uses_one_market_snapshot_not_one_query_per_stock(tmp_path, monkeypatch):
    _store(
        tmp_path,
        "600000",
        [{"date": "2026-09-24", "is_st": False, "trading": True}],
        empty_last=date(2026, 9, 24),
    )
    _store(
        tmp_path,
        "000001",
        [{"date": "2026-09-24", "is_st": False, "trading": True}],
        empty_last=date(2026, 9, 24),
    )
    history: list[str] = []
    snapshots: list[str] = []

    class _Login:
        error_code = "0"
        error_msg = "success"

    class _History:
        error_code = "0"
        fields = ["date", "code", "isST", "tradestatus"]

        def __init__(self) -> None:
            self._rows = [["2026-09-28", "sz.300001", "0", "1"]]
            self._index = 0

        def next(self) -> bool:
            if self._index >= len(self._rows):
                return False
            self._current = self._rows[self._index]
            self._index += 1
            return True

        def get_row_data(self) -> list[str]:
            return self._current

    class _Snapshot:
        error_code = "0"
        fields = ["code", "tradeStatus", "code_name"]

        def __init__(self, day: str) -> None:
            snapshots.append(day)
            self._rows = [
                ["sh.600000", "1", "浦发银行"],
                ["sz.000001", "0", "S*ST平安"],
                ["sh.000001", "1", "上证指数"],
            ]
            self._index = 0

        def next(self) -> bool:
            if self._index >= len(self._rows):
                return False
            self._current = self._rows[self._index]
            self._index += 1
            return True

        def get_row_data(self) -> list[str]:
            return self._current

    class _Baostock:
        def login(self):
            return _Login()

        def logout(self):
            return _Login()

        def query_history_k_data_plus(self, code, fields, start_date, end_date, frequency, adjustflag):
            del fields, start_date, end_date, frequency, adjustflag
            history.append(code)
            return _History()

        def query_all_stock(self, day=None):
            return _Snapshot(day)

    monkeypatch.setitem(sys.modules, "baostock", _Baostock())
    now = datetime(2026, 9, 28, 16, 0, tzinfo=SHANGHAI)
    summary = sync_st_history(
        tmp_path,
        ["600000", "000001", "300001"],
        date(2026, 9, 28),
        now=now,
        backfill_cap=1,
    )
    assert snapshots == ["2026-09-28"]
    assert history == ["sz.300001"]
    assert summary["bulk_days"] == 1
    assert summary["backfill"] == 1
    plain = load_status(tmp_path, "600000")
    assert plain is not None
    assert plain[date(2026, 9, 24)] == (False, True)
    assert plain[date(2026, 9, 28)] == (False, True)
    flagged = load_status(tmp_path, "000001")
    assert flagged is not None
    assert flagged[date(2026, 9, 28)] == (True, False)
    again = sync_st_history(
        tmp_path,
        ["600000", "000001", "300001"],
        date(2026, 9, 28),
        now=now,
        backfill_cap=1,
    )
    assert snapshots == ["2026-09-28"]
    assert history == ["sz.300001"]
    assert again["bulk_days"] == 0
    assert again["backfill"] == 0


def test_tdx_uses_fixed_hosts_then_auto_selection(monkeypatch):
    monkeypatch.setattr("ashare.data.intraday.known_tdx_hosts", lambda: [])
    calls: list[dict] = []

    class _Client:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def __enter__(self):
            if self.kwargs.get("host") == "bad:7709":
                raise TimeoutError("7709 response timed out during connect")
            return self

    def factory(**kwargs):
        calls.append(kwargs)
        return _Client(**kwargs)

    client = connect_tdx(["bad:7709", "116.205.183.150:7709"], time.monotonic() + 30, factory)
    assert isinstance(client, _Client)
    assert calls[0] == {
        "host": "bad:7709",
        "probe_hosts": False,
        "server_count": 1,
        "timeout": 5.0,
    }
    assert calls[1]["host"] == "116.205.183.150:7709"
    assert calls[1]["probe_hosts"] is False
    assert len(calls) == 2

    calls.clear()
    connect_tdx(["bad:7709"], time.monotonic() + 30, factory)
    assert "host" not in calls[-1]
    assert calls[-1]["timeout"] == 5.0

    with pytest.raises(TimeoutError):
        connect_tdx(["116.205.183.150:7709"], time.monotonic() - 1, factory)


def test_tdx_rotates_known_hosts_and_stops_at_the_budget(monkeypatch):
    monkeypatch.setattr(
        "ashare.data.intraday.known_tdx_hosts",
        lambda: ["cfg:7709", "a:7709", "b:7709", "c:7709"],
    )
    clock = {"now": 1_000.0}

    def monotonic():
        return clock["now"]

    monkeypatch.setattr("ashare.data.intraday.time.monotonic", monotonic)
    calls: list[str] = []

    def factory(**kwargs):
        assert "host" in kwargs
        assert kwargs["probe_hosts"] is False
        calls.append(kwargs["host"])
        clock["now"] += 10
        raise ConnectionError("closed")

    with pytest.raises(TimeoutError):
        connect_tdx(["cfg:7709"], 1_025.0, factory)
    assert calls == ["cfg:7709", "a:7709", "b:7709"]


def test_tdx_known_list_does_not_fall_back_to_auto_selection(monkeypatch):
    monkeypatch.setattr("ashare.data.intraday.known_tdx_hosts", lambda: ["extra:7709", "cfg:7709"])
    calls: list[dict] = []

    def factory(**kwargs):
        calls.append(kwargs)
        raise ConnectionError("closed")

    with pytest.raises(ConnectionError, match="已知服务器"):
        connect_tdx(["cfg:7709"], time.monotonic() + 30, factory)
    assert [item["host"] for item in calls] == ["cfg:7709", "extra:7709"]
    assert all(item["probe_hosts"] is False for item in calls)


def test_st_sync_command_does_not_need_a_ledger():
    args = build_parser().parse_args(["st-sync", "--backfill-cap", "0"])
    assert args.command == "st-sync"
    assert args.backfill_cap == 0


def test_derive_seal_counts_breaks_and_a_reclose():
    bars = [
        {"time": "09:31", "high": 10.2, "low": 10.0, "close": 10.1},
        {"time": "09:35", "high": 11.0, "low": 11.0, "close": 11.0},
        {"time": "09:40", "high": 11.0, "low": 10.5, "close": 10.6},
        {"time": "09:50", "high": 11.0, "low": 11.0, "close": 11.0},
    ]
    facts = derive_seal(bars, 11.0)
    assert facts["first_seal"] == "09:35"
    assert facts["broken_seals"] == 1
    assert facts["resealed"] is True
    touched = [{"time": "09:32", "high": 11.0, "low": 10.8, "close": 10.9}]
    opened = derive_seal(touched, 11.0)
    assert opened["broken_seals"] == 1
    assert opened["resealed"] is False


def test_minute_facts_do_not_change_the_rule_score():
    rows = [{"code": "600000", "strategy_id": "first_board_follow", "reason": "昨日首板", "score": 60.0}]
    annotate_rows(
        rows,
        [
            {
                "code": "600000",
                "first_seal": "09:35",
                "broken_seals": 1,
                "resealed": True,
                "auction_price": 10.2,
                "auction_volume": 15,
            }
        ],
    )
    assert rows[0]["score"] == 60.0
    assert rows[0]["first_seal"] == "09:35"
    assert rows[0]["resealed"] == "是"
    assert "首封 09:35" in rows[0]["reason"]


def test_pool_retries_429_and_does_not_leak_the_key(tmp_path, monkeypatch):
    monkeypatch.setenv("THS_API_KEY", "super-secret-key")
    monkeypatch.setattr("ashare.data.limit_pool.time.sleep", lambda _seconds: None)
    calls = {"n": 0}

    class _Response:
        def __init__(self, status, body):
            self.status_code = status
            self._body = body

        def json(self):
            return self._body

    def fake_get(url, params=None, headers=None, timeout=None):
        del url, timeout
        assert headers["X-api-key"] == "super-secret-key"
        calls["n"] += 1
        if calls["n"] == 1:
            return _Response(429, {})
        kind = "limit_up"
        if "limit-down" in (params and "") or False:
            kind = "limit_down"
        return _Response(
            200,
            {
                "code": 0,
                "data": {
                    "item": [
                        {
                            "ticker": "600000",
                            "name": "浦发银行",
                            "limit_up_time": "09:31",
                            "limit_up_reason": "银行",
                            "continue_day_cnt": 2,
                        }
                    ]
                },
            },
        )

    monkeypatch.setattr(requests, "get", fake_get)
    # requests is used as the module inside limit_pool, so patch that binding.
    monkeypatch.setattr("ashare.data.limit_pool.requests.get", fake_get)
    summary = snapshot_limit_pools(tmp_path, tmp_path, date(2026, 9, 24), loose_settings())
    assert summary["source"] == "ths"
    assert summary["limit_up_count"] >= 1
    assert summary["max_streak"] == 2
    assert calls["n"] >= 2
    text = json.dumps(summary, ensure_ascii=False)
    assert "super-secret-key" not in text
    assert (tmp_path / "pool_latest.json").exists()


def test_pool_retries_vendor_rate_limit_code(tmp_path, monkeypatch):
    monkeypatch.setenv("THS_API_KEY", "super-secret-key")
    monkeypatch.setattr("ashare.data.limit_pool.time.sleep", lambda _seconds: None)
    calls = {"n": 0}

    class _Response:
        def __init__(self, body):
            self.status_code = 200
            self._body = body

        def json(self):
            return self._body

    def fake_get(url, params=None, headers=None, timeout=None):
        del url, params, headers, timeout
        calls["n"] += 1
        if calls["n"] == 1:
            return _Response({"code": 4001, "message": "rate limit"})
        return _Response(
            {
                "code": 0,
                "data": {
                    "item": [
                        {
                            "ticker": "000001",
                            "name": "平安银行",
                            "first_limit_time": "09:35",
                            "open_times": 3,
                        }
                    ]
                },
            }
        )

    monkeypatch.setattr("ashare.data.limit_pool.requests.get", fake_get)
    summary = snapshot_limit_pools(tmp_path, tmp_path, date(2026, 9, 24), loose_settings())
    assert summary["source"] == "ths"
    assert calls["n"] >= 2
    assert "super-secret-key" not in json.dumps(summary)


def test_missing_key_and_missing_akshare_still_write_a_summary(tmp_path, monkeypatch):
    monkeypatch.delenv("THS_API_KEY", raising=False)
    monkeypatch.delenv("HITHINK_FINANCE_API_KEY", raising=False)

    def boom(day):
        raise RuntimeError("未安装 akshare")

    monkeypatch.setattr("ashare.data.limit_pool._akshare_rows", boom)
    summary = snapshot_limit_pools(tmp_path, tmp_path, date(2026, 9, 24), loose_settings())
    assert summary["source"] == "none"
    assert summary["limit_up_count"] == 0
    assert "THS_API_KEY" in summary["warning"]
    assert (tmp_path / "pool_latest.json").exists()
