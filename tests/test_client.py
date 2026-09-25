import json
from datetime import date, timedelta

import pytest

from laliga.data.client import SportMonksClient, SportMonksError, advance_pagination
from laliga.data.fetch import _reject_foreign_leagues, iter_date_windows, select_seasons


def _client(tmp_path, transport, **kwargs):
    return SportMonksClient(
        "client-token",
        tmp_path,
        transport=transport,
        sleep=kwargs.pop("sleep", lambda _seconds: None),
        min_interval=0,
        clock=lambda: 0.0,
        **kwargs,
    )


def test_cursor_pagination_and_cache_omits_token(tmp_path):
    calls = []

    def transport(url, params):
        calls.append((url, dict(params)))
        if "cursor" not in params:
            body = {
                "data": [{"id": 1, "league_id": 564}, {"id": 2, "league_id": 564}],
                "pagination": {"has_more": True, "next_cursor": "abc", "per_page": 50, "count": 2},
                "rate_limit": {"remaining": 100, "resets_in_seconds": 3600, "requested_entity": "Fixture"},
            }
        else:
            body = {
                "data": [{"id": 3, "league_id": 564}],
                "pagination": {"has_more": False, "next_cursor": None},
                "rate_limit": {"remaining": 99, "resets_in_seconds": 3590},
            }
        return 200, {}, body

    client = _client(tmp_path, transport)
    rows = list(client.paginate("fixtures", {"include": "participants;scores", "filters": "fixtureSeasons:1"}))
    assert [row["id"] for row in rows] == [1, 2, 3]
    assert calls[1][1]["cursor"] == "abc"
    assert calls[1][1]["api_token"] == "client-token"
    assert all("api_token" not in call[0] for call in calls)

    cached = list(tmp_path.glob("*.json"))
    assert cached
    blob = "\n".join(path.read_text(encoding="utf-8") for path in cached)
    assert "client-token" not in blob

    calls.clear()
    again = list(client.paginate("fixtures", {"include": "participants;scores", "filters": "fixtureSeasons:1"}))
    assert [row["id"] for row in again] == [1, 2, 3]
    assert calls == []


def test_next_page_url_strips_embedded_token(tmp_path):
    def transport(url, params):
        if params.get("page") == "2":
            return 200, {}, {"data": [{"id": 9}], "pagination": {"has_more": False}}
        return 200, {}, {
            "data": [{"id": 8}],
            "pagination": {
                "has_more": True,
                "next_page": (
                    "https://api.sportmonks.com/v3/football/fixtures"
                    "?page=2&api_token=secret-token&include=participants"
                ),
            },
        }

    client = _client(tmp_path, transport)
    rows = list(client.paginate("fixtures", {"include": "participants", "per_page": 50}))
    assert [row["id"] for row in rows] == [8, 9]
    blob = "\n".join(path.read_text(encoding="utf-8") for path in tmp_path.glob("*.json"))
    assert "secret-token" not in blob
    assert "client-token" not in blob


def test_page_number_fallback(tmp_path):
    seen = []

    def transport(url, params):
        seen.append(params.get("page"))
        if params.get("page") == "2":
            return 200, {}, {"data": [{"id": 2}], "pagination": {"has_more": False, "current_page": 2}}
        return 200, {}, {"data": [{"id": 1}], "pagination": {"has_more": True, "current_page": 1, "per_page": 50}}

    client = _client(tmp_path, transport)
    assert [row["id"] for row in client.paginate("fixtures", {})] == [1, 2]
    assert seen == [None, "2"]


def test_retries_http_429_using_retry_after(tmp_path):
    sleeps = []
    state = {"n": 0}

    def transport(url, params):
        state["n"] += 1
        if state["n"] == 1:
            return 429, {"retry-after": "0"}, {"message": "Too Many Requests"}
        return 200, {}, {"data": {"id": 1}, "rate_limit": {"remaining": 10}}

    client = _client(tmp_path, transport, sleep=sleeps.append)
    body = client.get("fixtures/1")
    assert body["data"]["id"] == 1
    assert sleeps == [0.0]


def test_low_remaining_budget_sleeps_then_refuses_long_waits(tmp_path):
    sleeps = []

    def transport(url, params):
        return 200, {}, {"data": [], "pagination": {"has_more": False}, "rate_limit": {"remaining": 0, "resets_in_seconds": 3}}

    client = _client(tmp_path, transport, sleep=sleeps.append)
    client.get("leagues/564")
    assert sleeps == [pytest.approx(3.5)]

    def transport_long(url, params):
        return 200, {}, {"data": [], "rate_limit": {"remaining": 1, "resets_in_seconds": 3600}}

    client = _client(tmp_path / "b", transport_long, sleep=sleeps.append, max_rate_limit_sleep=10)
    with pytest.raises(SportMonksError, match="剩余请求数"):
        client.get("leagues/564")


def test_missing_token_and_error_redaction(tmp_path):
    with pytest.raises(SportMonksError, match="SPORTMONKS_API_TOKEN"):
        SportMonksClient("", tmp_path, transport=lambda url, params: (200, {}, {}), min_interval=0)

    def transport(url, params):
        return 403, {}, {"message": f"subscription blocked for {params['api_token']}"}

    client = _client(tmp_path, transport)
    with pytest.raises(SportMonksError, match=r"\*\*\*") as caught:
        client.get("leagues/564")
    assert "client-token" not in str(caught.value)
    assert "免费计划" in str(caught.value)


def test_advance_pagination_prefers_cursor():
    path, params = advance_pagination(
        "fixtures",
        {"include": "scores", "per_page": "50"},
        {"has_more": True, "next_cursor": "xyz", "next_page": "https://example.test?page=2&api_token=nope"},
    )
    assert path == "fixtures"
    assert params["cursor"] == "xyz"
    assert "page" not in params
    assert "api_token" not in params


def test_date_windows_respect_100_day_limit():
    windows = list(iter_date_windows(date(2024, 1, 1), date(2024, 6, 1)))
    assert windows[0][0] == date(2024, 1, 1)
    assert windows[-1][1] == date(2024, 6, 1)
    for start, stop in windows:
        assert (stop - start).days + 1 <= 100
    for previous, nxt in zip(windows, windows[1:]):
        assert nxt[0] == previous[1] + timedelta(days=1)


def test_select_seasons_keeps_most_recent():
    seasons = [
        {"id": 1, "name": "2022/2023", "starting_at": "2022-08-01"},
        {"id": 3, "name": "2024/2025", "starting_at": "2024-08-01", "is_current": True},
        {"id": 2, "name": "2023/2024", "starting_at": "2023-08-01"},
    ]
    chosen = select_seasons(seasons, 2)
    assert [item["id"] for item in chosen] == [3, 2]


def test_foreign_league_rows_are_rejected():
    with pytest.raises(SportMonksError, match="非西甲"):
        _reject_foreign_leagues([{"league_id": 564}, {"league_id": 8}])


def test_cache_file_is_json_without_query_token(tmp_path):
    def transport(url, params):
        return 200, {}, {"data": {"ok": True}}

    client = _client(tmp_path, transport)
    client.get("leagues/564", {"include": "seasons"})
    document = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert document["query"]["include"] == "seasons"
    assert "api_token" not in document["query"]
