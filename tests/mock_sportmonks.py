"""Local HTTP stand-in for SportMonks Football API v3.

Responses use the documented list/object envelope (``data``, ``pagination``,
``rate_limit``). The real ``SportMonksClient`` talks to this process through
``SPORTMONKS_API_BASE``; nothing in the dashboard is stubbed in Python.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from laliga.config import LA_LIGA_LEAGUE_ID
from laliga.synthetic import generate_synthetic_fixtures

PER_PAGE_DEFAULT = 50


class MockSportMonks:
    def __init__(self) -> None:
        self.fixtures = generate_synthetic_fixtures()
        self.mode = "ok"
        self._once_used = False
        self.hits: list[str] = []
        self.tokens_seen: list[str] = []
        self._lock = threading.Lock()
        handler = _make_handler(self)
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        host, port = self.httpd.server_address
        return f"http://{host}:{port}/v3/football"

    def set_mode(self, mode: str) -> None:
        with self._lock:
            self.mode = mode
            self._once_used = False

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def decide(self, token: str, path: str) -> tuple[int, dict[str, str], dict]:
        with self._lock:
            self.tokens_seen.append(token)
            self.hits.append(path)
            mode = self.mode
            if mode == "429-once" and not self._once_used:
                self._once_used = True
                mode = "429"
        if mode == "401":
            return 401, {}, {"message": "Unauthorized"}
        if mode == "403":
            return 403, {}, {"message": "You don't have access to this endpoint on your subscription."}
        if mode == "429":
            return (
                429,
                {"Retry-After": "0"},
                {"message": "Too Many Requests", "rate_limit": {"remaining": 0, "resets_in_seconds": 0}},
            )
        if not token:
            return 401, {}, {"message": "Unauthorized"}
        return 200, {}, self._ok_body(path)

    def _ok_body(self, path: str) -> dict:
        parsed = urlparse(path)
        query = parse_qs(parsed.query)
        relative = parsed.path
        marker = "/v3/football/"
        if marker in relative:
            relative = relative.split(marker, 1)[1]
        relative = relative.strip("/")
        if relative == f"leagues/{LA_LIGA_LEAGUE_ID}":
            seasons = _season_rows(self.fixtures)
            current = next(item for item in seasons if item["is_current"])
            return _envelope({"id": LA_LIGA_LEAGUE_ID, "name": "La Liga", "seasons": seasons, "currentseason": current})
        if relative == "fixtures":
            season_id = _filter_value(query, "fixtureSeasons")
            rows = [item for item in self.fixtures if str(item["season_id"]) == season_id]
            return _page(rows, query)
        if relative.startswith("fixtures/between/"):
            parts = relative.split("/")
            start, end = parts[-2], parts[-1]
            league = _filter_value(query, "fixtureLeagues")
            rows = []
            for item in self.fixtures:
                if league and str(item["league_id"]) != league:
                    continue
                day = str(item["starting_at"])[:10]
                if start <= day <= end:
                    rows.append(item)
            return _page(rows, query)
        return {"message": f"no result for {relative}"}


def _make_handler(mock: MockSportMonks):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            token = (parse_qs(parsed.query).get("api_token") or [""])[0]
            status, headers, payload = mock.decide(token, self.path)
            body = json.dumps(payload).encode("utf-8")
            if token and token.encode("utf-8") in body:
                status = 500
                body = b'{"message":"mock echoed the token"}'
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            for key, value in headers.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args) -> None:
            return

    return Handler


def _envelope(data) -> dict:
    return {"data": data, "rate_limit": {"remaining": 2500, "resets_in_seconds": 3600}}


def _page(rows: list, query: dict[str, list[str]]) -> dict:
    try:
        per_page = int((query.get("per_page") or [str(PER_PAGE_DEFAULT)])[0])
    except ValueError:
        per_page = PER_PAGE_DEFAULT
    per_page = max(1, min(per_page, 50))
    try:
        offset = int((query.get("cursor") or ["0"])[0])
    except ValueError:
        offset = 0
    chunk = rows[offset : offset + per_page]
    nxt = offset + per_page
    has_more = nxt < len(rows)
    pagination = {
        "count": len(chunk),
        "per_page": per_page,
        "current_page": None,
        "next_page": None,
        "has_more": has_more,
        "next_cursor": str(nxt) if has_more else None,
    }
    return {"data": chunk, "pagination": pagination, "rate_limit": {"remaining": 2500, "resets_in_seconds": 3600}}


def _filter_value(query: dict[str, list[str]], key: str) -> str:
    raw = (query.get("filters") or [""])[0]
    for part in raw.split(";"):
        name, _, value = part.partition(":")
        if name == key:
            return value
    return ""


def _season_rows(fixtures: list[dict]) -> list[dict]:
    grouped: dict[int, dict] = {}
    for item in fixtures:
        season_id = int(item["season_id"])
        bucket = grouped.setdefault(
            season_id,
            {"id": season_id, "name": item["season"]["name"], "days": []},
        )
        bucket["days"].append(str(item["starting_at"])[:10])
    latest = max(grouped, key=lambda season_id: min(grouped[season_id]["days"]))
    rows = []
    for season_id, bucket in grouped.items():
        rows.append(
            {
                "id": season_id,
                "sport_id": 1,
                "league_id": LA_LIGA_LEAGUE_ID,
                "name": bucket["name"],
                "finished": season_id != latest,
                "pending": season_id == latest,
                "is_current": season_id == latest,
                "starting_at": min(bucket["days"]),
                "ending_at": max(bucket["days"]),
            }
        )
    rows.sort(key=lambda item: item["starting_at"])
    return rows


def scheduled_day() -> str:
    """UTC date of the unplayed round produced by ``generate_synthetic_fixtures``."""

    today = datetime.now(timezone.utc).date()
    return (today + timedelta(days=3)).isoformat()
