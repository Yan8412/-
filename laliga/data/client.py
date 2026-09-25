"""HTTP client for the SportMonks Football API v3.

Documented behaviour this client relies on
(https://docs.sportmonks.com/v3):

* Base URL ``https://api.sportmonks.com/v3/football``.
* Auth via the ``api_token`` query parameter on every request.
* Includes are semicolon-separated (``include=participants;scores``).
* Filters are ``key:value`` and combined with ``;`` (``filters=fixtureLeagues:564``).
* List payloads are ``{"data": [...], "pagination": {...}}``.
* Pagination is cursor-based (``next_cursor`` + ``has_more``). The older
  ``page`` / ``next_page`` style still works and is followed when no cursor
  is present. ``per_page`` max is 50.
* Each page counts as one request. The body may include
  ``rate_limit.remaining`` and ``rate_limit.resets_in_seconds``. HTTP 429
  is backed off using ``Retry-After`` or that reset hint.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import urllib.parse
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

Transport = Callable[[str, dict[str, str]], tuple[int, dict[str, str], Any]]
Sleeper = Callable[[float], None]


class SportMonksError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class SportMonksClient:
    def __init__(
        self,
        token: str,
        cache_dir: Path,
        *,
        base_url: str = "https://api.sportmonks.com/v3/football",
        transport: Transport | None = None,
        sleep: Sleeper = time.sleep,
        min_interval: float = 0.25,
        max_retries: int = 4,
        timeout: float = 30.0,
        max_rate_limit_sleep: float = 90.0,
        refresh: bool = False,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not token or not str(token).strip():
            raise SportMonksError(
                "缺少 SPORTMONKS_API_TOKEN。请在环境变量或 .env 中设置，不要把 token 写进代码。"
            )
        self.token = str(token).strip()
        self.cache_dir = Path(cache_dir)
        self.base_url = base_url.rstrip("/")
        self._transport = transport or self._http_transport
        self._sleep = sleep
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.timeout = timeout
        self.max_rate_limit_sleep = max_rate_limit_sleep
        self.refresh = refresh
        self._clock = clock
        self._last_request_at: float | None = None
        self._session: requests.Session | None = None

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        query = _clean_params(params)
        cache_path = self.cache_dir / f"{_cache_key(path, query)}.json"
        if not self.refresh and cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            body = cached.get("body")
            if isinstance(body, dict):
                logger.debug("cache hit %s", path)
                return body
        body = self._fetch_live(path, query)
        self._write_cache(cache_path, path, query, body)
        return body

    def paginate(self, path: str, params: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
        """Yield every row in ``data``, following cursor or page pagination."""

        original = _clean_params(params)
        original.setdefault("per_page", 50)
        next_path = path
        next_params = dict(original)
        seen: set[tuple[str, str]] = set()
        pages = 0
        while True:
            payload = self.get(next_path, next_params)
            data = payload.get("data", [])
            if isinstance(data, dict):
                yield data
                return
            if not isinstance(data, list):
                raise SportMonksError(f"意外的 data 类型：{type(data).__name__}")
            yield from data

            pagination = payload.get("pagination") or {}
            if not pagination.get("has_more"):
                return
            pages += 1
            if pages > 1000:
                raise SportMonksError("分页超过 1000 页，已中止。")
            step = advance_pagination(next_path, original, pagination)
            if step is None:
                raise SportMonksError("pagination.has_more 为 true，但没有下一页信息。")
            next_path, next_params = step
            marker = (next_path, _stable_query(next_params))
            if marker in seen:
                raise SportMonksError("检测到重复分页请求，已中止。")
            seen.add(marker)

    def _fetch_live(self, path: str, params: dict[str, str]) -> dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        delay = 1.0
        last_error = "请求失败"
        for attempt in range(self.max_retries):
            self._pace()
            query = dict(params)
            query["api_token"] = self.token
            try:
                status, headers, payload = self._transport(url, query)
            except requests.RequestException as exc:
                last_error = f"网络错误：{exc.__class__.__name__}"
                logger.warning("%s（第 %s 次）", last_error, attempt + 1)
                self._sleep(delay)
                delay = min(delay * 2, 30.0)
                continue

            if status == 429:
                wait = _retry_wait(headers, payload, delay)
                last_error = f"触发限流（HTTP 429），等待 {wait:.1f} 秒"
                logger.warning(last_error)
                self._sleep(wait)
                delay = min(delay * 2, 30.0)
                continue
            if status >= 500:
                last_error = f"服务端错误 HTTP {status}"
                logger.warning("%s（第 %s 次）", last_error, attempt + 1)
                self._sleep(delay)
                delay = min(delay * 2, 30.0)
                continue
            if status >= 400 or not isinstance(payload, dict):
                raise SportMonksError(self._format_error(status, payload), status=status)
            if "data" not in payload and payload.get("message"):
                raise SportMonksError(self._format_error(status, payload), status=status)
            self._honour_rate_limit(payload)
            return payload
        raise SportMonksError(last_error)

    def _honour_rate_limit(self, payload: dict[str, Any]) -> None:
        info = payload.get("rate_limit") or {}
        remaining = info.get("remaining")
        if remaining is None:
            return
        try:
            left = int(remaining)
        except (TypeError, ValueError):
            return
        if left > 1:
            return
        resets = info.get("resets_in_seconds")
        try:
            wait = float(resets if resets is not None else 1) + 0.5
        except (TypeError, ValueError):
            wait = 1.0
        if wait > self.max_rate_limit_sleep:
            raise SportMonksError(
                f"API 剩余请求数只有 {left}，官方重置时间约 {wait:.0f} 秒，"
                f"超过本程序单次等待上限（{self.max_rate_limit_sleep:.0f} 秒）。请稍后再运行 fetch。"
            )
        logger.info("rate_limit.remaining=%s，等待 %.1f 秒", left, wait)
        self._sleep(wait)

    def _pace(self) -> None:
        if self.min_interval <= 0 or self._last_request_at is None:
            self._last_request_at = self._clock()
            return
        elapsed = self._clock() - self._last_request_at
        if elapsed < self.min_interval:
            self._sleep(self.min_interval - elapsed)
        self._last_request_at = self._clock()

    def _http_transport(self, url: str, params: dict[str, str]) -> tuple[int, dict[str, str], Any]:
        if self._session is None:
            self._session = requests.Session()
            self._session.headers["Accept"] = "application/json"
            self._session.headers["User-Agent"] = "laliga-predictor/1.0"
        response = self._session.get(url, params=params, timeout=self.timeout)
        try:
            body: Any = response.json()
        except ValueError:
            body = {"message": response.text[:500]}
        return response.status_code, {k.lower(): v for k, v in response.headers.items()}, body

    def _format_error(self, status: int, payload: Any) -> str:
        if isinstance(payload, dict):
            message = str(payload.get("message") or payload)
        else:
            message = str(payload)
        message = message.replace(self.token, "***")
        hint = ""
        lowered = message.lower()
        if "subscription" in lowered or "don't have access" in lowered or "do not have access" in lowered:
            hint = " 当前订阅很可能不包含西甲（La Liga，联赛 ID 564）或该历史赛季。免费计划不含西甲。"
        elif status in (401, 403) or "unauthor" in lowered:
            hint = " 请检查 SPORTMONKS_API_TOKEN 是否正确。"
        return f"SportMonks API 错误（HTTP {status}）：{message}.{hint}"

    def _write_cache(self, path: Path, request_path: str, params: dict[str, str], body: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(
            {
                "path": request_path,
                "query": _redact_secrets(params, self.token),
                "body": _redact_secrets(body, self.token),
            },
            ensure_ascii=False,
        )
        if self.token and self.token in text:
            raise SportMonksError("拒绝写入缓存：序列化结果意外包含 API token。")
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)


_TOKEN_IN_URL = re.compile(r"(api_token=)[^&\s\"']+", re.IGNORECASE)


def _redact_secrets(value: Any, token: str) -> Any:
    """Drop API tokens from cached payloads, including next_page URLs."""

    if isinstance(value, str):
        text = value.replace(token, "***") if token else value
        return _TOKEN_IN_URL.sub(r"\1***", text)
    if isinstance(value, list):
        return [_redact_secrets(item, token) for item in value]
    if isinstance(value, dict):
        return {key: _redact_secrets(item, token) for key, item in value.items()}
    return value


def _clean_params(params: dict[str, Any] | None) -> dict[str, str]:
    cleaned: dict[str, str] = {}
    for key, value in (params or {}).items():
        if value is None or key == "api_token":
            continue
        cleaned[str(key)] = str(value)
    return cleaned


def _stable_query(params: dict[str, str]) -> str:
    items = sorted((key, value) for key, value in params.items() if key != "api_token")
    return urllib.parse.urlencode(items)


def _cache_key(path: str, params: dict[str, str]) -> str:
    raw = f"{path}?{_stable_query(params)}".encode()
    return hashlib.sha256(raw).hexdigest()


def _retry_wait(headers: dict[str, str], payload: Any, fallback: float) -> float:
    retry_after = headers.get("retry-after") or headers.get("Retry-After")
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            pass
    if isinstance(payload, dict):
        info = payload.get("rate_limit") or {}
        resets = info.get("resets_in_seconds")
        if resets is not None:
            try:
                return max(0.0, float(resets))
            except (TypeError, ValueError):
                pass
    return fallback


def advance_pagination(
    path: str,
    original: dict[str, str],
    pagination: dict[str, Any],
) -> tuple[str, dict[str, str]] | None:
    """Choose the next request from a SportMonks ``pagination`` object.

    Preference order: ``next_cursor``, then ``next_page`` (token stripped),
    then ``current_page + 1``.
    """

    cursor = pagination.get("next_cursor")
    if cursor:
        params = dict(original)
        params.pop("page", None)
        params["cursor"] = str(cursor)
        return path, params

    next_page = pagination.get("next_page")
    if next_page:
        return _merge_next_page(path, original, str(next_page))

    if "current_page" in pagination or "per_page" in pagination:
        try:
            current = int(pagination.get("current_page") or 1)
        except (TypeError, ValueError):
            current = 1
        params = dict(original)
        params.pop("cursor", None)
        params["page"] = str(current + 1)
        return path, params
    return None


def _merge_next_page(
    fallback_path: str, original: dict[str, str], next_page: str
) -> tuple[str, dict[str, str]]:
    parsed = urllib.parse.urlparse(next_page)
    query = urllib.parse.parse_qs(parsed.query)
    overlay = {key: values[0] for key, values in query.items() if key != "api_token" and values}
    params = dict(original)
    params.update(overlay)
    params.pop("api_token", None)
    if "cursor" in overlay:
        params.pop("page", None)
    if "page" in overlay:
        params.pop("cursor", None)
    url_path = parsed.path
    marker = "/football/"
    if marker in url_path:
        relative = url_path.split(marker, 1)[1]
    else:
        relative = fallback_path
    return relative or fallback_path, params
