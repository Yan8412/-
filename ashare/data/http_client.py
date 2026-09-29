"""Small HTTP client with a polite interval between calls."""

from __future__ import annotations

import json
import threading
import time
from typing import Any

import requests


class RateLimiter:
    def __init__(self, min_interval: float) -> None:
        self.min_interval = min_interval
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if now < self._next:
                time.sleep(self._next - now)
            self._next = time.monotonic() + self.min_interval


def fetch_text(
    url: str,
    limiter: RateLimiter,
    timeout: float = 20.0,
    headers: dict[str, str] | None = None,
    encoding: str | None = None,
    attempts: int = 3,
) -> str:
    last_error: Exception | None = None
    merged = {"User-Agent": "Mozilla/5.0 ashare-short/0.1 (research)"}
    if headers:
        merged.update(headers)
    for attempt in range(max(1, attempts)):
        limiter.wait()
        try:
            response = requests.get(url, headers=merged, timeout=timeout)
            response.raise_for_status()
            if encoding:
                response.encoding = encoding
            else:
                response.encoding = response.encoding or "utf-8"
            return response.text
        except (requests.RequestException, OSError) as exc:
            last_error = exc
            time.sleep(0.4 * (attempt + 1))
    raise RuntimeError(f"请求失败: {url}") from last_error


def fetch_json(url: str, limiter: RateLimiter, timeout: float = 20.0) -> Any:
    text = fetch_text(url, limiter, timeout=timeout).lstrip("\ufeff").strip()
    if "=" in text[:40] and text.split("=", 1)[0].isidentifier():
        text = text.split("=", 1)[1]
    if not text or text == "null":
        return None
    return json.loads(text)
