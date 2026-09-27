"""Small HTTP client with a polite interval between calls."""

from __future__ import annotations

import json
import time
from typing import Any

import requests


class RateLimiter:
    def __init__(self, min_interval: float) -> None:
        self.min_interval = min_interval
        self._next = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        if now < self._next:
            time.sleep(self._next - now)
        self._next = time.monotonic() + self.min_interval


def fetch_text(url: str, limiter: RateLimiter, timeout: float = 20.0) -> str:
    last_error: Exception | None = None
    headers = {"User-Agent": "Mozilla/5.0 ashare-short/0.1 (research)"}
    for attempt in range(3):
        limiter.wait()
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
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
