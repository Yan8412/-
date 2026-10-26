"""Pre-match odds beyond 1X2: parse, de-vig, and store them beside matches.csv.

SportMonks market ids were checked against ``GET /v3/odds/markets/{id}``:

* 1 full-time result, 6 Asian handicap, 12 goals odd/even
* 26 first-half Asian handicap, 28 first-half goals
* 29 half-time/full-time, 30 half-time correct score, 31 half-time result
* 57 correct score, 80 goals over/under

The daily ``fetch`` does not request these. ``fetch-markets`` does, after the
existing 1X2 request. A missing market or a missing field leaves that block
empty. Quotes at or after kickoff are dropped. The earliest pre-match quote
per bookmaker is opening; the latest is closing. Several timestamps are
required before the report says opening and closing both exist.

Direct prices (HT/FT, correct score, half-time result, half-time correct
score, odd/even) are stored for comparison. The rate solver uses 1X2,
over/under, and Asian handicap only.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd

from laliga.model.devig import devig, devig_vector, raw_implied_from_decimal
from laliga.model.implied_goals import MarketTargets
from laliga.model.joint import HTFT_CODES

MARKET_NAMES = {
    1: "全场胜平负",
    6: "亚洲让球",
    12: "进球单双",
    26: "上半场亚洲让球",
    28: "上半场进球",
    29: "半全场",
    30: "半场正确比分",
    31: "半场胜平负",
    57: "正确比分",
    80: "进球大小",
}
EXTENDED_MARKET_IDS = (6, 12, 26, 28, 29, 30, 31, 57, 80)
ODDS_FILTER = "markets:" + ",".join(str(market_id) for market_id in (1, *EXTENDED_MARKET_IDS))
EXTENDED_FILTER = "markets:" + ",".join(str(market_id) for market_id in EXTENDED_MARKET_IDS)

_SCORE_RE = re.compile(r"(\d+)\s*[-:]\s*(\d+)")
_LINE_RE = re.compile(r"(-?\d+(?:\.\d+)?)")
_SIDE = {
    "home": "home",
    "1": "home",
    "draw": "draw",
    "x": "draw",
    "away": "away",
    "2": "away",
    "over": "over",
    "under": "under",
    "odd": "odd",
    "even": "even",
}
_HTFT_WORDS = {
    "home": "1",
    "1": "1",
    "draw": "X",
    "x": "X",
    "away": "2",
    "2": "2",
}


def quotes_path(root: Path) -> Path:
    return Path(root) / "processed" / "odds_quotes.json"


def load_quotes(root: Path) -> dict:
    path = quotes_path(root)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def save_quotes(root: Path, payload: dict) -> None:
    path = quotes_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def absorb_fixture_odds(root: Path, payload: dict, *, kickoff, home_id: int, away_id: int) -> bool:
    """Merge one fixture payload into ``odds_quotes.json``. True when any market was kept."""

    try:
        fixture_id = int(payload.get("id"))
    except (TypeError, ValueError):
        return False
    rows = _odd_rows(payload)
    if not rows:
        return False
    bundle = parse_odds_rows(rows, kickoff=kickoff, home_id=home_id, away_id=away_id)
    if not bundle.get("markets_seen"):
        return False
    bundle["fixture_id"] = fixture_id
    store = load_quotes(root)
    previous = store.get(str(fixture_id)) if isinstance(store.get(str(fixture_id)), dict) else {}
    merged = dict(previous)
    merged.update(bundle)
    seen = set(previous.get("markets_seen") or []) | set(bundle.get("markets_seen") or [])
    merged["markets_seen"] = sorted(seen)
    names = dict(previous.get("market_names") or {})
    names.update(bundle.get("market_names") or {})
    merged["market_names"] = {str(key): value for key, value in names.items()}
    merged["bookmakers"] = sorted(set(previous.get("bookmakers") or []) | set(bundle.get("bookmakers") or []))
    merged["opening_and_closing"] = bool(previous.get("opening_and_closing") or bundle.get("opening_and_closing"))
    store[str(fixture_id)] = merged
    save_quotes(root, store)
    return True


def parse_odds_rows(rows: list[dict], *, kickoff, home_id: int, away_id: int) -> dict:
    kickoff_ts = _stamp(kickoff)
    kept = [row for row in rows if isinstance(row, dict) and _is_prematch(row, kickoff_ts)]
    by_market: dict[int, list[dict]] = {}
    bookmakers: set[int] = set()
    history = False
    for row in kept:
        market_id = _int(row.get("market_id"))
        if market_id is None:
            continue
        book = _int(row.get("bookmaker_id"))
        if book is not None:
            bookmakers.add(book)
        by_market.setdefault(market_id, []).append(row)
    for market_rows in by_market.values():
        if _has_history(market_rows):
            history = True
            break
    bundle = {
        "bookmakers": sorted(bookmakers),
        "markets_seen": sorted(by_market),
        "market_names": {str(market_id): MARKET_NAMES.get(market_id, str(market_id)) for market_id in sorted(by_market)},
        "opening_and_closing": history,
        "home_team_id": home_id,
        "away_team_id": away_id,
    }
    ft = _result_market(by_market.get(1, ()))
    if ft:
        bundle["ft_1x2"] = ft
    ht = _result_market(by_market.get(31, ()))
    if ht:
        bundle["ht_1x2"] = ht
    ou = _totals_market(by_market.get(80, ()))
    if ou:
        bundle["ou"] = ou
    ht_ou = _totals_market(by_market.get(28, ()))
    if ht_ou:
        bundle["ht_ou"] = ht_ou
    ah = _handicap_market(by_market.get(6, ()))
    if ah:
        bundle["ah"] = ah
    ht_ah = _handicap_market(by_market.get(26, ()))
    if ht_ah:
        bundle["ht_ah"] = ht_ah
    htft = _labelled_market(by_market.get(29, ()), _htft_label)
    if htft:
        bundle["htft"] = htft
    scores = _labelled_market(by_market.get(57, ()), _score_label)
    if scores:
        bundle["correct_score"] = scores
    ht_scores = _labelled_market(by_market.get(30, ()), _score_label)
    if ht_scores:
        bundle["ht_correct_score"] = ht_scores
    odd_even = _labelled_market(by_market.get(12, ()), _odd_even_label)
    if odd_even:
        bundle["odd_even"] = odd_even
    return bundle


def targets_from_bundle(bundle: dict | None, *, when: str = "closing", devig_method: str = "proportional") -> MarketTargets | None:
    if not bundle:
        return None
    ft = _devig_triple((bundle.get("ft_1x2") or {}).get(when), devig_method)
    if ft is None:
        return None
    targets = MarketTargets(ft_probs=ft)
    targets.ou = _lines(bundle.get("ou"), when)
    targets.ah = _handicaps(bundle.get("ah"), when)
    targets.ht_ou = _lines(bundle.get("ht_ou"), when)
    targets.ht_ah = _handicaps(bundle.get("ht_ah"), when)
    return targets


def targets_from_row(row) -> MarketTargets | None:
    """1X2 plus optional football-data OU 2.5 and Asian handicap columns."""

    probs = _row_ft_probs(row)
    if probs is None:
        return None
    targets = MarketTargets(ft_probs=probs)
    over = _row_float(row, "ou25_pre_over")
    under = _row_float(row, "ou25_pre_under")
    if over and under:
        raw = raw_implied_from_decimal([over, under])[0]
        fair = devig_vector(raw, ("proportional",))
        if "proportional" in fair:
            targets.ou.append((2.5, float(fair["proportional"][0])))
    line = _row_float(row, "ah_pre_line", positive_ok=True)
    home = _row_float(row, "ah_pre_home")
    away = _row_float(row, "ah_pre_away")
    if line is not None and home and away:
        raw = raw_implied_from_decimal([home, away])[0]
        fair = devig_vector(raw, ("proportional",))
        if "proportional" in fair:
            targets.ah.append((float(line), float(fair["proportional"][0])))
    return targets


def direct_probabilities(bundle: dict | None, *, when: str = "closing") -> dict[str, dict]:
    """De-vigged direct markets for comparison. Incomplete books are flagged."""

    if not bundle:
        return {}
    found = {}
    for key, required in (
        ("htft", set(HTFT_CODES)),
        ("correct_score", None),
        ("ht_correct_score", None),
        ("ht_1x2", {"home", "draw", "away"}),
        ("odd_even", {"odd", "even"}),
    ):
        block = bundle.get(key)
        if not isinstance(block, dict):
            continue
        snapshot = block.get(when)
        if key == "ht_1x2":
            probs = _devig_triple(snapshot, "proportional")
            if probs is not None:
                found[key] = {"probs": {"home": probs[0], "draw": probs[1], "away": probs[2]}, "complete": True}
            continue
        if not isinstance(snapshot, dict):
            continue
        labels = snapshot.get("probs") or snapshot.get("labels")
        if not isinstance(labels, dict) or not labels:
            continue
        found[key] = {"probs": {str(name): float(value) for name, value in labels.items()}, "complete": bool(snapshot.get("complete"))}
        if required is not None:
            found[key]["complete"] = required.issubset(found[key]["probs"])
    return found


def _result_market(rows) -> dict | None:
    opening, closing, history = _book_snapshots(rows, _result_selection)
    if closing is None:
        return None
    payload = {"closing": _triple_snapshot(closing), "opening": None if opening is None else _triple_snapshot(opening)}
    payload["history"] = history
    return payload


def _totals_market(rows) -> dict | None:
    opening, closing, history = _book_snapshots(rows, _total_selection)
    if not closing:
        return None
    return {"closing": _average_lines(closing), "opening": _average_lines(opening or []), "history": history}


def _handicap_market(rows) -> dict | None:
    opening, closing, history = _book_snapshots(rows, _handicap_selection)
    if not closing:
        return None
    return {"closing": _average_handicaps(closing), "opening": _average_handicaps(opening or []), "history": history}


def _labelled_market(rows, labeler) -> dict | None:
    opening, closing, history = _book_snapshots(rows, lambda row: labeler(row))
    if not closing:
        return None
    return {
        "closing": _average_labels(closing),
        "opening": _average_labels(opening or []),
        "history": history,
    }


def _book_snapshots(rows, select):
    """Per bookmaker, the earliest and latest complete selection maps."""

    grouped: dict[int, list[tuple[pd.Timestamp, dict]]] = {}
    for row in rows:
        selected = select(row)
        if not selected:
            continue
        book = _int(row.get("bookmaker_id")) or 0
        stamp = _quote_time(row)
        if stamp is None:
            stamp = pd.Timestamp("1970-01-01", tz="UTC")
        grouped.setdefault(book, []).append((stamp, selected))
    if not grouped:
        return None, None, False
    opening_books = []
    closing_books = []
    history = False
    for quotes in grouped.values():
        quotes.sort(key=lambda item: item[0])
        if len({item[0] for item in quotes}) >= 2:
            history = True
        opening_books.append(_collapse(quotes[0][1] if False else _merge_at(quotes, earliest=True)))
        closing_books.append(_collapse(_merge_at(quotes, earliest=False)))
    opening_books = [item for item in opening_books if item]
    closing_books = [item for item in closing_books if item]
    return opening_books or None, closing_books or None, history


def _merge_at(quotes: list[tuple[pd.Timestamp, dict]], *, earliest: bool) -> dict:
    """First or last price of each selection. Nested over/under sides are merged."""

    ordered = quotes if earliest else list(reversed(quotes))
    chosen: dict = {}
    for _stamp, selected in ordered:
        for key, value in selected.items():
            if isinstance(value, dict):
                slot = chosen.setdefault(key, {})
                if isinstance(slot, dict):
                    for inner, price in value.items():
                        slot.setdefault(inner, price)
            else:
                chosen.setdefault(key, value)
    return chosen


def _collapse(selected: dict) -> dict:
    return selected


def _triple_snapshot(books: list[dict]) -> dict | None:
    raws = []
    for book in books:
        if all(side in book for side in ("home", "draw", "away")):
            raws.append([1.0 / float(book[side]) for side in ("home", "draw", "away")])
    if not raws:
        return None
    raw = [sum(column) / len(raws) for column in zip(*raws, strict=True)]
    priced = devig([raw])
    odds = [1.0 / value if value else None for value in priced["proportional"][0]]
    return {
        "raw": {"home": raw[0], "draw": raw[1], "away": raw[2]},
        "devig": {
            method: {"home": float(priced[method][0, 0]), "draw": float(priced[method][0, 1]), "away": float(priced[method][0, 2])}
            for method in priced
            if priced[method].shape[0] and math.isfinite(priced[method][0, 0])
        },
        "odds": {"home": odds[0], "draw": odds[1], "away": odds[2]},
        "books": len(raws),
    }


def _average_lines(books: list[dict]) -> list[dict]:
    bucket: dict[float, list[tuple[float, float]]] = {}
    for book in books:
        for line, pair in book.items():
            if not isinstance(pair, dict) or "over" not in pair or "under" not in pair:
                continue
            bucket.setdefault(float(line), []).append((float(pair["over"]), float(pair["under"])))
    lines = []
    for line, pairs in sorted(bucket.items()):
        over = sum(item[0] for item in pairs) / len(pairs)
        under = sum(item[1] for item in pairs) / len(pairs)
        fair = devig_vector([1.0 / over, 1.0 / under], ("proportional", "shin"))
        if "proportional" not in fair:
            continue
        lines.append(
            {
                "line": line,
                "odds_over": over,
                "odds_under": under,
                "over_prob": float(fair["proportional"][0]),
                "under_prob": float(fair["proportional"][1]),
                "shin_over_prob": None if "shin" not in fair else float(fair["shin"][0]),
                "books": len(pairs),
            }
        )
    return lines


def _average_handicaps(books: list[dict]) -> list[dict]:
    bucket: dict[float, list[tuple[float, float]]] = {}
    for book in books:
        for line, pair in book.items():
            if not isinstance(pair, dict) or "home" not in pair or "away" not in pair:
                continue
            bucket.setdefault(float(line), []).append((float(pair["home"]), float(pair["away"])))
    lines = []
    for line, pairs in sorted(bucket.items()):
        home = sum(item[0] for item in pairs) / len(pairs)
        away = sum(item[1] for item in pairs) / len(pairs)
        fair = devig_vector([1.0 / home, 1.0 / away], ("proportional",))
        if "proportional" not in fair:
            continue
        lines.append(
            {
                "handicap": line,
                "odds_home": home,
                "odds_away": away,
                "home_prob": float(fair["proportional"][0]),
                "books": len(pairs),
            }
        )
    return lines


def _average_labels(books: list[dict]) -> dict:
    bucket: dict[str, list[float]] = {}
    for book in books:
        for label, price in book.items():
            try:
                decimal = float(price)
            except (TypeError, ValueError):
                continue
            if decimal <= 1.0:
                continue
            bucket.setdefault(str(label), []).append(decimal)
    if not bucket:
        return {"labels": {}, "probs": {}, "complete": False}
    raw = {label: 1.0 / (sum(prices) / len(prices)) for label, prices in bucket.items()}
    fair = devig_vector(list(raw.values()), ("proportional",))
    if "proportional" not in fair:
        return {"labels": {}, "probs": {}, "complete": False}
    probs = {label: float(probability) for label, probability in zip(raw, fair["proportional"], strict=True)}
    return {"labels": probs, "probs": probs, "complete": len(probs) >= 2, "books": 1}


def _devig_triple(snapshot, method: str):
    if not isinstance(snapshot, dict):
        return None
    table = (snapshot.get("devig") or {}).get(method) or (snapshot.get("devig") or {}).get("proportional")
    if not isinstance(table, dict):
        return None
    try:
        probs = (float(table["home"]), float(table["draw"]), float(table["away"]))
    except (KeyError, TypeError, ValueError):
        return None
    if any(probability <= 0 or not math.isfinite(probability) for probability in probs):
        return None
    total = sum(probs)
    return tuple(probability / total for probability in probs)


def _lines(block, when: str) -> list[tuple[float, float]]:
    if not isinstance(block, dict):
        return []
    rows = block.get(when) or []
    found = []
    for row in rows:
        try:
            found.append((float(row["line"]), float(row["over_prob"])))
        except (KeyError, TypeError, ValueError):
            continue
    return found


def _handicaps(block, when: str) -> list[tuple[float, float]]:
    if not isinstance(block, dict):
        return []
    rows = block.get(when) or []
    found = []
    for row in rows:
        try:
            found.append((float(row["handicap"]), float(row["home_prob"])))
        except (KeyError, TypeError, ValueError):
            continue
    return found


def _result_selection(row: dict) -> dict:
    side = _side(row)
    price = _decimal(row.get("value"))
    if side in ("home", "draw", "away") and price:
        return {side: price}
    return {}


def _total_selection(row: dict) -> dict:
    side = _side(row)
    price = _decimal(row.get("value"))
    line = _line_value(row)
    if side in ("over", "under") and price and line is not None:
        return {line: {side: price}}
    return {}


def _handicap_selection(row: dict) -> dict:
    side = _side(row)
    price = _decimal(row.get("value"))
    line = _line_value(row)
    if side not in ("home", "away") or not price or line is None:
        return {}
    # Store the line from the home selection's point of view.
    home_line = line if side == "home" else -line
    return {home_line: {side: price}}


def _htft_label(row: dict) -> dict:
    price = _decimal(row.get("value"))
    if not price:
        return {}
    text = _label_text(row).replace(" ", "")
    if "/" in text:
        left, right = text.split("/", 1)
        code = f"{_HTFT_WORDS.get(left.lower(), left.upper()[:1])}/{_HTFT_WORDS.get(right.lower(), right.upper()[:1])}"
        if code in HTFT_CODES:
            return {code: price}
    return {}


def _score_label(row: dict) -> dict:
    price = _decimal(row.get("value"))
    if not price:
        return {}
    match = _SCORE_RE.search(_label_text(row))
    if not match:
        return {}
    return {f"{int(match.group(1))}-{int(match.group(2))}": price}


def _odd_even_label(row: dict) -> dict:
    side = _side(row)
    price = _decimal(row.get("value"))
    if side in ("odd", "even") and price:
        return {side: price}
    return {}


def _row_ft_probs(row) -> tuple[float, float, float] | None:
    raw = [_row_probability(row, f"raw_implied_{side}") for side in ("home", "draw", "away")]
    if all(value is not None for value in raw):
        fair = devig([raw])["proportional"][0]
        return float(fair[0]), float(fair[1]), float(fair[2])
    implied = [_row_probability(row, f"implied_{side}") for side in ("home", "draw", "away")]
    if all(value is not None for value in implied):
        total = sum(implied)
        return tuple(value / total for value in implied)
    odds = [_row_float(row, f"odds_{side}") for side in ("home", "draw", "away")]
    if all(value is not None and value > 1 for value in odds):
        fair = devig(raw_implied_from_decimal(odds))["proportional"][0]
        return float(fair[0]), float(fair[1]), float(fair[2])
    return None


def _row_probability(row, name: str) -> float | None:
    """A stored probability, not a decimal price. ``_row_float`` rejects numbers at or below 1."""

    number = _row_float(row, name, positive_ok=True)
    if number is None or not 0.0 < number < 1.0:
        return None
    return number


def _row_float(row, name: str, positive_ok: bool = False):
    if isinstance(row, dict):
        value = row.get(name)
    else:
        value = row[name] if name in getattr(row, "index", ()) else None
    try:
        if value is None or pd.isna(value):
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if not positive_ok and number <= 1.0:
        return None
    return number


def _odd_rows(payload: dict) -> list[dict]:
    rows = []
    for key in ("odds", "premiumOdds", "premiumodds"):
        chunk = payload.get(key)
        if isinstance(chunk, list):
            rows.extend(item for item in chunk if isinstance(item, dict))
    return rows


def _is_prematch(row: dict, kickoff: pd.Timestamp | None) -> bool:
    if row.get("inplay") in (True, 1, "1", "true", "True"):
        # The 1X2 aggregator ignores this flag and uses the timestamp. Do the same.
        pass
    stamp = _quote_time(row)
    if kickoff is not None and stamp is not None and stamp >= kickoff:
        return False
    return _decimal(row.get("value")) is not None


def _has_history(rows: list[dict]) -> bool:
    seen: dict[tuple, set[str]] = {}
    for row in rows:
        book = _int(row.get("bookmaker_id"))
        label = _label_text(row)
        stamp = _quote_time(row)
        if book is None or stamp is None:
            continue
        seen.setdefault((book, label, _line_value(row)), set()).add(stamp.isoformat())
    return any(len(stamps) >= 2 for stamps in seen.values())


def _quote_time(row: dict) -> pd.Timestamp | None:
    for key in ("latest_bookmaker_update", "updated_at", "created_at"):
        stamp = _stamp(row.get(key))
        if stamp is not None:
            return stamp
    return None


def _stamp(value) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    stamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(stamp):
        return None
    return pd.Timestamp(stamp)


def _side(row: dict) -> str | None:
    for key in ("label", "name", "original_label"):
        text = str(row.get(key) or "").strip().lower()
        if text in _SIDE:
            return _SIDE[text]
    return None


def _label_text(row: dict) -> str:
    for key in ("label", "name", "original_label"):
        text = str(row.get(key) or "").strip()
        if text:
            return text
    return ""


def _line_value(row: dict):
    for key in ("total", "handicap"):
        number = _maybe_float(row.get(key))
        if number is not None:
            return number
    match = _LINE_RE.search(_label_text(row))
    if match:
        return float(match.group(1))
    return None


def _decimal(value):
    number = _maybe_float(value)
    if number is None or number <= 1.0:
        return None
    return number


def _maybe_float(value):
    if value is None or value == "":
        return None
    try:
        number = float(str(value).replace("%", ""))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
