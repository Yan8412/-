"""Pre-match odds beyond 1X2: parse, de-vig, and store them beside matches.csv.

SportMonks market ids were checked against ``GET /v3/odds/markets/{id}``:

* 1 full-time result, 6 Asian handicap, 12 goals odd/even
* 26 first-half Asian handicap, 28 first-half goals
* 29 half-time/full-time, 30 half-time correct score, 31 half-time result
* 57 correct score, 80 goals over/under

The daily ``fetch`` does not request these. ``fetch-markets`` does, after the
existing 1X2 request. A missing market or a missing field leaves that block
empty. A quote is pre-match when it is not marked in-play and its timestamp
is before kickoff, or when a post-kickoff ``latest_bookmaker_update`` is only
a settlement touch (``stopped`` true, or a ``winning`` field). Opening exists
only when the same selection has two timestamps. One row per selection, even
with different clocks, is not a price history.

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
_NUMBER_RE = re.compile(r"[+-]?\d+(?:\.\d+)?")
_BARE_SIDE = frozenset({"home", "away", "draw", "over", "under", "odd", "even", "1", "2", "x", "yes", "no"})
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


def absorb_fixture_odds(
    root: Path,
    payload: dict,
    *,
    kickoff,
    home_id: int,
    away_id: int,
    store: dict | None = None,
    flush: bool = True,
) -> bool:
    """Merge one fixture payload into the quotes store.

    Pass the same ``store`` and ``flush=False`` while paging, then
    ``save_quotes`` once. The default still loads and writes the file, so a
    single call keeps the previous behaviour. True when any market was kept.
    """

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
    if store is None:
        store = load_quotes(root)
        flush = True
    previous = store.get(str(fixture_id)) if isinstance(store.get(str(fixture_id)), dict) else {}
    merged = dict(previous)
    merged.update(bundle)
    seen = set(previous.get("markets_seen") or []) | set(bundle.get("markets_seen") or [])
    merged["markets_seen"] = sorted(seen)
    names = dict(previous.get("market_names") or {})
    names.update(bundle.get("market_names") or {})
    merged["market_names"] = {str(key): value for key, value in names.items()}
    merged["bookmakers"] = sorted(set(previous.get("bookmakers") or []) | set(bundle.get("bookmakers") or []))
    history = {}
    if isinstance(previous.get("market_history"), dict):
        history.update({str(key): bool(value) for key, value in previous["market_history"].items()})
    history.update({str(key): bool(value) for key, value in (bundle.get("market_history") or {}).items()})
    merged["market_history"] = history
    merged["opening_and_closing"] = any(history.values())
    store[str(fixture_id)] = merged
    if flush:
        save_quotes(root, store)
    return True


def parse_odds_rows(rows: list[dict], *, kickoff, home_id: int, away_id: int) -> dict:
    kickoff_ts = _stamp(kickoff)
    kept = [row for row in rows if isinstance(row, dict) and is_prematch_quote(row, kickoff_ts)]
    by_market: dict[int, list[dict]] = {}
    bookmakers: set[int] = set()
    for row in kept:
        market_id = _int(row.get("market_id"))
        if market_id is None:
            continue
        book = _int(row.get("bookmaker_id"))
        if book is not None:
            bookmakers.add(book)
        by_market.setdefault(market_id, []).append(row)
    market_history = {str(market_id): False for market_id in by_market}
    bundle = {
        "bookmakers": sorted(bookmakers),
        "markets_seen": sorted(by_market),
        "market_names": {str(market_id): MARKET_NAMES.get(market_id, str(market_id)) for market_id in sorted(by_market)},
        "opening_and_closing": False,
        "market_history": market_history,
        "home_team_id": home_id,
        "away_team_id": away_id,
    }

    def keep(market_id: int, key: str, payload: dict | None) -> None:
        if not payload:
            return
        bundle[key] = payload
        market_history[str(market_id)] = bool(payload.get("history"))

    keep(1, "ft_1x2", _result_market(by_market.get(1, ())))
    keep(31, "ht_1x2", _result_market(by_market.get(31, ())))
    keep(80, "ou", _totals_market(by_market.get(80, ())))
    keep(28, "ht_ou", _totals_market(by_market.get(28, ())))
    keep(6, "ah", _handicap_market(by_market.get(6, ())))
    keep(26, "ht_ah", _handicap_market(by_market.get(26, ())))
    keep(29, "htft", _labelled_market(by_market.get(29, ()), _htft_label))
    keep(57, "correct_score", _labelled_market(by_market.get(57, ()), _score_label))
    keep(30, "ht_correct_score", _labelled_market(by_market.get(30, ()), _score_label))
    keep(12, "odd_even", _labelled_market(by_market.get(12, ()), _odd_even_label))
    bundle["opening_and_closing"] = any(market_history.values())
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
    """Asian handicap, with the home-line convention chosen per bookmaker.

    Some books put each side's own line on the selection (Home −1, Away +1).
    Others print the home line on both sides (both say −0.5). Negating the
    second kind splits one line into two and the cover probabilities stop
    rising as the home line gets easier. The convention with more complete
    pairs wins; a tie goes to the monotonic one, then to the home-line view
    when both sides literally quote the same numbers.
    """

    parsed = []
    for row in rows:
        side = _side(row)
        price = _decimal(row.get("value"))
        line = _line_value(row)
        if side not in ("home", "away") or not price or line is None:
            continue
        book = _int(row.get("bookmaker_id")) or 0
        stamp = _quote_time(row) or pd.Timestamp("1970-01-01", tz="UTC")
        parsed.append((book, stamp, side, float(line), price))
    if not parsed:
        return None
    by_book: dict[int, list[tuple]] = {}
    for item in parsed:
        by_book.setdefault(item[0], []).append(item)
    opening_books = []
    closing_books = []
    history = False
    for items in by_book.values():
        convention = _ah_convention(items)
        atoms: dict[tuple, list[tuple[pd.Timestamp, float]]] = {}
        for _book, stamp, side, line, price in items:
            home_line = line if convention == "home" or side == "home" else -line
            atoms.setdefault((_line_key(home_line), side), []).append((stamp, price))
        book_history = any(len({stamp for stamp, _price in quotes}) >= 2 for quotes in atoms.values())
        history = history or book_history
        closing_books.append(_handicap_book(atoms, earliest=False))
        if book_history:
            opening_books.append(_handicap_book(atoms, earliest=True))
    closing_books = [item for item in closing_books if item]
    if not closing_books:
        return None
    closing, dropped = _enforce_monotonic(_average_handicaps(closing_books))
    opening, _opening_dropped = _enforce_monotonic(_average_handicaps(opening_books))
    return {
        "closing": closing,
        "opening": opening if history else [],
        "history": history,
        "monotonic": True,
        "dropped_lines": dropped,
    }


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
    """Earliest and latest price of each selection.

    History is true only when one selection (one side, or one line and side)
    has two timestamps. Different selections updated a minute apart are still
    a single price.
    """

    grouped: dict[int, dict[tuple, list[tuple[pd.Timestamp, float]]]] = {}
    for row in rows:
        selected = select(row)
        if not selected:
            continue
        book = _int(row.get("bookmaker_id")) or 0
        stamp = _quote_time(row) or pd.Timestamp("1970-01-01", tz="UTC")
        slot = grouped.setdefault(book, {})
        for key, price in _atomic(selected):
            slot.setdefault(key, []).append((stamp, float(price)))
    if not grouped:
        return None, None, False
    opening_books = []
    closing_books = []
    history = False
    for atoms in grouped.values():
        book_history = any(len({stamp for stamp, _price in quotes}) >= 2 for quotes in atoms.values())
        history = history or book_history
        closing_books.append(_book_from_atoms(atoms, earliest=False))
        if book_history:
            opening_books.append(_book_from_atoms(atoms, earliest=True))
    opening_books = [item for item in opening_books if item]
    closing_books = [item for item in closing_books if item]
    return opening_books or None, closing_books or None, history


def _atomic(selected: dict) -> list[tuple[tuple, float]]:
    atoms = []
    for key, value in selected.items():
        if isinstance(value, dict):
            for inner, price in value.items():
                atoms.append(((key, inner), float(price)))
        else:
            atoms.append(((key,), float(value)))
    return atoms


def _book_from_atoms(atoms: dict, *, earliest: bool) -> dict:
    chosen: dict = {}
    for key, quotes in atoms.items():
        ordered = sorted(quotes, key=lambda item: item[0])
        price = ordered[0][1] if earliest else ordered[-1][1]
        if len(key) == 1:
            chosen[key[0]] = price
        else:
            line, side = key
            slot = chosen.setdefault(line, {})
            if isinstance(slot, dict):
                slot[side] = price
    return chosen


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
    """De-vig each book on its own labels, then average and renormalise.

    Averaging decimals on the union of every book's labels pulls probability
    toward scores that only one book priced. A label quoted by one book keeps
    that book's fair probability until the final renormalisation.
    """

    per_book: list[dict[str, float]] = []
    for book in books:
        raw: dict[str, float] = {}
        for label, price in book.items():
            try:
                decimal = float(price)
            except (TypeError, ValueError):
                continue
            if decimal <= 1.0:
                continue
            raw[str(label)] = 1.0 / decimal
        if len(raw) < 2:
            continue
        fair = devig_vector(list(raw.values()), ("proportional",))
        if "proportional" not in fair:
            continue
        per_book.append({label: float(probability) for label, probability in zip(raw, fair["proportional"], strict=True)})
    if not per_book:
        return {"labels": {}, "probs": {}, "complete": False}
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for probs in per_book:
        for label, probability in probs.items():
            totals[label] = totals.get(label, 0.0) + probability
            counts[label] = counts.get(label, 0) + 1
    averaged = {label: totals[label] / counts[label] for label in totals}
    scale = sum(averaged.values())
    if scale <= 0:
        return {"labels": {}, "probs": {}, "complete": False}
    probs = {label: value / scale for label, value in averaged.items()}
    return {"labels": probs, "probs": probs, "complete": len(probs) >= 2, "books": len(per_book)}


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


def _htft_label(row: dict) -> dict:
    price = _decimal(row.get("value"))
    if not price:
        return {}
    for key in ("name", "original_label", "label"):
        code = _htft_code(str(row.get(key) or ""))
        if code:
            return {code: price}
    return {}


def _htft_code(text: str) -> str | None:
    compact = text.replace(" ", "")
    if "/" not in compact:
        return None
    left, right = compact.split("/", 1)
    left_code = _HTFT_WORDS.get(left.lower())
    right_code = _HTFT_WORDS.get(right.lower())
    if left_code is None or right_code is None:
        return None
    code = f"{left_code}/{right_code}"
    return code if code in HTFT_CODES else None


def _score_label(row: dict) -> dict:
    """Correct score as home-away.

    Bookmaker 2 sends ``label`` ``2`` and ``name`` ``2-1``: the name is the
    winner's score first. Away therefore swaps to 1-2. A draw must be level.
    ``label`` ``1:2`` with the same ``name`` is already home-away.
    """

    price = _decimal(row.get("value"))
    if not price:
        return {}
    name = str(row.get("name") or "").strip()
    label = str(row.get("label") or "").strip()
    name_match = _SCORE_RE.search(name) if name else None
    label_match = _SCORE_RE.search(label) if label else None
    side = _SIDE.get(label.lower()) if label else None
    if name_match and side in ("home", "away", "draw") and label_match is None:
        winner = int(name_match.group(1))
        loser = int(name_match.group(2))
        if side == "away":
            home, away = loser, winner
        elif side == "draw":
            if winner != loser:
                return {}
            home = away = winner
        else:
            home, away = winner, loser
        return {f"{home}-{away}": price}
    match = label_match or name_match
    if match is None:
        original = str(row.get("original_label") or "")
        match = _SCORE_RE.search(original) if original else None
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


def is_prematch_quote(row: dict, kickoff) -> bool:
    """Keep a pre-match price, including a settlement touch after kickoff.

    SportMonks ``odds`` (not ``inplayOdds``) often rewrites
    ``latest_bookmaker_update`` when the selection is settled, so a finished
    match can show a timestamp at or after kickoff on the pre-match row.
    Those rows stay when ``stopped`` is true or ``winning`` is present.
    Anything else at or after kickoff is dropped, so a live price cannot
    enter the pre-match average. An explicit in-play flag or an in-play
    market description is always dropped.
    """

    if not isinstance(row, dict) or _explicit_inplay(row):
        return False
    if _decimal(row.get("value")) is None:
        return False
    kickoff_ts = kickoff if isinstance(kickoff, pd.Timestamp) else _stamp(kickoff)
    stamp = _quote_time(row)
    if kickoff_ts is None or stamp is None or stamp < kickoff_ts:
        return True
    return _settled_touch(row)


def _is_prematch(row: dict, kickoff: pd.Timestamp | None) -> bool:
    return is_prematch_quote(row, kickoff)


def _explicit_inplay(row: dict) -> bool:
    for key in ("inplay", "is_live"):
        if row.get(key) in (True, 1, "1", "true", "True"):
            return True
    text = str(row.get("market_description") or "").lower()
    return "inplay" in text or "in-play" in text or "in play" in text


def _settled_touch(row: dict) -> bool:
    if row.get("stopped") in (True, 1, "1", "true", "True"):
        return True
    return "winning" in row


def _ah_convention(items: list[tuple]) -> str:
    own_n, own_mono = _ah_view(items, "own")
    home_n, home_mono = _ah_view(items, "home")
    if (home_n, home_mono) > (own_n, own_mono):
        return "home"
    if (own_n, own_mono) > (home_n, home_mono):
        return "own"
    home_lines = {_line_key(line) for _book, _stamp, side, line, _price in items if side == "home"}
    away_lines = {_line_key(line) for _book, _stamp, side, line, _price in items if side == "away"}
    if home_lines and home_lines == away_lines:
        return "home"
    return "own"


def _ah_view(items: list[tuple], mode: str) -> tuple[int, bool]:
    grouped: dict[float, dict[str, float]] = {}
    for _book, _stamp, side, line, price in items:
        home_line = line if mode == "home" or side == "home" else -line
        grouped.setdefault(_line_key(home_line), {})[side] = price
    pairs = []
    for home_line, sides in grouped.items():
        if "home" not in sides or "away" not in sides:
            continue
        fair = devig_vector([1.0 / sides["home"], 1.0 / sides["away"]], ("proportional",))
        if "proportional" not in fair:
            continue
        pairs.append((home_line, float(fair["proportional"][0])))
    pairs.sort()
    return len(pairs), _probs_monotonic([probability for _line, probability in pairs])


def _probs_monotonic(probs: list[float]) -> bool:
    return all(probs[index + 1] + 1e-4 >= probs[index] for index in range(len(probs) - 1))


def _handicap_book(atoms: dict, *, earliest: bool) -> dict:
    chosen: dict[float, dict[str, float]] = {}
    for (home_line, side), quotes in atoms.items():
        ordered = sorted(quotes, key=lambda item: item[0])
        price = ordered[0][1] if earliest else ordered[-1][1]
        chosen.setdefault(home_line, {})[side] = price
    return chosen


def _enforce_monotonic(lines: list[dict]) -> tuple[list[dict], int]:
    """Drop the thinner line of each adjacent inversion until cover probs rise."""

    kept = sorted(lines, key=lambda row: float(row["handicap"]))
    dropped = 0
    while len(kept) >= 2:
        violator = None
        for index in range(len(kept) - 1):
            if kept[index + 1]["home_prob"] + 1e-4 < kept[index]["home_prob"]:
                violator = index
                break
        if violator is None:
            break
        left_books = int(kept[violator].get("books") or 1)
        right_books = int(kept[violator + 1].get("books") or 1)
        kept.pop(violator if left_books <= right_books else violator + 1)
        dropped += 1
    for row in kept:
        row["monotonic"] = True
    return kept, dropped


def _line_key(value: float) -> float:
    return round(float(value), 4)


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
        if _SCORE_RE.fullmatch(text.replace(" ", "")):
            continue
        parts = text.replace("+", " +").replace("-", " -").split()
        if parts and parts[0] in _SIDE and any(character.isdigit() for character in text):
            return _SIDE[parts[0]]
    return None


def _line_value(row: dict):
    """Total or handicap. Bare ``1`` / ``2`` are sides, not lines.

    A null ``total`` with ``name`` ``3.5`` uses the name. ``-1,-1.5`` and
    ``-1/-1.5`` are quarter lines, stored as their average.
    """

    for key in ("total", "handicap"):
        parsed = _line_from_field(row.get(key))
        if parsed is not None:
            return parsed
    for key in ("name", "label", "original_label"):
        parsed = _line_from_field(row.get(key))
        if parsed is not None:
            return parsed
    return None


def _line_from_field(raw) -> float | None:
    if raw is None or raw == "" or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        number = float(raw)
        return number if math.isfinite(number) else None
    text = str(raw).strip()
    if not text or text.lower() in _BARE_SIDE:
        return None
    numbers = [float(match) for match in _NUMBER_RE.findall(text)]
    if not numbers:
        return None
    if len(numbers) == 1:
        return numbers[0]
    return (numbers[0] + numbers[1]) / 2.0


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
