"""La Liga results and pre-match odds from football-data.co.uk.

The daily SportMonks fetch does not call this. ``import-history`` writes
``processed/history.csv``. ``compare --history football-data`` is the only
command that reads it.

SP1 columns, from the site's notes: PSH/PSD/PSA are the Pinnacle prices
collected before the match (Friday afternoon for a weekend, Tuesday afternoon
for midweek). PSCH/PSCD/PSCA are the closing prices and are not available at
the time this project makes a prediction. B365H/D/A is the Bet365 price from
the same pre-closing collection. Avg/BbAv is the market average from that
same snapshot.
"""

from __future__ import annotations

import csv
import io
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from laliga.config import LA_LIGA_LEAGUE_ID
from laliga.data.teams import canonical_team, team_id_for
from laliga.model.devig import devig, raw_implied_from_decimal

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
HISTORY_FILENAME = "history.csv"
FIRST_SEASON = 2012

_ODDS_GROUPS = {
    "pin_pre": (("PSH", "PH"), ("PSD", "PD"), ("PSA", "PA")),
    "pin_close": (("PSCH",), ("PSCD",), ("PSCA",)),
    "b365_pre": (("B365H",), ("B365D",), ("B365A",)),
    "avg_pre": (("AvgH", "BbAvH"), ("AvgD", "BbAvD"), ("AvgA", "BbAvA")),
    "max_pre": (("MaxH", "BbMxH"), ("MaxD", "BbMxD"), ("MaxA", "BbMxA")),
}


class HistoryError(RuntimeError):
    pass


@dataclass
class HistoryBuild:
    frame: pd.DataFrame
    mismatches: list[dict] = field(default_factory=list)
    unmapped_sportmonks: list[str] = field(default_factory=list)
    duplicate_football_data: int = 0
    fallback_bet365: int = 0
    seasons: list[str] = field(default_factory=list)


def season_code(start_year: int) -> str:
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_url(start_year: int) -> str:
    return f"https://www.football-data.co.uk/mmz4281/{season_code(start_year)}/SP1.csv"


def history_path(root: Path) -> Path:
    return Path(root) / "processed" / HISTORY_FILENAME


def cache_path(root: Path, start_year: int) -> Path:
    return Path(root) / "cache" / "football-data" / f"SP1_{season_code(start_year)}.csv"


def download_season(start_year: int, root: Path, *, refresh: bool = False, fetch=None) -> str:
    """Return the CSV text, using the on-disk cache unless ``refresh`` is set."""

    path = cache_path(root, start_year)
    if path.exists() and not refresh:
        return path.read_text(encoding="utf-8-sig")
    text = (fetch or _fetch_url)(season_url(start_year))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def import_history(
    root: Path,
    sportmonks: pd.DataFrame | None = None,
    *,
    since: int = FIRST_SEASON,
    until: int | None = None,
    refresh: bool = False,
    fetch=None,
) -> HistoryBuild:
    """Download SP1 seasons, merge them with SportMonks, and write ``history.csv``.

    SportMonks wins on any match that exists in both: score, half-time score,
    fixture id, and team ids. A different full-time score is recorded and the
    SportMonks score is kept. Seasons that 404 after the last success are the
    end of the published archive, not a failed import.
    """

    if until is None:
        until = pd.Timestamp.now(tz="UTC").year
    texts: list[str] = []
    seen_success = False
    for year in range(since, until + 1):
        try:
            texts.append(download_season(year, root, refresh=refresh, fetch=fetch))
            seen_success = True
        except HistoryError as exc:
            if not seen_success or "404" not in str(exc):
                raise
            break
    if not texts:
        raise HistoryError(f"没有下载到 {since} 之后的西甲 CSV。")
    football = parse_seasons(texts)
    built = build_history(football, sportmonks if sportmonks is not None else pd.DataFrame())
    _write_history(root, built.frame)
    return built


def parse_seasons(texts: list[str]) -> pd.DataFrame:
    frames = [_parse_csv(text) for text in texts]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_history(football: pd.DataFrame, sportmonks: pd.DataFrame) -> HistoryBuild:
    """Merge football-data rows with SportMonks. SportMonks is the source of truth."""

    if football.empty and (sportmonks is None or sportmonks.empty):
        raise HistoryError("没有可合并的历史比赛。")
    sport = sportmonks if sportmonks is not None else pd.DataFrame()
    ids, unmapped = _sportmonks_ids(sport)
    indexed, ambiguous = _index_sportmonks(sport)
    used: set[int] = set()
    mismatches: list[dict] = []
    rows: list[dict] = []
    duplicates = 0
    fallback = 0
    seen_keys: set[tuple] = set()
    if not football.empty:
        ordered = football.sort_values(["starting_at", "fd_home", "fd_away"])
        for record in ordered.itertuples(index=False):
            key = (str(record.madrid_date), record.canonical_home, record.canonical_away)
            if key in seen_keys:
                duplicates += 1
                continue
            seen_keys.add(key)
            home_id = team_id_for(record.fd_home, ids)
            away_id = team_id_for(record.fd_away, ids)
            matched = None if key in ambiguous else indexed.get(key)
            if matched is not None:
                used.add(int(matched["fixture_id"]))
                score_home = int(matched["home_goals_ft"])
                score_away = int(matched["away_goals_ft"])
                mismatch = score_home != int(record.home_goals_ft) or score_away != int(record.away_goals_ft)
                if mismatch:
                    mismatches.append(
                        {
                            "date": key[0],
                            "home": record.fd_home,
                            "away": record.fd_away,
                            "football_data": [int(record.home_goals_ft), int(record.away_goals_ft)],
                            "sportmonks": [score_home, score_away],
                            "fixture_id": int(matched["fixture_id"]),
                        }
                    )
                base = _sportmonks_base(matched, home_id, away_id)
                base["history_source"] = "both"
                base["ft_score_mismatch"] = 1 if mismatch else 0
            else:
                base = _football_base(record, home_id, away_id)
                base["history_source"] = "football-data"
                base["ft_score_mismatch"] = 0
            _copy_odds(base, record)
            fallback += _set_fair_price(base)
            rows.append(base)
    if not sport.empty:
        for _, matched in sport.iterrows():
            fixture_id = int(matched["fixture_id"])
            if fixture_id in used or str(matched.get("status") or "") not in {"finished", "scheduled"}:
                continue
            home_name = str(matched.get("home_team_name") or "")
            away_name = str(matched.get("away_team_name") or "")
            try:
                home_id = team_id_for(home_name, ids)
                away_id = team_id_for(away_name, ids)
            except KeyError:
                home_id = int(matched["home_team_id"])
                away_id = int(matched["away_team_id"])
            base = _sportmonks_base(matched, home_id, away_id)
            base["history_source"] = "sportmonks"
            base["ft_score_mismatch"] = 0
            _set_fair_price(base)
            rows.append(base)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise HistoryError("合并后没有比赛。")
    frame["starting_at"] = pd.to_datetime(frame["starting_at"], utc=True)
    frame = frame.sort_values(["starting_at", "fixture_id"]).reset_index(drop=True)
    seasons = sorted({str(value) for value in frame["season_name"].dropna() if str(value)})
    return HistoryBuild(
        frame=frame,
        mismatches=mismatches,
        unmapped_sportmonks=unmapped,
        duplicate_football_data=duplicates,
        fallback_bet365=fallback,
        seasons=seasons,
    )


def load_history(root: Path, *, since: int | None = None) -> pd.DataFrame:
    path = history_path(root)
    if not path.exists():
        raise HistoryError(
            "还没有 football-data 历史表。先运行 "
            "python -m laliga import-history --since 2012"
        )
    frame = pd.read_csv(path)
    if frame.empty:
        raise HistoryError("history.csv 是空的。")
    frame["starting_at"] = pd.to_datetime(frame["starting_at"], utc=True, errors="coerce")
    if since is not None and "season_start" in frame.columns:
        frame = frame[pd.to_numeric(frame["season_start"], errors="coerce") >= int(since)].copy()
    if frame.empty:
        raise HistoryError(f"history.csv 里没有 {since} 年之后开赛的赛季。")
    return frame.reset_index(drop=True)


def format_history_build(built: HistoryBuild) -> str:
    finished = int((built.frame["status"] == "finished").sum()) if "status" in built.frame else len(built.frame)
    lines = [
        f"历史表 {len(built.frame)} 场，其中完场 {finished} 场，赛季 {len(built.seasons)} 个。",
        "2024/25 之后若两边都有，比分、半场和球队 id 用 SportMonks。football-data 只补充更早的赛季和赛前赔率。",
        f"Pinnacle 赛前盘缺失、改用 Bet365 赛前盘的比赛：{built.fallback_bet365} 场。",
        f"football-data 全场比分和 SportMonks 不一致：{len(built.mismatches)} 场。保留 SportMonks 的比分。",
    ]
    for item in built.mismatches[:15]:
        fd_score = "-".join(str(goal) for goal in item["football_data"])
        sm_score = "-".join(str(goal) for goal in item["sportmonks"])
        lines.append(f"  {item['date']} {item['home']} vs {item['away']}：football-data {fd_score}，SportMonks {sm_score}")
    if len(built.mismatches) > 15:
        lines.append(f"  其余 {len(built.mismatches) - 15} 场写在 history.csv 的 ft_score_mismatch 列。")
    if built.duplicate_football_data:
        lines.append(f"football-data 里同一天同一对球队的重复行：{built.duplicate_football_data}，已丢掉多余的。")
    if built.unmapped_sportmonks:
        names = "、".join(built.unmapped_sportmonks[:12])
        lines.append(f"SportMonks 队名还没有对照，这些场保留原 id、不与 football-data 合并：{names}")
    lines.append(f"文件：processed/{HISTORY_FILENAME}。每日 fetch 不会读它，也不会改它。")
    return "\n".join(lines)


def _fetch_url(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            status = getattr(response, "status", 200)
            if status == 404:
                raise HistoryError(f"404 {url}")
            return response.read().decode("utf-8-sig")
    except urllib.error.HTTPError as exc:
        raise HistoryError(f"{exc.code} {url}") from exc
    except urllib.error.URLError as exc:
        raise HistoryError(f"下载失败 {url}：{exc.reason}") from exc


def _parse_csv(text: str) -> pd.DataFrame:
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    rows = []
    for raw in reader:
        if (raw.get("Div") or "").strip() not in {"", "SP1"}:
            continue
        home = (raw.get("HomeTeam") or "").strip()
        away = (raw.get("AwayTeam") or "").strip()
        if not home or not away:
            continue
        try:
            canonical_home = canonical_team(home)
            canonical_away = canonical_team(away)
        except KeyError as exc:
            raise HistoryError(str(exc)) from exc
        kickoff, madrid_date, season_start = _kickoff(raw)
        if kickoff is None:
            continue
        ft_home = _int_cell(raw.get("FTHG"))
        ft_away = _int_cell(raw.get("FTAG"))
        if ft_home is None or ft_away is None:
            continue
        row = {
            "fd_fixture_id": 9_000_000_000 + season_start * 10_000 + len(rows) + 1,
            "fd_home": home,
            "fd_away": away,
            "canonical_home": canonical_home,
            "canonical_away": canonical_away,
            "starting_at": kickoff,
            "madrid_date": madrid_date,
            "season_start": season_start,
            "season_name": f"{season_start}/{season_start + 1}",
            "home_goals_ft": ft_home,
            "away_goals_ft": ft_away,
            "home_goals_ht": _int_cell(raw.get("HTHG")),
            "away_goals_ht": _int_cell(raw.get("HTAG")),
        }
        for group, pairs in _ODDS_GROUPS.items():
            values = [_float_cell(_first(raw, options)) for options in pairs]
            if any(value is None for value in values):
                values = [None, None, None]
            for side, value in zip(("home", "draw", "away"), values, strict=True):
                row[f"{group}_{side}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _kickoff(raw: dict) -> tuple[pd.Timestamp | None, str | None, int | None]:
    date_text = (raw.get("Date") or "").strip()
    time_text = (raw.get("Time") or "").strip() or "16:00"
    parsed = None
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            parsed = pd.to_datetime(date_text, format=fmt)
            break
        except (ValueError, TypeError):
            continue
    if parsed is None or pd.isna(parsed):
        return None, None, None
    clock = time_text if len(time_text) == 5 else "16:00"
    try:
        local = pd.Timestamp(f"{parsed.date().isoformat()} {clock}").tz_localize(
            "Europe/Madrid", nonexistent="shift_forward", ambiguous=True
        )
    except Exception:
        local = pd.Timestamp(f"{parsed.date().isoformat()} 16:00").tz_localize("Europe/Madrid")
    season_start = int(local.year if local.month >= 7 else local.year - 1)
    return local.tz_convert("UTC"), local.date().isoformat(), season_start


def _sportmonks_ids(frame: pd.DataFrame) -> tuple[dict[str, int], list[str]]:
    if frame is None or frame.empty:
        return {}, []
    counts: dict[str, dict[int, int]] = {}
    unmapped: set[str] = set()
    for column_name, column_id in (("home_team_name", "home_team_id"), ("away_team_name", "away_team_id")):
        if column_name not in frame.columns:
            continue
        for name, team_id in zip(frame[column_name], frame[column_id], strict=False):
            if pd.isna(team_id):
                continue
            try:
                canonical = canonical_team(str(name))
            except KeyError:
                if str(name).strip():
                    unmapped.add(str(name).strip())
                continue
            counts.setdefault(canonical, {})
            counts[canonical][int(team_id)] = counts[canonical].get(int(team_id), 0) + 1
    chosen = {name: max(ids, key=ids.get) for name, ids in counts.items()}
    return chosen, sorted(unmapped)


def _index_sportmonks(frame: pd.DataFrame) -> tuple[dict[tuple, pd.Series], set[tuple]]:
    indexed: dict[tuple, pd.Series] = {}
    ambiguous: set[tuple] = set()
    if frame is None or frame.empty:
        return indexed, ambiguous
    finished = frame[frame["status"] == "finished"] if "status" in frame.columns else frame
    for _, row in finished.iterrows():
        if pd.isna(row.get("home_goals_ft")) or pd.isna(row.get("away_goals_ft")):
            continue
        try:
            home = canonical_team(str(row.get("home_team_name") or ""))
            away = canonical_team(str(row.get("away_team_name") or ""))
        except KeyError:
            continue
        kickoff = pd.Timestamp(row["starting_at"])
        if kickoff.tzinfo is None:
            kickoff = kickoff.tz_localize("UTC")
        madrid = kickoff.tz_convert("Europe/Madrid")
        key = (madrid.date().isoformat(), home, away)
        if key in indexed or key in ambiguous:
            ambiguous.add(key)
            indexed.pop(key, None)
            continue
        indexed[key] = row
    return indexed, ambiguous


def _sportmonks_base(row: pd.Series, home_id: int, away_id: int) -> dict:
    kickoff = pd.Timestamp(row["starting_at"])
    if kickoff.tzinfo is None:
        kickoff = kickoff.tz_localize("UTC")
    else:
        kickoff = kickoff.tz_convert("UTC")
    madrid = kickoff.tz_convert("Europe/Madrid")
    season_start = madrid.year if madrid.month >= 7 else madrid.year - 1
    base = _empty_odds()
    base.update(
        {
            "fixture_id": int(row["fixture_id"]),
            "league_id": int(row["league_id"]) if pd.notna(row.get("league_id")) else LA_LIGA_LEAGUE_ID,
            "season_id": int(row["season_id"]) if pd.notna(row.get("season_id")) else season_start,
            "season_name": str(row.get("season_name") or f"{season_start}/{season_start + 1}"),
            "season_start": season_start,
            "round_id": None if pd.isna(row.get("round_id")) else int(row["round_id"]),
            "round_name": str(row.get("round_name") or ""),
            "starting_at": kickoff,
            "state_id": None if pd.isna(row.get("state_id")) else int(row["state_id"]),
            "state_name": str(row.get("state_name") or ""),
            "home_team_id": home_id,
            "home_team_name": str(row.get("home_team_name") or ""),
            "away_team_id": away_id,
            "away_team_name": str(row.get("away_team_name") or ""),
            "home_goals_ht": None if pd.isna(row.get("home_goals_ht")) else int(row["home_goals_ht"]),
            "away_goals_ht": None if pd.isna(row.get("away_goals_ht")) else int(row["away_goals_ht"]),
            "home_goals_ft": None if pd.isna(row.get("home_goals_ft")) else int(row["home_goals_ft"]),
            "away_goals_ft": None if pd.isna(row.get("away_goals_ft")) else int(row["away_goals_ft"]),
            "status": str(row.get("status") or "finished"),
            "home_xg": _optional_float(row, "home_xg"),
            "away_xg": _optional_float(row, "away_xg"),
            "home_xga": _optional_float(row, "home_xga"),
            "away_xga": _optional_float(row, "away_xga"),
            "raw_implied_home": _optional_float(row, "raw_implied_home"),
            "raw_implied_draw": _optional_float(row, "raw_implied_draw"),
            "raw_implied_away": _optional_float(row, "raw_implied_away"),
            "implied_home": _optional_float(row, "implied_home"),
            "implied_draw": _optional_float(row, "implied_draw"),
            "implied_away": _optional_float(row, "implied_away"),
            "fair_book": "sportmonks" if _optional_float(row, "raw_implied_home") is not None or _optional_float(row, "implied_home") is not None else "",
        }
    )
    return base


def _football_base(record, home_id: int, away_id: int) -> dict:
    base = _empty_odds()
    base.update(
        {
            "fixture_id": int(record.fd_fixture_id),
            "league_id": LA_LIGA_LEAGUE_ID,
            "season_id": int(record.season_start),
            "season_name": record.season_name,
            "season_start": int(record.season_start),
            "round_id": None,
            "round_name": "",
            "starting_at": pd.Timestamp(record.starting_at),
            "state_id": 5,
            "state_name": "FT",
            "home_team_id": home_id,
            "home_team_name": record.fd_home,
            "away_team_id": away_id,
            "away_team_name": record.fd_away,
            "home_goals_ht": None if pd.isna(record.home_goals_ht) else int(record.home_goals_ht),
            "away_goals_ht": None if pd.isna(record.away_goals_ht) else int(record.away_goals_ht),
            "home_goals_ft": int(record.home_goals_ft),
            "away_goals_ft": int(record.away_goals_ft),
            "status": "finished",
            "fair_book": "",
        }
    )
    return base


def _copy_odds(base: dict, record) -> None:
    for group in _ODDS_GROUPS:
        for side in ("home", "draw", "away"):
            value = getattr(record, f"{group}_{side}")
            base[f"{group}_{side}"] = None if value is None or (isinstance(value, float) and pd.isna(value)) else float(value)


def _set_fair_price(base: dict) -> int:
    """Point ``raw_implied_*`` at Pinnacle pre-closing, else Bet365, else leave SportMonks."""

    pinnacle = [base.get(f"pin_pre_{side}") for side in ("home", "draw", "away")]
    bet365 = [base.get(f"b365_pre_{side}") for side in ("home", "draw", "away")]
    if all(_positive_odds(value) for value in pinnacle):
        _write_fair(base, pinnacle, "pinnacle_pre")
        return 0
    if all(_positive_odds(value) for value in bet365):
        _write_fair(base, bet365, "bet365_pre")
        return 1
    return 0


def _write_fair(base: dict, odds: list[float], book: str) -> None:
    raw = raw_implied_from_decimal(np.asarray(odds, dtype=float))[0]
    fair = devig(raw)["proportional"][0]
    for index, side in enumerate(("home", "draw", "away")):
        base[f"raw_implied_{side}"] = float(raw[index])
        base[f"implied_{side}"] = float(fair[index])
    base["fair_book"] = book


def _empty_odds() -> dict:
    base = {
        "home_xg": None,
        "away_xg": None,
        "home_xga": None,
        "away_xga": None,
        "odds_home": None,
        "odds_draw": None,
        "odds_away": None,
        "implied_home": None,
        "implied_draw": None,
        "implied_away": None,
        "raw_implied_home": None,
        "raw_implied_draw": None,
        "raw_implied_away": None,
    }
    for group in (*_ODDS_GROUPS,):
        for side in ("home", "draw", "away"):
            base[f"{group}_{side}"] = None
    return base


def _write_history(root: Path, frame: pd.DataFrame) -> None:
    path = history_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame.copy()
    out["starting_at"] = pd.to_datetime(out["starting_at"], utc=True).dt.strftime("%Y-%m-%d %H:%M:%S%z")
    out.to_csv(path, index=False)


def _first(raw: dict, options: tuple[str, ...]) -> str | None:
    for name in options:
        if raw.get(name) not in (None, ""):
            return raw.get(name)
    return None


def _int_cell(value) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _float_cell(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number <= 1.0:
        return None
    return number


def _optional_float(row: pd.Series, column: str) -> float | None:
    if column not in row.index or pd.isna(row[column]):
        return None
    try:
        return float(row[column])
    except (TypeError, ValueError):
        return None


def _positive_odds(value) -> bool:
    try:
        return value is not None and float(value) > 1.0
    except (TypeError, ValueError):
        return False
