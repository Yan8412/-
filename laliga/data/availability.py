"""Lineups and sidelined (injury / suspension / doubtful) records for stored fixtures.

Documented SportMonks v3 behaviour this relies on:

* ``GET /fixtures/multi/{ids}?include=lineups;sidelined.sideline;metadata``.
  Lineups and sidelined are in every base plan (Starter included), with no
  add-on. Coverage still varies by league and fixture.
* ``lineups[].type_id`` 11 is the starting XI, 12 the bench. Official
  lineups appear about an hour before kickoff. Predicted lineups before that
  need the Expected Lineups add-on, which this project does not assume.
* ``metadata`` type 572 carries ``{"confirmed": bool}`` for the lineup.
* ``sidelined[]`` links a player to a fixture. Dates, games missed and the
  ``injury`` / ``suspension`` / ``doubtful`` category live on the nested
  ``sideline`` object. An empty array means no records, not a fit squad.

The daily ``fetch`` does not ask for these includes. ``fetch-availability``
fills ``processed/lineups.csv``, ``processed/sidelined.csv`` and
``processed/availability_fixtures.csv``. Finished fixtures are fetched once
and then served from the raw cache. Upcoming fixtures are requested again on
every run, because injuries and lineups change until kickoff.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from laliga.data.client import SportMonksClient, SportMonksError
from laliga.data.store import MatchStore

logger = logging.getLogger(__name__)

AVAILABILITY_INCLUDES = "lineups;sidelined.sideline;metadata"
STARTING_XI_TYPE = 11
BENCH_TYPE = 12
LINEUP_CONFIRMED_TYPE = 572
MULTI_CHUNK = 50

LINEUP_COLUMNS = ["fixture_id", "team_id", "player_id", "player_name", "type_id", "position_id", "formation_position"]
SIDELINED_COLUMNS = [
    "fixture_id",
    "team_id",
    "player_id",
    "sideline_id",
    "type_id",
    "category",
    "start_date",
    "end_date",
    "games_missed",
    "completed",
]
FIXTURE_COLUMNS = [
    "fixture_id",
    "fetched_at",
    "kickoff_utc",
    "status_at_fetch",
    "lineup_confirmed",
    "n_lineup",
    "n_starters",
    "n_sidelined",
]

AVAILABILITY_DENIED = (
    "订阅不能读取 lineups / sidelined（GET /fixtures/multi/{ids}?include=lineups;sidelined.sideline;metadata，HTTP 403）。"
    "SportMonks 文档写的是基础计划（含 Starter）都包含这两个 include。请在 MySportmonks 里确认订阅包含西甲 564，"
    "并用同一个 token 手动请求一次上面的地址。已停止其余请求，本地已有的阵容和伤停表保持不变。"
)


class AvailabilityStore:
    def __init__(self, root: Path) -> None:
        processed = Path(root) / "processed"
        self.lineups_path = processed / "lineups.csv"
        self.sidelined_path = processed / "sidelined.csv"
        self.fixtures_path = processed / "availability_fixtures.csv"

    def load(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return (
            _read(self.lineups_path, LINEUP_COLUMNS),
            _read(self.sidelined_path, SIDELINED_COLUMNS),
            _read(self.fixtures_path, FIXTURE_COLUMNS),
        )

    def upsert(self, lineups: list[dict], sidelined: list[dict], fixtures: list[dict]) -> None:
        if not fixtures:
            return
        touched = {int(item["fixture_id"]) for item in fixtures}
        current_lineups, current_sidelined, current_fixtures = self.load()
        _write(self.lineups_path, _replace(current_lineups, lineups, touched, LINEUP_COLUMNS))
        _write(self.sidelined_path, _replace(current_sidelined, sidelined, touched, SIDELINED_COLUMNS))
        _write(self.fixtures_path, _replace(current_fixtures, fixtures, touched, FIXTURE_COLUMNS))


def _read(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path)
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[columns]


def _replace(current: pd.DataFrame, rows: list[dict], touched: set[int], columns: list[str]) -> pd.DataFrame:
    kept = current[~pd.to_numeric(current["fixture_id"], errors="coerce").isin(touched)]
    incoming = pd.DataFrame(rows, columns=columns)
    frames = [frame for frame in (kept, incoming) if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)[columns]


def _write(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def parse_availability(payload: dict[str, Any], *, fetched_at: str, kickoff: Any = None, status: str = "") -> tuple[list[dict], list[dict], dict]:
    fixture_id = int(payload["id"])
    lineups = []
    for item in payload.get("lineups") or []:
        if not isinstance(item, dict) or item.get("player_id") is None or item.get("team_id") is None:
            continue
        lineups.append(
            {
                "fixture_id": fixture_id,
                "team_id": int(item["team_id"]),
                "player_id": int(item["player_id"]),
                "player_name": item.get("player_name") or "",
                "type_id": _as_int(item.get("type_id")),
                "position_id": _as_int(item.get("position_id")),
                "formation_position": _as_int(item.get("formation_position")),
            }
        )
    sidelined = []
    for item in payload.get("sidelined") or []:
        if not isinstance(item, dict) or item.get("player_id") is None:
            continue
        detail = item.get("sideline") if isinstance(item.get("sideline"), dict) else {}
        team_id = item.get("participant_id") if item.get("participant_id") is not None else detail.get("team_id")
        if team_id is None:
            continue
        sidelined.append(
            {
                "fixture_id": fixture_id,
                "team_id": int(team_id),
                "player_id": int(item["player_id"]),
                "sideline_id": _as_int(item.get("sideline_id") or detail.get("id")),
                "type_id": _as_int(item.get("type_id") or detail.get("type_id")),
                "category": str(detail.get("category") or "").lower(),
                "start_date": detail.get("start_date") or "",
                "end_date": detail.get("end_date") or "",
                "games_missed": _as_int(detail.get("games_missed")),
                "completed": detail.get("completed"),
            }
        )
    meta = {
        "fixture_id": fixture_id,
        "fetched_at": fetched_at,
        "kickoff_utc": "" if kickoff is None else pd.Timestamp(kickoff).isoformat(),
        "status_at_fetch": status,
        "lineup_confirmed": _lineup_confirmed(payload.get("metadata")),
        "n_lineup": len(lineups),
        "n_starters": sum(1 for row in lineups if row["type_id"] == STARTING_XI_TYPE),
        "n_sidelined": len(sidelined),
    }
    return lineups, sidelined, meta


def _lineup_confirmed(metadata: Any) -> bool | None:
    if not isinstance(metadata, list):
        return None
    for item in metadata:
        if not isinstance(item, dict):
            continue
        type_obj = item.get("type") if isinstance(item.get("type"), dict) else {}
        values = item.get("values")
        is_lineup = _as_int(item.get("type_id")) == LINEUP_CONFIRMED_TYPE or str(type_obj.get("developer_name") or "").upper() == "LINEUP_CONFIRMED"
        if is_lineup and isinstance(values, dict) and "confirmed" in values:
            return bool(values["confirmed"])
    return None


@dataclass
class AvailabilityFetchResult:
    finished_requested: int
    upcoming_requested: int
    fixtures_returned: int
    with_starting_xi: int
    with_sidelined: int
    denied: str | None


def fetch_availability(
    client: SportMonksClient,
    store: MatchStore,
    availability: AvailabilityStore,
    *,
    force: bool = False,
    days_ahead: int = 14,
    chunk_size: int = MULTI_CHUNK,
    now: pd.Timestamp | None = None,
) -> AvailabilityFetchResult:
    frame = store.load()
    if frame.empty:
        raise SportMonksError("本地没有比赛，无法回填阵容和伤停。请先运行 fetch。")
    now = pd.Timestamp(datetime.now(timezone.utc)) if now is None else pd.Timestamp(now)
    _, _, seen = availability.load()
    seen_ids = set(pd.to_numeric(seen["fixture_id"], errors="coerce").dropna().astype(int))
    finished = frame[frame["status"] == "finished"]
    finished_ids = [int(value) for value in finished["fixture_id"] if force or int(value) not in seen_ids]
    horizon = now + pd.Timedelta(days=days_ahead)
    upcoming = frame[(frame["status"] == "scheduled") & (frame["starting_at"] >= now) & (frame["starting_at"] <= horizon)]
    upcoming_ids = [int(value) for value in upcoming["fixture_id"]]
    meta = {int(row.fixture_id): (row.starting_at, row.status) for row in frame.itertuples(index=False)}

    returned = starters = with_sidelined = 0
    denied = None
    fetched_at = now.isoformat()
    for ids, live in ((finished_ids, force or None), (upcoming_ids, True)):
        for chunk in _chunks(ids, chunk_size):
            path = "fixtures/multi/" + ",".join(str(item) for item in chunk)
            lineups: list[dict] = []
            sidelined: list[dict] = []
            fixtures: list[dict] = []
            try:
                for payload in client.paginate(path, {"include": AVAILABILITY_INCLUDES}, refresh=live):
                    if not isinstance(payload, dict) or payload.get("id") is None:
                        continue
                    kickoff, status = meta.get(int(payload["id"]), (None, ""))
                    rows, absences, info = parse_availability(payload, fetched_at=fetched_at, kickoff=kickoff, status=status)
                    lineups.extend(rows)
                    sidelined.extend(absences)
                    fixtures.append(info)
            except SportMonksError as exc:
                if exc.status == 403 or "access" in str(exc).lower():
                    logger.warning("%s", AVAILABILITY_DENIED)
                    denied = AVAILABILITY_DENIED
                    break
                if exc.status == 404 or "no result" in str(exc).lower():
                    logger.info("这批比赛没有阵容或伤停：%s", exc)
                    continue
                raise
            availability.upsert(lineups, sidelined, fixtures)
            returned += len(fixtures)
            starters += sum(1 for item in fixtures if item["n_starters"] >= 20)
            with_sidelined += sum(1 for item in fixtures if item["n_sidelined"] > 0)
            logger.info("阵容/伤停：已处理 %s 场", returned)
        if denied:
            break
    return AvailabilityFetchResult(
        finished_requested=len(finished_ids),
        upcoming_requested=len(upcoming_ids),
        fixtures_returned=returned,
        with_starting_xi=starters,
        with_sidelined=with_sidelined,
        denied=denied,
    )


def format_availability_fetch(result: AvailabilityFetchResult) -> str:
    lines = [
        f"本次请求完场 {result.finished_requested} 场、未来 {result.upcoming_requested} 场（未开赛的每次都重新请求）。",
        f"接口返回 {result.fixtures_returned} 场；两队首发都齐的 {result.with_starting_xi} 场；有伤停记录的 {result.with_sidelined} 场。",
        "include=lineups;sidelined.sideline;metadata。正式首发大约开球前 1 小时公布；预测首发需要 Expected Lineups 附加包，这里不请求。",
        "伤停记录为空表示接口没有记录，不等于全员健康。",
    ]
    if result.denied:
        lines.append(result.denied)
    return "\n".join(lines)


def _chunks(values: list[int], size: int) -> Iterator[list[int]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _as_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
