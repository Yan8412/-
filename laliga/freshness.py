"""Staleness guard for the daily run.

STALE (exit code 2) when the local match table cannot be trusted for a new
prediction:

* a fixture is still unplayed (scheduled, or stuck in an in-play state) more
  than 48 hours after its kickoff, or
* the latest result is more than 10 days old and no upcoming fixture is
  listed.

A gap between results is fine when a later fixture is already listed, which
is what an international break looks like. A long gap with no fixture
listed is fine only when the latest season looks complete (off-season):
nothing in it is left unplayed and it has close to a full 380-match
schedule of results, or the saved season list marks it finished.

``LALIGA_FRESHNESS_NOW`` (ISO 8601) replaces the clock, for tests and for
replaying a past day.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from laliga.config import SKIP_STATE_NAMES

NOW_ENV = "LALIGA_FRESHNESS_NOW"
UNPLAYED_GRACE_HOURS = 48
RESULT_MAX_AGE_DAYS = 10
FULL_SEASON_RESULTS = 370
EXIT_OK = 0
EXIT_STALE = 2

IN_PLAY_STATE_NAMES = frozenset(
    name
    for name in SKIP_STATE_NAMES
    if name.startswith("INPLAY") or name in {"HT", "BREAK", "EXTRA_TIME_BREAK", "PEN_BREAK", "AWAITING_UPDATES"}
)


@dataclass
class FreshnessResult:
    stale: bool
    reasons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    now: str = ""
    latest_result: str | None = None
    next_fixture: str | None = None
    overdue: list[dict] = field(default_factory=list)

    @property
    def status(self) -> str:
        return "STALE" if self.stale else "OK"

    @property
    def exit_code(self) -> int:
        return EXIT_STALE if self.stale else EXIT_OK

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "exit_code": self.exit_code,
            "now": self.now,
            "latest_result": self.latest_result,
            "next_fixture": self.next_fixture,
            "reasons": self.reasons,
            "notes": self.notes,
            "overdue": self.overdue,
        }

    def format(self) -> str:
        lines = [f"{self.status}: " + ("；".join(self.reasons) if self.stale else "；".join(self.notes or ["数据是新的"]))]
        lines.append(f"  当前时间（UTC）：{self.now}")
        lines.append(f"  最近一场完场：{self.latest_result or '没有'}")
        lines.append(f"  下一场未开赛：{self.next_fixture or '没有'}")
        for item in self.overdue[:10]:
            lines.append(f"  超时未完场：{item['kickoff_utc']} {item['home_team']} vs {item['away_team']}（{item['state']}，id {item['fixture_id']}）")
        if len(self.overdue) > 10:
            lines.append(f"  其余 {len(self.overdue) - 10} 场未列出。")
        return "\n".join(lines)


def resolve_now(explicit: str | None = None) -> pd.Timestamp:
    text = explicit or os.environ.get(NOW_ENV, "").strip()
    if not text:
        return pd.Timestamp(datetime.now(timezone.utc))
    stamp = pd.to_datetime(text, utc=True, errors="coerce")
    if pd.isna(stamp):
        raise ValueError(f"{NOW_ENV} 或 --now 不是 ISO 8601 时间：{text!r}")
    return pd.Timestamp(stamp)


def check_freshness(
    matches: pd.DataFrame,
    now: pd.Timestamp,
    *,
    seasons_doc: dict | None = None,
    unplayed_grace_hours: float = UNPLAYED_GRACE_HOURS,
    result_max_age_days: float = RESULT_MAX_AGE_DAYS,
) -> FreshnessResult:
    now = pd.Timestamp(now)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    result = FreshnessResult(stale=False, now=now.isoformat())
    if matches is None or matches.empty:
        result.stale = True
        result.reasons.append("本地比赛表是空的，先运行 fetch")
        return result

    frame = matches.dropna(subset=["starting_at"])
    state = frame["state_name"].fillna("").astype(str).str.upper()
    unplayed = (frame["status"] == "scheduled") | state.isin(IN_PLAY_STATE_NAMES)
    deadline = now - pd.Timedelta(hours=unplayed_grace_hours)
    overdue = frame[unplayed & (frame["starting_at"] < deadline)].sort_values("starting_at")
    for _, row in overdue.iterrows():
        result.overdue.append(
            {
                "fixture_id": int(row["fixture_id"]),
                "kickoff_utc": pd.Timestamp(row["starting_at"]).isoformat(),
                "home_team": str(row["home_team_name"]),
                "away_team": str(row["away_team_name"]),
                "state": str(row["state_name"] or row["status"]),
            }
        )
    if result.overdue:
        result.stale = True
        result.reasons.append(f"{len(result.overdue)} 场比赛开球已超过 {unplayed_grace_hours:g} 小时仍未完场")

    finished = frame[frame["status"] == "finished"]
    upcoming = frame[(frame["status"] == "scheduled") & (frame["starting_at"] >= now)].sort_values("starting_at")
    if not upcoming.empty:
        first = upcoming.iloc[0]
        result.next_fixture = f"{pd.Timestamp(first['starting_at']).isoformat()} {first['home_team_name']} vs {first['away_team_name']}"
    if finished.empty:
        if upcoming.empty:
            result.stale = True
            result.reasons.append("本地没有完场比赛，也没有未开赛比赛")
        else:
            result.notes.append("还没有完场比赛，但已有未开赛比赛（赛季开始前）")
        return result

    latest = pd.Timestamp(finished["starting_at"].max())
    result.latest_result = latest.isoformat()
    age_days = (now - latest).total_seconds() / 86400.0
    if age_days <= result_max_age_days:
        if not result.stale:
            result.notes.append(f"最近一场完场在 {age_days:.1f} 天前")
        return result
    if not upcoming.empty:
        if not result.stale:
            result.notes.append(f"最近一场完场在 {age_days:.1f} 天前，但下一场已经排出（国家队比赛日或休赛间隙）")
        return result
    if _latest_season_complete(frame, finished, latest, seasons_doc):
        if not result.stale:
            result.notes.append(f"最近一场完场在 {age_days:.1f} 天前；上个赛季已踢完，新赛程还没公布（休赛期）")
        return result
    result.stale = True
    result.reasons.append(f"最近一场完场在 {age_days:.1f} 天前（超过 {result_max_age_days:g} 天），而且没有任何未开赛比赛")
    return result


def _latest_season_complete(frame: pd.DataFrame, finished: pd.DataFrame, latest: pd.Timestamp, seasons_doc: dict | None) -> bool:
    season_ids = finished.loc[finished["starting_at"] == latest, "season_id"].dropna()
    if season_ids.empty:
        return False
    season_id = int(season_ids.iloc[0])
    season = frame[frame["season_id"] == season_id]
    if (season["status"] == "scheduled").any():
        return False
    if int((season["status"] == "finished").sum()) >= FULL_SEASON_RESULTS:
        return True
    for item in (seasons_doc or {}).get("seasons") or []:
        if str(item.get("id")) == str(season_id) and item.get("finished") is True:
            return True
    return False
