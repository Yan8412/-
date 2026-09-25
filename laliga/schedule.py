"""Choose the next unplayed matchday from a local match table."""

from __future__ import annotations

import pandas as pd


def select_next_matchday(
    matches: pd.DataFrame, now: pd.Timestamp | None = None
) -> tuple[pd.DataFrame, str]:
    """Return the next round of scheduled fixtures and a note for the user.

    Preference is the earliest future kickoff's ``season_id`` + ``round_id``.
    When every scheduled row is already in the past, the earliest scheduled
    round is used and the note explains that.
    """

    if matches.empty or "status" not in matches.columns:
        return matches.iloc[0:0].copy(), ""
    scheduled = matches[matches["status"] == "scheduled"].dropna(subset=["starting_at"])
    if scheduled.empty:
        return scheduled, ""
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    future = scheduled[scheduled["starting_at"] >= now]
    note = ""
    pool = future
    if pool.empty:
        pool = scheduled
        note = "没有晚于当前时间的未开赛比赛，改为使用数据集里最早的未开赛轮次。"
    pool = pool.sort_values(["starting_at", "fixture_id"])
    first = pool.iloc[0]
    if pd.notna(first["round_id"]) and pd.notna(first["season_id"]):
        chosen = pool[(pool["season_id"] == first["season_id"]) & (pool["round_id"] == first["round_id"])]
    else:
        day = pd.Timestamp(first["starting_at"]).floor("D")
        chosen = pool[pool["starting_at"].dt.floor("D") == day]
    return chosen.sort_values(["starting_at", "fixture_id"]).reset_index(drop=True), note


def scheduled_between(matches: pd.DataFrame, start: pd.Timestamp, end_exclusive: pd.Timestamp) -> pd.DataFrame:
    if matches.empty:
        return matches
    mask = (
        (matches["status"] == "scheduled")
        & (matches["starting_at"] >= start)
        & (matches["starting_at"] < end_exclusive)
    )
    return matches.loc[mask].sort_values(["starting_at", "fixture_id"]).reset_index(drop=True)
