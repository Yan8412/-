"""Pre-match features for the gradient-boosted challenger.

Every feature for a match is known before kickoff:

* Rolling form uses only matches from earlier UTC dates. Same-day results
  stay out, matching the Dixon–Coles walk-forward cut (the whole date is
  predicted from midnight, so an earlier kickoff that day is not available
  to the production model either).
* Expected goals are the historical xG of previous matches. The current
  match's xG is a post-match statistic and is never a feature of itself.
* Dixon–Coles attack, defence, expected goals, and 1X2 probabilities come
  from a model the caller has already fit on matches before this date.
* Decimal odds, when the stored row has them, are pre-match prices for
  this fixture. They are optional. Missing prices stay missing; nothing is
  imputed from future matches.

Trees do not need a scaler. Columns that have no history are NaN, which
XGBoost treats as missing. No encoder is fit on the evaluation period.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from laliga.model.dixon_coles import outcome_probabilities
from laliga.model.service import ScorelineModel

FORM_WINDOW = 5
REST_DAY_CAP = 30.0

FORM_FEATURES = [
    f"{side}_{stat}"
    for side in ("home", "away")
    for stat in (
        "ppg_5",
        "gf_5",
        "ga_5",
        "venue_ppg_5",
        "venue_gf_5",
        "venue_ga_5",
        "season_ppg",
        "season_gf",
        "season_ga",
        "matches",
        "rest_days",
        "ht_gf_5",
        "ht_ga_5",
    )
]
XG_FEATURES = [
    "home_xg_for_5",
    "home_xg_against_5",
    "away_xg_for_5",
    "away_xg_against_5",
]
DC_FEATURES = [
    "dc_home_attack",
    "dc_home_defence",
    "dc_away_attack",
    "dc_away_defence",
    "dc_exp_home",
    "dc_exp_away",
    "dc_p_home",
    "dc_p_draw",
    "dc_p_away",
    "dc_ht_exp_home",
    "dc_ht_exp_away",
    "dc_ht_p_home",
    "dc_ht_p_draw",
    "dc_ht_p_away",
]
ODDS_FEATURES = ["odds_implied_home", "odds_implied_draw", "odds_implied_away"]
DECIMAL_ODDS_COLUMNS = ("odds_home", "odds_draw", "odds_away")
STORED_IMPLIED_COLUMNS = ("implied_home", "implied_draw", "implied_away")
XG_COLUMNS = ("home_xg", "away_xg")

# Labels and raw post-match fields. None of these may be model inputs.
LEAKAGE_COLUMNS = frozenset(
    {
        "home_goals_ft",
        "away_goals_ft",
        "home_goals_ht",
        "away_goals_ht",
        "home_xg",
        "away_xg",
        "y_ft",
        "y_ht",
        "score",
    }
)


def model_feature_names(*, use_xg: bool, use_odds: bool) -> list[str]:
    names = list(FORM_FEATURES)
    if use_xg:
        names.extend(XG_FEATURES)
    names.extend(DC_FEATURES)
    if use_odds:
        names.extend(ODDS_FEATURES)
    overlap = LEAKAGE_COLUMNS.intersection(names)
    if overlap:
        raise RuntimeError(f"特征名与赛后字段重叠：{sorted(overlap)}")
    return names


def xg_column_status(matches: pd.DataFrame) -> str:
    """``absent``, ``empty``, or ``present`` for stored full-time xG."""

    if any(column not in matches.columns for column in XG_COLUMNS):
        return "absent"
    populated = matches[list(XG_COLUMNS)].apply(pd.to_numeric, errors="coerce").notna().any().any()
    return "present" if bool(populated) else "empty"


def odds_column_status(matches: pd.DataFrame) -> str:
    """``absent``, ``empty``, or ``present`` for a usable 1X2 price on any row."""

    decimal_here = all(column in matches.columns for column in DECIMAL_ODDS_COLUMNS)
    implied_here = all(column in matches.columns for column in STORED_IMPLIED_COLUMNS)
    if not decimal_here and not implied_here:
        return "absent"
    if _any_complete_odds(matches):
        return "present"
    return "empty"


def training_window_has_odds(matches: pd.DataFrame) -> bool:
    return _any_complete_odds(matches)


def compute_rolling_features(matches: pd.DataFrame, *, use_xg: bool) -> pd.DataFrame:
    """One row per fixture. History is every earlier UTC date, and nothing else."""

    if matches.empty:
        return pd.DataFrame(columns=["fixture_id", *model_feature_names(use_xg=use_xg, use_odds=False)])
    ordered = matches.sort_values(["starting_at", "fixture_id"]).copy()
    ordered["match_day"] = pd.to_datetime(ordered["starting_at"], utc=True).dt.floor("D")
    histories: dict[int, dict[str, list]] = {}
    rows: list[dict[str, Any]] = []
    for _, day_matches in ordered.groupby("match_day", sort=True):
        pending: list[pd.Series] = []
        for _, match in day_matches.iterrows():
            home_id = int(match["home_team_id"])
            away_id = int(match["away_team_id"])
            kickoff = pd.Timestamp(match["starting_at"])
            season_id = _season_id(match.get("season_id"))
            features: dict[str, Any] = {"fixture_id": int(match["fixture_id"])}
            features.update(_side_features("home", _bucket(histories, home_id), kickoff, season_id, "home", use_xg))
            features.update(_side_features("away", _bucket(histories, away_id), kickoff, season_id, "away", use_xg))
            rows.append(features)
            pending.append(match)
        for match in pending:
            _record(histories, match, use_xg=use_xg)
    return pd.DataFrame(rows)


def dixon_coles_features(model: ScorelineModel | None, home_id: int, away_id: int) -> dict[str, float]:
    """Strength and 1X2 from an already-fit model. ``None`` yields missing values."""

    if model is None:
        return {name: math.nan for name in DC_FEATURES}
    home_attack, home_defence = model.ft.strength(home_id)
    away_attack, away_defence = model.ft.strength(away_id)
    exp_home, exp_away = model.ft.expected_goals(home_id, away_id)
    p_home, p_draw, p_away = outcome_probabilities(model.ft.score_matrix(home_id, away_id))
    ht_home, ht_away = model.ht.expected_goals(home_id, away_id)
    ht_p_home, ht_p_draw, ht_p_away = outcome_probabilities(model.ht.score_matrix(home_id, away_id))
    return {
        "dc_home_attack": float(home_attack),
        "dc_home_defence": float(home_defence),
        "dc_away_attack": float(away_attack),
        "dc_away_defence": float(away_defence),
        "dc_exp_home": float(exp_home),
        "dc_exp_away": float(exp_away),
        "dc_p_home": float(p_home),
        "dc_p_draw": float(p_draw),
        "dc_p_away": float(p_away),
        "dc_ht_exp_home": float(ht_home),
        "dc_ht_exp_away": float(ht_away),
        "dc_ht_p_home": float(ht_p_home),
        "dc_ht_p_draw": float(ht_p_draw),
        "dc_ht_p_away": float(ht_p_away),
    }


def odds_features(row: pd.Series) -> dict[str, float]:
    """Normalised implied probabilities from this row's pre-match 1X2 prices."""

    implied = _decimal_implied(row)
    if implied is None:
        implied = _stored_implied(row)
    if implied is None:
        return {name: math.nan for name in ODDS_FEATURES}
    return {
        "odds_implied_home": implied[0],
        "odds_implied_draw": implied[1],
        "odds_implied_away": implied[2],
    }


def observed_feature_names(frame: pd.DataFrame, names: list[str]) -> list[str]:
    """Drop columns that are entirely missing inside this training window.

    The decision uses only ``frame``. A column that first appears on the
    test date is not a training feature.
    """

    kept = []
    for name in names:
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(values).any():
            kept.append(name)
    return kept


def _any_complete_odds(matches: pd.DataFrame) -> bool:
    if matches.empty:
        return False
    for _, row in matches.iterrows():
        if _decimal_implied(row) is not None or _stored_implied(row) is not None:
            return True
    return False


def _decimal_implied(row: pd.Series) -> tuple[float, float, float] | None:
    if any(column not in row.index for column in DECIMAL_ODDS_COLUMNS):
        return None
    prices = []
    for column in DECIMAL_ODDS_COLUMNS:
        value = row[column]
        if pd.isna(value):
            return None
        try:
            price = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(price) or price <= 1.0:
            return None
        prices.append(price)
    return _normalise([1.0 / price for price in prices])


def _stored_implied(row: pd.Series) -> tuple[float, float, float] | None:
    if any(column not in row.index for column in STORED_IMPLIED_COLUMNS):
        return None
    values = []
    for column in STORED_IMPLIED_COLUMNS:
        value = row[column]
        if pd.isna(value):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number) or number <= 0.0:
            return None
        values.append(number)
    return _normalise(values)


def _normalise(weights: list[float]) -> tuple[float, float, float] | None:
    total = float(sum(weights))
    if total <= 0.0 or not math.isfinite(total):
        return None
    home, draw, away = (weight / total for weight in weights)
    return float(home), float(draw), float(away)


def _bucket(histories: dict[int, dict[str, list]], team_id: int) -> dict[str, list]:
    bucket = histories.get(team_id)
    if bucket is None:
        bucket = {key: [] for key in ("kickoff", "gf", "ga", "pts", "venue", "season", "xg_for", "xg_against", "ht_gf", "ht_ga")}
        histories[team_id] = bucket
    return bucket


def _record(histories: dict[int, dict[str, list]], match: pd.Series, *, use_xg: bool) -> None:
    home_id = int(match["home_team_id"])
    away_id = int(match["away_team_id"])
    kickoff = pd.Timestamp(match["starting_at"])
    season_id = _season_id(match.get("season_id"))
    home_goals = float(match["home_goals_ft"])
    away_goals = float(match["away_goals_ft"])
    _push(_bucket(histories, home_id), kickoff, home_goals, away_goals, "home", season_id)
    _push(_bucket(histories, away_id), kickoff, away_goals, home_goals, "away", season_id)
    home_xg, away_xg = _pair(match, "home_xg", "away_xg") if use_xg else (None, None)
    if home_xg is not None and away_xg is not None:
        _bucket(histories, home_id)["xg_for"].append(home_xg)
        _bucket(histories, home_id)["xg_against"].append(away_xg)
        _bucket(histories, away_id)["xg_for"].append(away_xg)
        _bucket(histories, away_id)["xg_against"].append(home_xg)
    home_ht, away_ht = _pair(match, "home_goals_ht", "away_goals_ht")
    if home_ht is not None and away_ht is not None:
        _bucket(histories, home_id)["ht_gf"].append(home_ht)
        _bucket(histories, home_id)["ht_ga"].append(away_ht)
        _bucket(histories, away_id)["ht_gf"].append(away_ht)
        _bucket(histories, away_id)["ht_ga"].append(home_ht)


def _push(bucket: dict[str, list], kickoff: pd.Timestamp, gf: float, ga: float, venue: str, season_id: int | None) -> None:
    bucket["kickoff"].append(kickoff)
    bucket["gf"].append(gf)
    bucket["ga"].append(ga)
    if gf > ga:
        points = 3.0
    elif gf == ga:
        points = 1.0
    else:
        points = 0.0
    bucket["pts"].append(points)
    bucket["venue"].append(venue)
    bucket["season"].append(season_id)


def _side_features(
    prefix: str,
    bucket: dict[str, list],
    kickoff: pd.Timestamp,
    season_id: int | None,
    venue: str,
    use_xg: bool,
) -> dict[str, float]:
    features = {
        f"{prefix}_ppg_5": _tail(bucket["pts"]),
        f"{prefix}_gf_5": _tail(bucket["gf"]),
        f"{prefix}_ga_5": _tail(bucket["ga"]),
        f"{prefix}_venue_ppg_5": _masked_tail(bucket["pts"], bucket["venue"], venue),
        f"{prefix}_venue_gf_5": _masked_tail(bucket["gf"], bucket["venue"], venue),
        f"{prefix}_venue_ga_5": _masked_tail(bucket["ga"], bucket["venue"], venue),
        f"{prefix}_season_ppg": _season_mean(bucket["pts"], bucket["season"], season_id),
        f"{prefix}_season_gf": _season_mean(bucket["gf"], bucket["season"], season_id),
        f"{prefix}_season_ga": _season_mean(bucket["ga"], bucket["season"], season_id),
        f"{prefix}_matches": float(len(bucket["gf"])),
        f"{prefix}_rest_days": _rest_days(bucket["kickoff"], kickoff),
        f"{prefix}_ht_gf_5": _tail(bucket["ht_gf"]),
        f"{prefix}_ht_ga_5": _tail(bucket["ht_ga"]),
    }
    if use_xg:
        features[f"{prefix}_xg_for_5"] = _tail(bucket["xg_for"])
        features[f"{prefix}_xg_against_5"] = _tail(bucket["xg_against"])
    return features


def _tail(values: list[float], window: int = FORM_WINDOW) -> float:
    if not values:
        return math.nan
    chunk = values[-window:]
    return float(sum(chunk) / len(chunk))


def _masked_tail(values: list[float], venues: list[str], venue: str, window: int = FORM_WINDOW) -> float:
    chosen = [value for value, side in zip(values, venues, strict=True) if side == venue]
    return _tail(chosen, window)


def _season_mean(values: list[float], seasons: list[int | None], season_id: int | None) -> float:
    if season_id is None:
        return math.nan
    chosen = [value for value, season in zip(values, seasons, strict=True) if season == season_id]
    if not chosen:
        return math.nan
    return float(sum(chosen) / len(chosen))


def _rest_days(kickoffs: list[pd.Timestamp], kickoff: pd.Timestamp) -> float:
    if not kickoffs:
        return math.nan
    delta = (pd.Timestamp(kickoff) - pd.Timestamp(kickoffs[-1])).total_seconds() / 86400.0
    if not math.isfinite(delta) or delta < 0:
        return math.nan
    return float(min(delta, REST_DAY_CAP))


def _season_id(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _pair(match: pd.Series, home_col: str, away_col: str) -> tuple[float | None, float | None]:
    if home_col not in match.index or away_col not in match.index:
        return None, None
    home = match[home_col]
    away = match[away_col]
    if pd.isna(home) or pd.isna(away):
        return None, None
    try:
        return float(home), float(away)
    except (TypeError, ValueError):
        return None, None
