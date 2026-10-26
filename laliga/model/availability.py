"""Pre-match availability as a small adjustment on Dixon–Coles expected goals.

Feature (per team, per fixture, all known before kickoff):

* A player's *starter share* is the fraction of the team's last 10 league
  matches, on earlier UTC dates, in which he was in the starting XI.
* ``missing`` is the sum of starter shares of players listed as sidelined
  for this fixture: injury and suspension count fully, doubtful counts half.
  A record that starts after the match date or ended before it is ignored.
  A long-term absentee has a share near 0, so he does not move the number:
  his absence is already in the team's recent results.
* ``lineup_gap`` is how much starter share the usual XI (the 11 highest
  shares) has that the named XI does not. It exists only when a starting XI
  is known, which for an upcoming match means about an hour before kickoff.

Adjustment, with two coefficients per feature:

    λ_home' = λ_home · exp(−a · missing_home + d · missing_away)
    λ_away' = λ_away · exp(−a · missing_away + d · missing_home)

``a`` is the attack lost per missing regular starter, ``d`` the extra goals
conceded per missing regular starter on the other side. ρ, the half-time
model and the score grid are unchanged; the same factors scale half-time
goals. Coefficients are fit by Poisson likelihood with a Gaussian prior at 0.

The walk-forward fits Dixon–Coles on each evaluation day exactly like
``walk_forward``. The coefficients for a day are fit only on earlier
evaluation days, using those days' out-of-sample Dixon–Coles expected goals.
Nothing from the day being scored, or later, is used.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from laliga.config import ModelConfig
from laliga.data.availability import STARTING_XI_TYPE
from laliga.model.baseline import fit_baseline
from laliga.model.dixon_coles import FitError, outcome_probabilities, score_matrix, top_scorelines
from laliga.model.metrics import (
    MetricBlock,
    iter_evaluation_folds,
    outcome_index,
    paired_logloss_difference,
    summarize_rows,
)
from laliga.model.service import GoalMultipliers, ScorelineModel, fit_models

SCHEMA_VERSION = 1
SHARE_WINDOW = 10
MIN_SHARE_HISTORY = 3
CATEGORY_WEIGHTS = {"injury": 1.0, "suspension": 1.0, "doubtful": 0.5}
MAX_MISSING = 6.0
PRIOR_PRECISION = 25.0
COEF_BOUND = 0.5
MIN_FIT_ROWS = 150
STALE_AFTER_HOURS = 36.0
MIN_RECOMMEND_N = 300

FEATURES = ("missing", "lineup_gap")
FEATURE_LABELS = {
    "missing": "Dixon–Coles + 伤停（缺阵主力）",
    "lineup_gap": "Dixon–Coles + 首发缺口（需要开球前约 1 小时的正式首发）",
}


def _day(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    return stamp.floor("D")


def _date(value) -> pd.Timestamp | None:
    if value is None or (isinstance(value, float) and math.isnan(value)) or value == "":
        return None
    stamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(stamp):
        return None
    return pd.Timestamp(stamp).floor("D")


@dataclass
class TeamAvailability:
    missing: float | None
    lineup_gap: float | None
    n_history: int
    absent: list[dict]


class AvailabilityFeatures:
    """Per-fixture features from stored lineups and sidelined records."""

    def __init__(self, matches: pd.DataFrame, lineups: pd.DataFrame, sidelined: pd.DataFrame, fetched: pd.DataFrame) -> None:
        self.matches = matches.dropna(subset=["starting_at"]).sort_values(["starting_at", "fixture_id"])
        starters = lineups[pd.to_numeric(lineups["type_id"], errors="coerce") == STARTING_XI_TYPE]
        self.starters: dict[tuple[int, int], set[int]] = {}
        for (fixture_id, team_id), group in starters.groupby(["fixture_id", "team_id"]):
            self.starters[(int(fixture_id), int(team_id))] = {int(value) for value in group["player_id"]}
        self.sidelined: dict[tuple[int, int], list[dict]] = {}
        for record in sidelined.to_dict("records"):
            key = (int(record["fixture_id"]), int(record["team_id"]))
            self.sidelined.setdefault(key, []).append(record)
        self.fetched = {int(value) for value in pd.to_numeric(fetched["fixture_id"], errors="coerce").dropna()}
        self.fetch_meta = {int(row["fixture_id"]): row for row in fetched.to_dict("records")}
        self._team_matches: dict[int, pd.DataFrame] = {}
        for team_id in pd.concat([self.matches["home_team_id"], self.matches["away_team_id"]]).dropna().unique():
            team = int(team_id)
            mask = (self.matches["home_team_id"] == team) | (self.matches["away_team_id"] == team)
            played = self.matches[mask & (self.matches["status"] == "finished")]
            self._team_matches[team] = played[["fixture_id", "starting_at"]]

    def starter_shares(self, team_id: int, before_day: pd.Timestamp) -> tuple[dict[int, float], int]:
        played = self._team_matches.get(int(team_id))
        if played is None or played.empty:
            return {}, 0
        earlier = played[played["starting_at"] < before_day]
        with_lineup = [int(fid) for fid in earlier["fixture_id"] if (int(fid), int(team_id)) in self.starters]
        window = with_lineup[-SHARE_WINDOW:]
        if not window:
            return {}, 0
        counts: dict[int, int] = {}
        for fixture_id in window:
            for player in self.starters[(fixture_id, int(team_id))]:
                counts[player] = counts.get(player, 0) + 1
        return {player: count / len(window) for player, count in counts.items()}, len(window)

    def team(self, fixture_id: int, team_id: int, kickoff) -> TeamAvailability:
        day = _day(kickoff)
        shares, n_history = self.starter_shares(team_id, day)
        if n_history < MIN_SHARE_HISTORY:
            return TeamAvailability(None, None, n_history, [])
        missing = None
        absent: list[dict] = []
        if fixture_id in self.fetched:
            weights: dict[int, float] = {}
            for record in self.sidelined.get((fixture_id, int(team_id)), []):
                weight = CATEGORY_WEIGHTS.get(str(record.get("category") or "").lower())
                if weight is None:
                    continue
                start = _date(record.get("start_date"))
                end = _date(record.get("end_date"))
                if start is not None and start > day:
                    continue
                if end is not None and end < day:
                    continue
                player = int(record["player_id"])
                weights[player] = max(weights.get(player, 0.0), weight)
            missing = 0.0
            for player, weight in weights.items():
                share = shares.get(player, 0.0)
                missing += weight * share
                if share > 0:
                    absent.append({"player_id": player, "starter_share": round(share, 3), "weight": weight})
            missing = min(missing, MAX_MISSING)
        lineup_gap = None
        named = self.starters.get((fixture_id, int(team_id)))
        if named is not None and len(named) >= 11:
            usual = sum(sorted(shares.values(), reverse=True)[:11])
            lineup_gap = max(0.0, usual - sum(shares.get(player, 0.0) for player in named))
            lineup_gap = min(lineup_gap, MAX_MISSING)
        return TeamAvailability(missing, lineup_gap, n_history, absent)

    def fixture(self, row) -> dict:
        fixture_id = int(row["fixture_id"])
        home = self.team(fixture_id, int(row["home_team_id"]), row["starting_at"])
        away = self.team(fixture_id, int(row["away_team_id"]), row["starting_at"])
        return {
            "home_missing": home.missing,
            "away_missing": away.missing,
            "home_lineup_gap": home.lineup_gap,
            "away_lineup_gap": away.lineup_gap,
            "home_history": home.n_history,
            "away_history": away.n_history,
            "home_absent": home.absent,
            "away_absent": away.absent,
        }


@dataclass
class Coefficients:
    attack: float = 0.0
    defence: float = 0.0
    n: int = 0

    @property
    def active(self) -> bool:
        return self.n > 0

    def multipliers(self, home: float | None, away: float | None) -> GoalMultipliers:
        if not self.active or home is None or away is None:
            return GoalMultipliers()
        return GoalMultipliers(
            home=math.exp(-self.attack * home + self.defence * away),
            away=math.exp(-self.attack * away + self.defence * home),
        )

    def to_dict(self) -> dict:
        return {"attack_per_missing_starter": self.attack, "defence_per_missing_starter": self.defence, "n": self.n}

    @classmethod
    def from_dict(cls, payload: dict | None) -> "Coefficients":
        if not payload:
            return cls()
        return cls(
            attack=float(payload.get("attack_per_missing_starter") or 0.0),
            defence=float(payload.get("defence_per_missing_starter") or 0.0),
            n=int(payload.get("n") or 0),
        )


def fit_coefficients(rows: list[dict], feature: str, *, min_rows: int = MIN_FIT_ROWS) -> Coefficients:
    """Poisson MLE for (a, d) on rows with ``lam_home``, ``lam_away``, goals, and the feature."""

    usable = [row for row in rows if row.get(f"home_{feature}") is not None and row.get(f"away_{feature}") is not None]
    if len(usable) < min_rows:
        return Coefficients()
    lam_h = np.array([row["lam_home"] for row in usable], dtype=float)
    lam_a = np.array([row["lam_away"] for row in usable], dtype=float)
    y_h = np.array([row["score"][0] for row in usable], dtype=float)
    y_a = np.array([row["score"][1] for row in usable], dtype=float)
    m_h = np.array([row[f"home_{feature}"] for row in usable], dtype=float)
    m_a = np.array([row[f"away_{feature}"] for row in usable], dtype=float)
    if not np.any(m_h) and not np.any(m_a):
        return Coefficients(n=len(usable))

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        attack, defence = theta
        adj_h = lam_h * np.exp(-attack * m_h + defence * m_a)
        adj_a = lam_a * np.exp(-attack * m_a + defence * m_h)
        nll = -float(np.sum(y_h * np.log(adj_h) - adj_h + y_a * np.log(adj_a) - adj_a))
        nll += 0.5 * PRIOR_PRECISION * float(attack**2 + defence**2)
        r_h = y_h - adj_h
        r_a = y_a - adj_a
        grad_attack = -float(np.sum(r_h * -m_h + r_a * -m_a)) + PRIOR_PRECISION * attack
        grad_defence = -float(np.sum(r_h * m_a + r_a * m_h)) + PRIOR_PRECISION * defence
        return nll, np.array([grad_attack, grad_defence])

    result = minimize(objective, np.zeros(2), jac=True, method="L-BFGS-B", bounds=[(-COEF_BOUND, COEF_BOUND)] * 2)
    attack, defence = (float(value) for value in result.x)
    return Coefficients(attack=attack, defence=defence, n=len(usable))


def _predict_with(model: ScorelineModel, row, multipliers: GoalMultipliers) -> tuple[tuple, tuple, list]:
    home_id = int(row["home_team_id"])
    away_id = int(row["away_team_id"])
    (ft_h, ft_a), (ht_h, ht_a) = model.expected_goals(home_id, away_id, multipliers)
    ft = score_matrix(ft_h, ft_a, model.ft.rho, model.ft.max_goals)
    ht = score_matrix(ht_h, ht_a, model.ht.rho, model.ht.max_goals)
    top3 = [(home, away) for home, away, _ in top_scorelines(ft, 3)]
    return outcome_probabilities(ft), outcome_probabilities(ht), top3


@dataclass
class AvailabilityReport:
    n_folds: int
    first_test_day: str | None
    last_test_day: str | None
    min_train_matches: int
    models: dict[str, MetricBlock]
    active: dict[str, dict[str, MetricBlock]]
    paired: list[dict]
    coverage: dict
    coefficients: dict[str, dict]
    coefficient_path: dict[str, dict]
    recommend_default: bool
    recommendation: str
    notes: list[str]
    rows: list[dict] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "computed_from_stored_matches": True,
            "production_model": "dixon_coles",
            "n_folds": self.n_folds,
            "first_test_day": self.first_test_day,
            "last_test_day": self.last_test_day,
            "min_train_matches": self.min_train_matches,
            "models": {key: block.to_dict() for key, block in self.models.items()},
            "active_subset": {
                feature: {key: block.to_dict() for key, block in blocks.items()} for feature, blocks in self.active.items()
            },
            "paired": self.paired,
            "coverage": self.coverage,
            "coefficients": self.coefficients,
            "coefficient_path": self.coefficient_path,
            "recommend_default": self.recommend_default,
            "recommendation": self.recommendation,
            "notes": self.notes,
        }


def compare_availability(
    matches: pd.DataFrame,
    features: AvailabilityFeatures,
    config: ModelConfig,
    *,
    min_fit_rows: int = MIN_FIT_ROWS,
) -> AvailabilityReport:
    finished = matches[matches["status"] == "finished"].dropna(subset=["home_goals_ft", "away_goals_ft"])
    if finished.empty:
        raise FitError("没有完场比赛，无法评估伤停特征。")
    rows: list[dict] = []
    history: list[dict] = []
    previous: ScorelineModel | None = None
    folds = 0
    first_day = last_day = None
    path: dict[str, list[dict]] = {feature: [] for feature in FEATURES}
    for day, train, test in iter_evaluation_folds(finished, config.min_train_matches):
        try:
            model = fit_models(
                train,
                pd.Timestamp(day),
                config,
                initial_ft=None if previous is None else previous.ft,
                initial_ht=None if previous is None else previous.ht,
                initial_sh=None if previous is None else previous.sh,
            )
        except FitError:
            continue
        previous = model
        baseline = fit_baseline(train)
        coefficients = {feature: fit_coefficients(history, feature, min_rows=min_fit_rows) for feature in FEATURES}
        for feature, coef in coefficients.items():
            if coef.active:
                path[feature].append({"day": pd.Timestamp(day).date().isoformat(), **coef.to_dict()})
        folds += 1
        first_day = day if first_day is None else first_day
        last_day = day
        day_rows = []
        for _, match in test.iterrows():
            home_ft = int(match["home_goals_ft"])
            away_ft = int(match["away_goals_ft"])
            y_ht = None
            if pd.notna(match["home_goals_ht"]) and pd.notna(match["away_goals_ht"]):
                y_ht = outcome_index(int(match["home_goals_ht"]), int(match["away_goals_ht"]))
            lam_home, lam_away = model.ft.expected_goals(int(match["home_team_id"]), int(match["away_team_id"]))
            info = features.fixture(match)
            dc_ft, dc_ht, dc_top3 = _predict_with(model, match, GoalMultipliers())
            row = {
                "fixture_id": int(match["fixture_id"]),
                "season_name": str(match.get("season_name") or ""),
                "y_ft": outcome_index(home_ft, away_ft),
                "y_ht": y_ht,
                "score": (home_ft, away_ft),
                "lam_home": lam_home,
                "lam_away": lam_away,
                **{key: info[key] for key in ("home_missing", "away_missing", "home_lineup_gap", "away_lineup_gap")},
                "model_ft_probs": dc_ft,
                "model_ht_probs": dc_ht,
                "model_top3": dc_top3,
                "baseline_ft_probs": baseline.ft_probs,
                "baseline_ht_probs": baseline.ht_probs,
                "baseline_top3": [(home, away) for home, away, _ in baseline.top_scores],
            }
            for feature, coef in coefficients.items():
                home_value = info[f"home_{feature}"]
                away_value = info[f"away_{feature}"]
                active = coef.active and home_value is not None and away_value is not None
                if active:
                    ft, ht, top3 = _predict_with(model, match, coef.multipliers(home_value, away_value))
                else:
                    ft, ht, top3 = dc_ft, dc_ht, dc_top3
                row[f"{feature}_ft_probs"] = ft
                row[f"{feature}_ht_probs"] = ht
                row[f"{feature}_top3"] = top3
                row[f"{feature}_active"] = active
            day_rows.append(row)
        rows.extend(day_rows)
        history.extend(day_rows)
    if not rows:
        raise FitError(
            f"没有评测任何比赛。完场比赛有 {len(finished)} 场，最少训练样本是 {config.min_train_matches}。可以降低 --min-train。"
        )
    return _report(rows, folds, first_day, last_day, config.min_train_matches, path, min_fit_rows)


def _report(rows, folds, first_day, last_day, min_train, path, min_fit_rows) -> AvailabilityReport:
    y = np.array([row["y_ft"] for row in rows], dtype=int)
    models = {
        "baseline": summarize_rows(rows, "baseline"),
        "dixon_coles": summarize_rows(rows, "model"),
    }
    active: dict[str, dict[str, MetricBlock]] = {}
    paired: list[dict] = []
    coverage = {
        "n_evaluated": len(rows),
        "with_missing_feature": sum(1 for row in rows if row["home_missing"] is not None and row["away_missing"] is not None),
        "with_nonzero_missing": sum(1 for row in rows if (row["home_missing"] or 0) > 0 or (row["away_missing"] or 0) > 0),
        "with_lineup_feature": sum(1 for row in rows if row["home_lineup_gap"] is not None and row["away_lineup_gap"] is not None),
    }
    recommend = False
    reasons = []
    for feature in FEATURES:
        models[feature] = summarize_rows(rows, feature)
        subset = [row for row in rows if row[f"{feature}_active"]]
        coverage[f"{feature}_active"] = len(subset)
        if subset:
            active[feature] = {
                "dixon_coles": summarize_rows(subset, "model"),
                feature: summarize_rows(subset, feature),
                "baseline": summarize_rows(subset, "baseline"),
            }
            sub_y = np.array([row["y_ft"] for row in subset], dtype=int)
            comparison = paired_logloss_difference(
                sub_y,
                np.array([row[f"{feature}_ft_probs"] for row in subset], dtype=float),
                np.array([row["model_ft_probs"] for row in subset], dtype=float),
            )
            comparison["label"] = f"{FEATURE_LABELS[feature]} − Dixon–Coles（调整生效的比赛）"
            comparison["feature"] = feature
            paired.append(comparison)
            if feature == "missing":
                clear = (
                    comparison["n"] >= MIN_RECOMMEND_N
                    and comparison["ci_high"] < 0
                    and comparison["mean_rps_difference"] < 0
                    and comparison["rps_ci_high"] < 0
                )
                recommend = bool(clear)
                reasons.append(
                    f"伤停调整对 Dixon–Coles：对数损失之差 {comparison['mean_logloss_difference']:+.4f}"
                    f"（95% 区间 {comparison['ci_low']:+.4f} 到 {comparison['ci_high']:+.4f}），"
                    f"RPS 之差 {comparison['mean_rps_difference']:+.4f}"
                    f"（{comparison['rps_ci_low']:+.4f} 到 {comparison['rps_ci_high']:+.4f}），n={comparison['n']}。"
                )
    paired.append(
        {
            **paired_logloss_difference(
                y,
                np.array([row["model_ft_probs"] for row in rows], dtype=float),
                np.array([row["baseline_ft_probs"] for row in rows], dtype=float),
            ),
            "label": "Dixon–Coles − 历史频率基准",
            "feature": None,
        }
    )
    final = {feature: fit_coefficients(rows, feature, min_rows=min_fit_rows).to_dict() for feature in FEATURES}
    if recommend:
        recommendation = "伤停调整在样本外明显更好（对数损失和 RPS 的 95% 区间都在 0 以下，n≥300）。可以考虑在每日脚本里打开 --availability。"
    else:
        recommendation = (
            "伤停调整没有达到默认开启的标准（需要 n≥300，且对数损失和 RPS 之差的 95% 区间都在 0 以下）。"
            "生产继续用 Dixon–Coles，伤停只作为 --availability 开关。"
        )
    notes = reasons + [
        "首发缺口用的是比赛的正式首发。历史比赛里它一定存在，但每日脚本运行时离开球通常远超过 1 小时，正式首发还没公布，"
        "所以这一行只说明「开球前约 1 小时再跑一次」能带来多少，不是每日脚本能拿到的结果。",
        "历史比赛的伤停记录是事后从 SportMonks 读的。接口没有说明记录是在开球前还是开球后补录的，这一点会让历史评估偏乐观。",
        f"系数每天只用更早评测日的样本外比赛拟合；累计不到 {min_fit_rows} 场之前不调整，这些比赛的概率等于 Dixon–Coles。",
    ]
    return AvailabilityReport(
        n_folds=folds,
        first_test_day=None if first_day is None else pd.Timestamp(first_day).date().isoformat(),
        last_test_day=None if last_day is None else pd.Timestamp(last_day).date().isoformat(),
        min_train_matches=min_train,
        models=models,
        active=active,
        paired=paired,
        coverage=coverage,
        coefficients=final,
        coefficient_path={feature: _path_summary(items) for feature, items in path.items()},
        recommend_default=recommend,
        recommendation=recommendation,
        notes=notes,
        rows=rows,
    )


def _path_summary(items: list[dict]) -> dict:
    if not items:
        return {"days": 0}
    return {"days": len(items), "first": items[0], "last": items[-1]}


def coefficients_path(root: Path) -> Path:
    return Path(root) / "models" / "availability.json"


def save_coefficients(root: Path, report: AvailabilityReport) -> Path:
    path = coefficients_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "fitted_at": datetime.now(timezone.utc).isoformat(),
        "coefficients": report.coefficients,
        "recommend_default": report.recommend_default,
        "recommendation": report.recommendation,
        "evaluation": {"first_test_day": report.first_test_day, "last_test_day": report.last_test_day, "paired": report.paired},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_coefficients(root: Path) -> dict[str, Coefficients] | None:
    path = coefficients_path(root)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {feature: Coefficients.from_dict((payload.get("coefficients") or {}).get(feature)) for feature in FEATURES}


def make_adjuster(features: AvailabilityFeatures, coefficients: dict[str, Coefficients], *, now: pd.Timestamp | None = None):
    """Production hook for ``pipeline.score_fixtures``.

    Uses the confirmed starting XI when SportMonks marks it confirmed and the
    lineup coefficients are fit; otherwise the sidelined list, if it was
    fetched within the last 36 hours. Anything else leaves Dixon–Coles as is.
    """

    now = pd.Timestamp(datetime.now(timezone.utc)) if now is None else pd.Timestamp(now)

    def adjust(row):
        fixture_id = int(row["fixture_id"])
        meta = features.fetch_meta.get(fixture_id)
        if meta is None:
            return GoalMultipliers(), {"used": None, "reason": "没有这场的伤停/阵容数据"}
        fetched_at = pd.to_datetime(meta.get("fetched_at"), utc=True, errors="coerce")
        if pd.isna(fetched_at) or (now - fetched_at) > pd.Timedelta(hours=STALE_AFTER_HOURS):
            return GoalMultipliers(), {"used": None, "reason": "伤停数据超过 36 小时没有更新"}
        info = features.fixture(row)
        confirmed = str(meta.get("lineup_confirmed")).lower() == "true"
        lineup = coefficients.get("lineup_gap") or Coefficients()
        if confirmed and lineup.active and info["home_lineup_gap"] is not None and info["away_lineup_gap"] is not None:
            multipliers = lineup.multipliers(info["home_lineup_gap"], info["away_lineup_gap"])
            used = "lineup_gap"
        else:
            missing = coefficients.get("missing") or Coefficients()
            if not missing.active or info["home_missing"] is None or info["away_missing"] is None:
                return GoalMultipliers(), {"used": None, "reason": "系数未拟合或球队首发历史不足"}
            multipliers = missing.multipliers(info["home_missing"], info["away_missing"])
            used = "missing"
        return multipliers, {
            "used": used,
            "fetched_at": pd.Timestamp(fetched_at).isoformat(),
            "home_missing": info["home_missing"],
            "away_missing": info["away_missing"],
            "home_lineup_gap": info["home_lineup_gap"],
            "away_lineup_gap": info["away_lineup_gap"],
            "home_multiplier": multipliers.home,
            "away_multiplier": multipliers.away,
            "home_absent": info["home_absent"],
            "away_absent": info["away_absent"],
        }

    return adjust
