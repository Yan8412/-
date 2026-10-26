"""Football, market, and blend tracks on one walk-forward cut.

The football track is the joint half-time / full-time Dixon–Coles model.
The market track backs expected goals out of that match's own pre-match
prices (1X2, and over/under or Asian handicap when they are stored). The
blend is chosen on earlier evaluation days only: a geometric mean of the
four half rates, and, separately, the existing probability log pool. The
day's results are not used to pick that day's weight.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from laliga.config import ModelConfig
from laliga.data.quotes import direct_probabilities, load_quotes, targets_from_bundle, targets_from_row
from laliga.model.blend import log_pool
from laliga.model.dixon_coles import FitError
from laliga.model.implied_goals import ImpliedRates, infer_rates
from laliga.model.joint import HTFT_CODES, HalfRates, convolve, htft_code
from laliga.model.metrics import (
    iter_evaluation_folds,
    multiclass_brier,
    multiclass_log_loss,
    multiclass_rps,
    outcome_index,
)
from laliga.model.service import ScorelineModel, fit_models

MIN_BLEND_HISTORY = 40
logger = logging.getLogger(__name__)


@dataclass
class TrackReport:
    payload: dict

    def to_dict(self) -> dict:
        return self.payload


def compare_tracks(
    matches: pd.DataFrame,
    config: ModelConfig,
    *,
    quotes: dict | None = None,
    min_blend_history: int = MIN_BLEND_HISTORY,
    refit_every_days: int = 1,
    notify=None,
) -> TrackReport:
    finished = matches[matches["status"] == "finished"].dropna(subset=["home_goals_ft", "away_goals_ft"])
    if finished.empty:
        raise FitError("没有完场比赛，无法比较三条预测。")
    rows = _walk(finished, config, quotes or {}, refit_every_days=refit_every_days, notify=notify)
    if not rows:
        raise FitError(
            f"三条预测没有评测任何比赛。完场 {len(finished)} 场，最少训练样本 {config.min_train_matches}。"
        )
    _attach_blends(rows, min_history=min_blend_history, notify=notify)
    return TrackReport(_summarize(rows, config, min_blend_history))


def market_rates_for_row(row, quotes: dict | None, *, devig_method: str = "proportional") -> ImpliedRates | None:
    bundle = None
    if quotes:
        bundle = quotes.get(str(int(row["fixture_id"])))
    targets = targets_from_bundle(bundle, devig_method=devig_method) if bundle else None
    if targets is None:
        targets = targets_from_row(row)
    if targets is None:
        return None
    rho = -0.05
    targets.rho = rho
    return infer_rates(targets)


def geometric_rates(football: HalfRates, market: HalfRates, market_weight: float) -> HalfRates:
    """``market_weight`` on the market rates and the rest on the football rates."""

    weight = float(np.clip(market_weight, 0.0, 1.0))

    def mix(left: float, right: float) -> float:
        left = max(left, 1e-6)
        right = max(right, 1e-6)
        return float(math.exp((1.0 - weight) * math.log(left) + weight * math.log(right)))

    return HalfRates(
        home_1=mix(football.home_1, market.home_1),
        away_1=mix(football.away_1, market.away_1),
        home_2=mix(football.home_2, market.home_2),
        away_2=mix(football.away_2, market.away_2),
        rho_1=(1.0 - weight) * football.rho_1 + weight * market.rho_1,
        rho_2=(1.0 - weight) * football.rho_2 + weight * market.rho_2,
        eta=(1.0 - weight) * football.eta + weight * market.eta,
    )


def _walk(finished: pd.DataFrame, config: ModelConfig, quotes: dict, *, refit_every_days: int, notify=None) -> list[dict]:
    rows: list[dict] = []
    previous: ScorelineModel | None = None
    last_fit: pd.Timestamp | None = None
    model: ScorelineModel | None = None
    for day, train, test in iter_evaluation_folds(finished, config.min_train_matches):
        as_of = pd.Timestamp(day)
        _say(
            f"评测日 {as_of.date().isoformat()}，训练 {len(train)} 场，评测 {len(test)} 场",
            notify,
        )
        due = last_fit is None or (as_of - last_fit).days >= max(refit_every_days, 1)
        if due or model is None:
            try:
                model = fit_models(
                    train,
                    as_of,
                    config,
                    initial_ft=None if previous is None else previous.ft,
                    initial_ht=None if previous is None else previous.ht,
                    initial_sh=None if previous is None else previous.sh,
                )
            except FitError:
                continue
            previous = model
            last_fit = as_of
        assert model is not None
        for _, match in test.iterrows():
            rows.append(_one(match, model, quotes, as_of))
    return rows


def _one(match, model: ScorelineModel, quotes: dict, as_of: pd.Timestamp) -> dict:
    home_ft = int(match["home_goals_ft"])
    away_ft = int(match["away_goals_ft"])
    home_ht = away_ht = None
    if pd.notna(match.get("home_goals_ht")) and pd.notna(match.get("away_goals_ht")):
        home_ht = int(match["home_goals_ht"])
        away_ht = int(match["away_goals_ht"])
    prediction = model.predict_row(match)
    home_id = int(match["home_team_id"])
    away_id = int(match["away_team_id"])
    football_rates = model.half_rates(home_id, away_id)
    implied = market_rates_for_row(match, quotes)
    market_probs = None
    market_ht = None
    market_top2 = None
    market_htft = None
    market_top2_htft = None
    market_rates = None
    if implied is not None:
        forecast = implied.forecast(eta=0.0, max_half=6, max_ft=max(model.ft.max_goals, 12))
        market_probs = forecast.ft_probs
        market_ht = forecast.ht_probs
        market_top2 = [(home, away) for home, away, _ in forecast.top2_scores]
        market_htft = forecast.htft
        market_top2_htft = [code for code, _probability in forecast.top_htft[:2]]
        market_rates = implied.half_rates(0.0)
    bundle = quotes.get(str(int(match["fixture_id"]))) if quotes else None
    direct = direct_probabilities(bundle)
    return {
        "fixture_id": int(match["fixture_id"]),
        "match_day": pd.Timestamp(as_of).date().isoformat(),
        "y_ft": outcome_index(home_ft, away_ft),
        "y_ht": None if home_ht is None else outcome_index(home_ht, away_ht),
        "score": (home_ft, away_ft),
        "ht_score": None if home_ht is None else (home_ht, away_ht),
        "htft_code": None if home_ht is None else htft_code(home_ht, away_ht, home_ft, away_ft),
        "football_ft": prediction.ft.as_tuple(),
        "football_ht": prediction.ht.as_tuple(),
        "football_top2": [(item.home_goals, item.away_goals) for item in prediction.top2_scores],
        "football_top3": [(item.home_goals, item.away_goals) for item in prediction.top_scores],
        "football_htft": dict(prediction.htft) if prediction.htft else None,
        "football_top2_htft": [code for code, _probability in (prediction.top_htft or [])[:2]],
        "football_rates": None if football_rates is None else football_rates,
        "market_ft": market_probs,
        "market_ht": market_ht,
        "market_top2": market_top2,
        "market_htft": market_htft,
        "market_top2_htft": market_top2_htft if implied is not None else None,
        "market_rates": market_rates,
        "direct": direct,
    }


def _attach_blends(rows: list[dict], *, min_history: int, notify=None) -> None:
    usable = [row for row in rows if row["market_ft"] is not None and row["football_rates"] is not None and row["market_rates"] is not None]
    usable.sort(key=lambda row: (row["match_day"], row["fixture_id"]))
    days: list[tuple[str, list[dict]]] = []
    for row in usable:
        if not days or days[-1][0] != row["match_day"]:
            days.append((row["match_day"], []))
        days[-1][1].append(row)
    grid = [index / 20 for index in range(21)]
    _say(f"开始计算混合权重，可用比赛 {len(usable)} 场。", notify)
    for index, row in enumerate(usable):
        if index % 50 == 0:
            _say(f"混合权重预计算 {index}/{len(usable)} 场", notify)
        cached = []
        for weight in grid:
            blended = geometric_rates(row["football_rates"], row["market_rates"], weight)
            cached.append(convolve(blended, max_half=5, max_ft=8).ft_probs)
        row["_ft_by_weight"] = cached
    history: list[dict] = []
    for _day, day_rows in days:
        if len(history) >= min_history:
            weight = _best_rate_weight(history, grid)
            log_weight = _best_log_weight(history, grid)
            _say(f"混合评测日 {_day}，历史 {len(history)} 场，市场权重 {weight:.2f}", notify)
            for row in day_rows:
                blended = geometric_rates(row["football_rates"], row["market_rates"], weight)
                forecast = convolve(blended, max_half=6, max_ft=12)
                row["blend_rate_ft"] = forecast.ft_probs
                row["blend_rate_ht"] = forecast.ht_probs
                row["blend_rate_htft"] = forecast.htft
                row["blend_rate_top2"] = [(home, away) for home, away, _ in forecast.top2_scores]
                row["blend_rate_top2_htft"] = [code for code, _probability in forecast.top_htft[:2]]
                row["blend_rate_weight"] = weight
                row["blend_log_ft"] = tuple(
                    log_pool(np.array([row["football_ft"]]), np.array([row["market_ft"]]), log_weight)[0]
                )
                row["blend_log_weight"] = log_weight
        history.extend(day_rows)
    for row in usable:
        row.pop("_ft_by_weight", None)


def _best_rate_weight(history: list[dict], grid: list[float]) -> float:
    y = np.array([row["y_ft"] for row in history], dtype=int)
    best_weight = 0.0
    best_loss = np.inf
    cached = all(isinstance(row.get("_ft_by_weight"), list) and len(row["_ft_by_weight"]) == len(grid) for row in history)
    for index, weight in enumerate(grid):
        if cached:
            probs = np.array([row["_ft_by_weight"][index] for row in history], dtype=float)
        else:
            probs = []
            for row in history:
                blended = geometric_rates(row["football_rates"], row["market_rates"], weight)
                probs.append(convolve(blended, max_half=5, max_ft=8).ft_probs)
            probs = np.array(probs, dtype=float)
        loss = multiclass_log_loss(y, probs)
        if loss < best_loss - 1e-10:
            best_loss = loss
            best_weight = weight
    return best_weight


def _best_log_weight(history: list[dict], grid: list[float]) -> float:
    """Weight on the football model, matching ``log_pool`` (the rest is the market)."""

    y = np.array([row["y_ft"] for row in history], dtype=int)
    football = np.array([row["football_ft"] for row in history], dtype=float)
    market = np.array([row["market_ft"] for row in history], dtype=float)
    best_weight = 0.0
    best_loss = np.inf
    for weight in grid:
        loss = multiclass_log_loss(y, log_pool(football, market, weight))
        if loss < best_loss - 1e-10:
            best_loss = loss
            best_weight = weight
    return best_weight


def _summarize(rows: list[dict], config: ModelConfig, min_blend_history: int) -> dict:
    football = _block(rows, "football_ft", "football_ht", "football_top2", "football_top3", "football_htft", "football_top2_htft")
    paired = [row for row in rows if row.get("market_ft") is not None]
    market = _block(paired, "market_ft", "market_ht", "market_top2", None, "market_htft", "market_top2_htft")
    football_on_market = _block(
        paired, "football_ft", "football_ht", "football_top2", "football_top3", "football_htft", "football_top2_htft"
    )
    blended = [row for row in rows if row.get("blend_rate_ft") is not None]
    blend_rate = _block(
        blended, "blend_rate_ft", "blend_rate_ht", "blend_rate_top2", None, "blend_rate_htft", "blend_rate_top2_htft"
    )
    blend_log = _block(blended, "blend_log_ft", None, None, None, None)
    market_on_blend = _block(blended, "market_ft", "market_ht", "market_top2", None, "market_htft", "market_top2_htft")
    football_on_blend = _block(
        blended, "football_ft", "football_ht", "football_top2", "football_top3", "football_htft", "football_top2_htft"
    )
    direct = _direct_block(rows)
    return {
        "schema_version": 1,
        "kind": "tracks",
        "computed_from_stored_matches": True,
        "min_train_matches": config.min_train_matches,
        "min_blend_history": min_blend_history,
        "n_football": len(rows),
        "n_market": len(paired),
        "n_blend": len(blended),
        "tracks": {
            "football": football,
            "football_on_market_rows": football_on_market,
            "market": market,
            "market_on_blend_rows": market_on_blend,
            "football_on_blend_rows": football_on_blend,
            "blend_rates": blend_rate,
            "blend_log": blend_log,
        },
        "direct_markets": direct,
        "blend_rate_weight": _weight_summary(blended),
        "notes": _notes(rows, paired, blended, min_blend_history),
    }


def _block(rows, ft_key, ht_key, top2_key, top3_key, htft_key, top2_htft_key=None) -> dict:
    if not rows or any(row.get(ft_key) is None for row in rows):
        return {"n": 0}
    y = np.array([row["y_ft"] for row in rows], dtype=int)
    probs = np.array([row[ft_key] for row in rows], dtype=float)
    block = {
        "n": len(rows),
        "ft_log_loss": multiclass_log_loss(y, probs),
        "ft_brier": multiclass_brier(y, probs),
        "ft_rps": multiclass_rps(y, probs),
        "ft_accuracy": float(np.mean(np.argmax(probs, axis=1) == y)),
        "top2_hit_rate": _hit(rows, top2_key, "score") if top2_key else None,
        "top3_hit_rate": _hit(rows, top3_key, "score") if top3_key else None,
        "top2_htft_hit_rate": _hit(rows, top2_htft_key, "htft_code") if top2_htft_key else None,
    }
    ht_rows = [row for row in rows if row["y_ht"] is not None and ht_key and row.get(ht_key) is not None]
    if ht_rows:
        y_ht = np.array([row["y_ht"] for row in ht_rows], dtype=int)
        ht_probs = np.array([row[ht_key] for row in ht_rows], dtype=float)
        block["n_ht"] = len(ht_rows)
        block["ht_log_loss"] = multiclass_log_loss(y_ht, ht_probs)
        block["ht_brier"] = multiclass_brier(y_ht, ht_probs)
        block["ht_rps"] = multiclass_rps(y_ht, ht_probs)
    htft_rows = [row for row in rows if row["htft_code"] and htft_key and row.get(htft_key)]
    if htft_rows:
        y_htft = np.array([HTFT_CODES.index(row["htft_code"]) for row in htft_rows], dtype=int)
        matrix = np.array([[row[htft_key].get(code, 1e-15) for code in HTFT_CODES] for row in htft_rows], dtype=float)
        block["n_htft"] = len(htft_rows)
        block["htft_log_loss"] = multiclass_log_loss(y_htft, matrix)
        block["htft_brier"] = multiclass_brier(y_htft, matrix)
    return block


def _direct_block(rows: list[dict]) -> dict:
    """Log loss of a direct de-vigged market, only on rows that quoted it."""

    found = {}
    for key in ("htft", "ht_1x2", "correct_score", "odd_even"):
        chosen = []
        for row in rows:
            block = (row.get("direct") or {}).get(key)
            if not block or not block.get("complete"):
                continue
            if key == "htft" and row["htft_code"] in (block.get("probs") or {}):
                chosen.append((row["htft_code"], block["probs"]))
            elif key == "ht_1x2" and row["y_ht"] is not None:
                chosen.append((row["y_ht"], block["probs"]))
            elif key == "correct_score" and _score_text(row["score"]) in (block.get("probs") or {}):
                chosen.append((_score_text(row["score"]), block["probs"]))
            elif key == "odd_even":
                parity = "odd" if sum(row["score"]) % 2 else "even"
                if parity in (block.get("probs") or {}):
                    chosen.append((parity, block["probs"]))
        if len(chosen) < 5:
            found[key] = {"n": len(chosen), "log_loss": None}
            continue
        if key == "ht_1x2":
            order = ("home", "draw", "away")
            y = np.array([item[0] for item in chosen], dtype=int)
            probs = np.array([[item[1][name] for name in order] for item in chosen], dtype=float)
        else:
            labels = sorted({label for _actual, probs in chosen for label in probs})
            y = np.array([labels.index(actual) for actual, _probs in chosen], dtype=int)
            probs = np.array([[item.get(label, 1e-15) for label in labels] for _actual, item in chosen], dtype=float)
        found[key] = {"n": len(chosen), "log_loss": multiclass_log_loss(y, probs)}
    return found


def _hit(rows, key, actual_key) -> float | None:
    if not key:
        return None
    hits = []
    for row in rows:
        predicted = row.get(key)
        actual = row.get(actual_key)
        if not predicted or actual is None:
            continue
        hits.append(actual in predicted)
    if not hits:
        return None
    return float(np.mean(hits))


def _score_text(score: tuple[int, int]) -> str:
    return f"{score[0]}-{score[1]}"


def _weight_summary(rows: list[dict]) -> dict | None:
    """``n_days`` is distinct match days, not the number of distinct weights."""

    if not rows:
        return None
    weights = [float(row["blend_rate_weight"]) for row in rows]
    ordered = sorted(weights)
    mid = ordered[len(ordered) // 2]
    return {
        "start": weights[0],
        "end": weights[-1],
        "median": mid,
        "n_days": len({str(row.get("match_day")) for row in rows}),
        "n_matches": len(rows),
        "n_distinct_weights": len(set(weights)),
    }


def _notes(rows, paired, blended, min_blend_history: int) -> list[str]:
    notes = [
        "纯足球是上下半场 Dixon–Coles 卷积。市场是用这场赛前盘反推的期望进球，再走同一套卷积。",
        "参数混合是两侧半场速率的几何平均，权重只在更早的评测日上按全场对数损失选取。",
        "概率对数混合沿用原有的 1X2 对数池，权重同样只看更早的评测日。",
        "直接盘（半全场、半场胜平负、正确比分、单双）不参与反推进球，只在报过价的场次上单独算对数损失。",
    ]
    if not paired:
        notes.append("这些比赛没有赛前 1X2，所以没有市场预测，也没有混合。")
    elif len(blended) == 0:
        notes.append(f"同时有足球概率和市场概率的历史不足 {min_blend_history} 场，混合还没有开始计分。")
    elif len(blended) != len(paired):
        notes.append(
            f"市场轨道 {len(paired)} 场，混合轨道 {len(blended)} 场。"
            "「市场（与混合同一批比赛）」和「纯足球（与混合同一批比赛）」用已经计分的混合场次，方便直接比较。"
        )
    notes.append("混合权重摘要里的 n_days 是混合已经计分的不同评测日，不是不同权重值的个数。")
    return notes


def _say(text: str, notify) -> None:
    if notify is not None:
        notify(text)
    else:
        logger.info("%s", text)


def quotes_for_root(root) -> dict:
    return load_quotes(root)
