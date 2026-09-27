"""Walk-forward blends of Dixon–Coles and the de-vigged pre-match 1X2.

The weight for a test day is fit only on earlier evaluation days: out-of-sample
Dixon–Coles probabilities and the pre-match market on those matches. The day's
own results are not in that fit. A fixed weight chosen on the scored matches
themselves is reported separately and labelled in-sample.
"""

from __future__ import annotations

import numpy as np

from laliga.model.devig import DEVIG_METHODS
from laliga.model.metrics import multiclass_log_loss

MIN_BLEND_HISTORY = 60
_PROB_FLOOR = 1e-15
# Renormalising an identical pair can move log loss by a couple of ulps.
# Improvements smaller than this stay with the earlier, more market-side grid point.
_LOSS_TIE = 1e-10


def linear_pool(dixon_coles: np.ndarray, market: np.ndarray, weight: float) -> np.ndarray:
    """``weight`` on Dixon–Coles and ``1 - weight`` on the market, then renorm."""

    left = _probabilities(dixon_coles)
    right = _probabilities(market)
    mixed = float(weight) * left + (1.0 - float(weight)) * right
    return _probabilities(mixed)


def log_pool(dixon_coles: np.ndarray, market: np.ndarray, weight: float) -> np.ndarray:
    """Weighted geometric mean of the two probability vectors, then renorm."""

    left = _probabilities(dixon_coles)
    right = _probabilities(market)
    mixed = float(weight) * np.log(left) + (1.0 - float(weight)) * np.log(right)
    return _softmax(mixed)


def stack_pool(dixon_coles: np.ndarray, market: np.ndarray, dixon_coles_coef: float, market_coef: float) -> np.ndarray:
    """Softmax of ``a * log(Dixon–Coles) + b * log(market)`` with ``a, b >= 0``.

    There is no class intercept, so the blend cannot add a home-win offset that
    neither input has. ``a = b = 0`` is the uniform distribution.
    """

    left = _probabilities(dixon_coles)
    right = _probabilities(market)
    mixed = float(dixon_coles_coef) * np.log(left) + float(market_coef) * np.log(right)
    return _softmax(mixed)


def fit_pool_weight(y: np.ndarray, dixon_coles: np.ndarray, market: np.ndarray, pool) -> float:
    """Smallest grid weight in ``[0, 1]`` that minimises log loss.

    The grid is ``0, 0.01, ..., 1``. Ties keep the smaller weight, which puts
    more of the blend on the market.
    """

    best_weight = 0.0
    best_loss = np.inf
    for weight in _weight_grid():
        loss = multiclass_log_loss(y, pool(dixon_coles, market, weight))
        if loss < best_loss - _LOSS_TIE:
            best_loss = loss
            best_weight = weight
    return best_weight


def fit_stack(y: np.ndarray, dixon_coles: np.ndarray, market: np.ndarray) -> tuple[float, float]:
    """Non-negative ``(a, b)`` on a 0.25 grid from 0 to 2.

    Ties keep the pair visited first, so a smaller Dixon–Coles coefficient
    wins, and for that coefficient a smaller market coefficient wins.
    """

    best = (0.0, 0.0)
    best_loss = np.inf
    for dixon_coles_coef in _stack_grid():
        for market_coef in _stack_grid():
            loss = multiclass_log_loss(y, stack_pool(dixon_coles, market, dixon_coles_coef, market_coef))
            if loss < best_loss - _LOSS_TIE:
                best_loss = loss
                best = (dixon_coles_coef, market_coef)
    return best


def walk_forward_blends(rows: list[dict], *, min_history: int = MIN_BLEND_HISTORY) -> dict:
    """Write blend probabilities onto later days and describe the weight path.

    Rows need ``match_day``, ``fixture_id``, ``y_ft``, ``dixon_coles_ft_probs``,
    and ``market_ft_probs``. Matches missing either probability are ignored.
    A day is scored only after ``min_history`` earlier matches, and those
    earlier matches are the only ones used to choose that day's weight.
    """

    if min_history < 1:
        raise ValueError("混合的热身场次至少是 1。")
    chosen = [
        row
        for row in rows
        if row.get("dixon_coles_ft_probs") is not None and row.get("market_ft_probs") is not None
    ]
    chosen.sort(key=lambda row: (str(row["match_day"]), int(row["fixture_id"])))
    if not chosen:
        return _unavailable(
            0,
            min_history,
            "评测比赛里没有同时具备 Dixon–Coles 概率和完整赛前 1X2 的场次，因此没有做混合。",
        )

    days: list[tuple[str, list[dict]]] = []
    for row in chosen:
        day = str(row["match_day"])
        if not days or days[-1][0] != day:
            days.append((day, []))
        days[-1][1].append(row)

    history: list[dict] = []
    scored: list[dict] = []
    linear_path: list[dict] = []
    log_path: list[dict] = []
    stack_path: list[dict] = []
    devig_path: list[dict] = []
    for day, day_rows in days:
        if len(history) >= min_history:
            method = _choose_devig(history)
            market_key = _devig_prob_key(method)
            if any(row.get(market_key) is None for row in day_rows):
                method = "proportional"
                market_key = "market_ft_probs"
            y_hist = np.array([item["y_ft"] for item in history], dtype=int)
            dc_hist = _matrix(history, "dixon_coles_ft_probs")
            market_hist = _matrix(history, market_key)
            linear_weight = fit_pool_weight(y_hist, dc_hist, market_hist, linear_pool)
            log_weight = fit_pool_weight(y_hist, dc_hist, market_hist, log_pool)
            stack_a, stack_b = fit_stack(y_hist, dc_hist, market_hist)
            dc_day = _matrix(day_rows, "dixon_coles_ft_probs")
            market_day = _matrix(day_rows, market_key)
            linear = linear_pool(dc_day, market_day, linear_weight)
            logarithmic = log_pool(dc_day, market_day, log_weight)
            stacked = stack_pool(dc_day, market_day, stack_a, stack_b)
            for index, row in enumerate(day_rows):
                row["blend_market_ft_probs"] = _as_tuple(market_day[index])
                row["blend_devig_method"] = method
                row["blend_linear_ft_probs"] = _as_tuple(linear[index])
                row["blend_log_ft_probs"] = _as_tuple(logarithmic[index])
                row["blend_stack_ft_probs"] = _as_tuple(stacked[index])
                row["blend_linear_weight"] = linear_weight
                row["blend_log_weight"] = log_weight
                row["blend_stack_a"] = stack_a
                row["blend_stack_b"] = stack_b
            scored.extend(day_rows)
            linear_path.append({"day": day, "n_history": len(history), "weight": linear_weight})
            log_path.append({"day": day, "n_history": len(history), "weight": log_weight})
            stack_path.append({"day": day, "n_history": len(history), "a": stack_a, "b": stack_b})
            devig_path.append({"day": day, "n_history": len(history), "method": method})
        history.extend(day_rows)

    if not scored:
        return _unavailable(
            len(chosen),
            min_history,
            f"同时有 Dixon–Coles 和完整赛前 1X2 的评测比赛有 {len(chosen)} 场，"
            f"少于走步拟合权重所需的 {min_history} 场，因此没有混合结果。",
        )

    y_scored = np.array([row["y_ft"] for row in scored], dtype=int)
    dc_scored = _matrix(scored, "dixon_coles_ft_probs")
    # The oracle refits the pool weight only. It does not pick another de-vig method.
    market_scored = _matrix(scored, "blend_market_ft_probs")
    oracle_linear = fit_pool_weight(y_scored, dc_scored, market_scored, linear_pool)
    oracle_log = fit_pool_weight(y_scored, dc_scored, market_scored, log_pool)
    oracle_a, oracle_b = fit_stack(y_scored, dc_scored, market_scored)
    return {
        "available": True,
        "min_history": int(min_history),
        "eligible_n": len(chosen),
        "warmup_n": len(chosen) - len(scored),
        "scored_n": len(scored),
        "weight_fit": "walk_forward_previous_evaluation_days_only",
        "devig": _devig_summary(devig_path),
        "linear": {
            "formula": "w*dixon_coles + (1-w)*market",
            "weight_on_dixon_coles": _path_summary(linear_path, "weight"),
            "path": linear_path,
            "oracle": _oracle_weight(oracle_linear, y_scored, linear_pool(dc_scored, market_scored, oracle_linear)),
        },
        "log": {
            "formula": "renormalised dixon_coles^w * market^(1-w)",
            "weight_on_dixon_coles": _path_summary(log_path, "weight"),
            "path": log_path,
            "oracle": _oracle_weight(oracle_log, y_scored, log_pool(dc_scored, market_scored, oracle_log)),
        },
        "stack": {
            "formula": "softmax(a*log(dixon_coles) + b*log(market))",
            "a": _path_summary(stack_path, "a"),
            "b": _path_summary(stack_path, "b"),
            "median_components_are_separate": True,
            "path": stack_path,
            "oracle": {
                "in_sample": True,
                "role": "hindsight_upper_bound",
                "label": "事后最优固定权重（样本内上界，不是走步结果）",
                "a": oracle_a,
                "b": oracle_b,
                "ft_log_loss": multiclass_log_loss(y_scored, stack_pool(dc_scored, market_scored, oracle_a, oracle_b)),
                "n": int(len(scored)),
            },
        },
    }


def blend_notes(description: dict) -> list[str]:
    """Chinese notes for the comparison report. Oracle text says it is in-sample."""

    if not description["available"]:
        return [description["reason"]]
    linear = description["linear"]["weight_on_dixon_coles"]
    logarithmic = description["log"]["weight_on_dixon_coles"]
    stack_a = description["stack"]["a"]
    stack_b = description["stack"]["b"]
    oracle_linear = description["linear"]["oracle"]
    oracle_log = description["log"]["oracle"]
    oracle_stack = description["stack"]["oracle"]
    return [
        "Dixon–Coles 与去水位赔率的混合只评全场，且只在同时有两者的比赛上计算。"
        "权重按评测日走步拟合：某一天只用更早评测日的样本外 Dixon–Coles 概率和那些场的赛前赔率，当天的结果不进入当天的权重。",
        (
            f"至少先有 {description['min_history']} 场这样的历史才开始计分。"
            f"热身 {description['warmup_n']} 场不进入混合的对数损失。计分 {description['scored_n']} 场。"
            f"表里「混合同一批」的 Dixon–Coles 和赔率行、以及混合减赔率的配对区间，用的都是这 {description['scored_n']} 场。"
        ),
        "线性混合是 w*Dixon–Coles + (1-w)*赔率。对数混合是两项概率的加权几何平均，再归一化。"
        "w 在 0 到 1 上以 0.01 搜索对数损失，打平时取更小的 w（更靠近赔率）。",
        "对数线性叠加是 softmax(a*log(Dixon–Coles) + b*log(赔率))，a 和 b 都不小于 0，在 0 到 2 上以 0.25 搜索。"
        "它没有结果类别的截距，不能单独抬高主胜。如果两个模型在同一档主胜概率上一起偏低，混合补不上。",
        (
            f"线性混合里 Dixon–Coles 的权重：起点 {_fmt(linear['start'])}（{linear['start_day']}），"
            f"中位数 {_fmt(linear['median'])}，终点 {_fmt(linear['end'])}（{linear['end_day']}）。"
            "天数为偶数时，中位数是中间两天的算术平均。"
            f"对数混合：起点 {_fmt(logarithmic['start'])}，中位数 {_fmt(logarithmic['median'])}，终点 {_fmt(logarithmic['end'])}。"
        ),
        (
            f"对数线性叠加的 a：起点 {_fmt(stack_a['start'])}，中位数 {_fmt(stack_a['median'])}，终点 {_fmt(stack_a['end'])}。"
            f"b：起点 {_fmt(stack_b['start'])}，中位数 {_fmt(stack_b['median'])}，终点 {_fmt(stack_b['end'])}。"
            "a 和 b 的中位数分开计算，不一定是同一天的一对。"
        ),
        _devig_note(description.get("devig")),
        (
            "事后最优固定权重只作样本内上界，不是走步结果，不能用来判断混合有没有打败赔率。"
            f"它只在计分的 {description['scored_n']} 场上重拟合混合权重，不重新选择去水位方法，热身场次不参与。"
            f"线性 w={_fmt(oracle_linear['weight_on_dixon_coles'])}（对数损失 {_fmt(oracle_linear['ft_log_loss'])}），"
            f"对数 w={_fmt(oracle_log['weight_on_dixon_coles'])}（对数损失 {_fmt(oracle_log['ft_log_loss'])}），"
            f"叠加 a={_fmt(oracle_stack['a'])}、b={_fmt(oracle_stack['b'])}（对数损失 {_fmt(oracle_stack['ft_log_loss'])}）。"
        ),
    ]


def _choose_devig(rows: list[dict]) -> str:
    """Pick the de-vig method with the lowest log loss on ``rows``.

    Proportional is first in ``DEVIG_METHODS``. A later method replaces it
    only when its log loss is lower by more than the tie tolerance. A method
    that is missing on any of these rows is not eligible, so the choice cannot
    use a price that appears only on the day being scored.
    """

    y = np.array([row["y_ft"] for row in rows], dtype=int)
    best = "proportional"
    best_loss = multiclass_log_loss(y, _matrix(rows, "market_ft_probs"))
    for method in DEVIG_METHODS:
        if method == "proportional":
            continue
        key = _devig_prob_key(method)
        if any(row.get(key) is None for row in rows):
            continue
        loss = multiclass_log_loss(y, _matrix(rows, key))
        if loss < best_loss - _LOSS_TIE:
            best_loss = loss
            best = method
    return best


def _devig_prob_key(method: str) -> str:
    if method == "proportional":
        return "market_ft_probs"
    return f"market_{method}_ft_probs"


def _devig_summary(path: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for item in path:
        counts[item["method"]] = counts.get(item["method"], 0) + 1
    return {
        "selection": "walk_forward_previous_evaluation_days_only",
        "tie_break": "proportional",
        "oracle_repeats_walk_forward_method": True,
        "closing_odds_excluded": True,
        "start": path[0]["method"],
        "end": path[-1]["method"],
        "counts": counts,
        "path": path,
    }


def _devig_note(devig: dict | None) -> str:
    if not devig:
        return "去水位方法没有单独走步选择，混合用的是比例去水位。"
    counts = "、".join(f"{name} {count} 天" for name, count in sorted(devig["counts"].items()))
    return (
        "混合用的赔率方法也按更早评测日的对数损失走步选择，当天和以后的结果不参与。"
        f"打平时留在比例去水位。起点 {devig['start']}，终点 {devig['end']}（{counts}）。"
        "事后最优权重只在已经选定的方法上重拟合混合系数，不会在计分比赛上改选 Shin 或幂。"
        "收盘赔率不进入混合。"
    )


def _unavailable(eligible_n: int, min_history: int, reason: str) -> dict:
    return {
        "available": False,
        "min_history": int(min_history),
        "eligible_n": int(eligible_n),
        "warmup_n": int(eligible_n),
        "scored_n": 0,
        "reason": reason,
    }


def _oracle_weight(weight: float, y: np.ndarray, probabilities: np.ndarray) -> dict:
    return {
        "in_sample": True,
        "role": "hindsight_upper_bound",
        "label": "事后最优固定权重（样本内上界，不是走步结果）",
        "weight_on_dixon_coles": weight,
        "ft_log_loss": multiclass_log_loss(y, probabilities),
        "n": int(len(y)),
    }


def _path_summary(path: list[dict], key: str) -> dict:
    values = np.array([float(item[key]) for item in path], dtype=float)
    return {
        "start": float(values[0]),
        "median": float(np.median(values)),
        "end": float(values[-1]),
        "start_day": path[0]["day"],
        "end_day": path[-1]["day"],
        "n_days": len(path),
    }


def _weight_grid() -> list[float]:
    return [round(step / 100, 2) for step in range(101)]


def _stack_grid() -> list[float]:
    return [round(step * 0.25, 2) for step in range(9)]


def _probabilities(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    clipped = np.clip(matrix, _PROB_FLOOR, None)
    return clipped / clipped.sum(axis=1, keepdims=True)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    mixed = np.exp(shifted)
    return mixed / mixed.sum(axis=1, keepdims=True)


def _matrix(rows: list[dict], key: str) -> np.ndarray:
    return np.array([row[key] for row in rows], dtype=float)


def _as_tuple(vector: np.ndarray) -> tuple[float, float, float]:
    return (float(vector[0]), float(vector[1]), float(vector[2]))


def _fmt(value: float) -> str:
    return f"{float(value):.4f}"
