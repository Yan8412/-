import pandas as pd

from laliga.backtest import walk_forward
from laliga.config import ModelConfig
from laliga.data.parse import fixtures_to_frame
from laliga.model.metrics import multiclass_brier, multiclass_log_loss, accuracy
from laliga.schedule import select_next_matchday
from laliga.synthetic import generate_synthetic_fixtures
import numpy as np


def test_metric_definitions():
    y = np.array([0, 1])
    probabilities = np.array([[0.5, 0.3, 0.2], [0.2, 0.6, 0.2]])
    assert multiclass_log_loss(y, probabilities) == np.mean([-np.log(0.5), -np.log(0.6)])
    first = (0.5 - 1) ** 2 + 0.3**2 + 0.2**2
    second = 0.2**2 + (0.6 - 1) ** 2 + 0.2**2
    assert multiclass_brier(y, probabilities) == np.mean([first, second])
    assert accuracy(y, probabilities) == 1.0


def test_walk_forward_beats_frequency_baseline_on_synthetic_league():
    frame = fixtures_to_frame(generate_synthetic_fixtures(seed=7, today=pd.Timestamp("2026-09-25").date()))
    report = walk_forward(
        frame,
        ModelConfig(xi=0.001, max_iter=80, min_train_matches=48, min_matches=8, min_ht_matches=20, l2=4.0),
    )
    assert report.n_folds > 1
    assert report.model.n > 10
    assert 0 < report.model.ft_log_loss < 2
    assert 0 < report.model.ft_brier < 2
    assert 0 <= report.model.ft_accuracy <= 1
    assert 0 <= report.model.top3_hit_rate <= 1
    assert report.model.ht_log_loss is not None
    assert report.model.ft_log_loss < report.baseline.ft_log_loss
    assert report.model.top3_hit_rate >= report.baseline.top3_hit_rate - 0.05
    assert report.model_by_season


def test_next_matchday_is_the_unplayed_round():
    today = pd.Timestamp("2026-09-25").date()
    frame = fixtures_to_frame(generate_synthetic_fixtures(seed=7, today=today))
    chosen, note = select_next_matchday(frame, now=pd.Timestamp(today, tz="UTC"))
    assert note == ""
    assert len(chosen) == 3
    assert set(chosen["status"]) == {"scheduled"}
    assert chosen["round_id"].nunique() == 1
    assert chosen["starting_at"].min() > pd.Timestamp(today, tz="UTC")
