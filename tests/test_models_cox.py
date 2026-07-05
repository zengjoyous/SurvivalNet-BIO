import pandas as pd

from survivalnet.models import CoxModel


def make_data():
    return pd.DataFrame(
        {
            "time": [5, 6, 6, 2, 4, 3],
            "event": [1, 0, 1, 1, 0, 1],
            "x1": [0.2, 0.1, 0.3, 0.8, 0.4, 0.6],
            "x2": [2.0, 2.1, 1.9, 3.0, 2.4, 2.6],
        }
    )


def test_cox_model_fit_uses_high_variance_features_first():
    data = make_data()
    model = CoxModel(penalizer=0.1).fit(data, "time", "event")

    assert model.feature_cols == ["x2", "x1"]
    assert not model.summary.empty
    assert not model.hazard_ratios.empty


def test_cox_model_predict_risk_score_returns_series():
    data = make_data()
    model = CoxModel(penalizer=0.1).fit(data, "time", "event")
    scores = model.predict_risk_score(data)

    assert list(scores.index) == list(data.index)
    assert scores.name == "risk_score"
