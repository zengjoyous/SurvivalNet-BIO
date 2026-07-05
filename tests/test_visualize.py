import pandas as pd

from survivalnet.visualize import (
    plot_feature_importance,
    plot_grouped_km,
    plot_km_curve,
    plot_risk_score_distribution,
)


def test_plot_km_curve_draws():
    import matplotlib.pyplot as plt

    from survivalnet.core import fit_km

    data = pd.DataFrame({"time": [1, 2, 3], "event": [1, 0, 1]})
    kmf = fit_km(data, "time", "event")
    ax = plot_km_curve(kmf)
    assert ax is not None
    plt.close(ax.figure)


def test_plot_grouped_km_draws():
    import matplotlib.pyplot as plt

    data = pd.DataFrame(
        {
            "time": [1, 2, 3, 4],
            "event": [1, 0, 1, 0],
            "group": ["A", "A", "B", "B"],
        }
    )
    ax = plot_grouped_km(data, "time", "event", "group")
    assert ax is not None
    plt.close(ax.figure)


def test_plot_risk_score_distribution_accepts_risk_group():
    import matplotlib.pyplot as plt

    data = pd.DataFrame(
        {
            "risk_score": [0.1, 0.3, 0.7, 1.0],
            "risk_group": ["low", "low", "high", "high"],
        }
    )
    ax = plot_risk_score_distribution(data, "risk_score", "risk_group")
    assert ax is not None
    plt.close(ax.figure)


def test_plot_feature_importance_draws():
    import matplotlib.pyplot as plt

    hazard_ratios = pd.DataFrame(
        {
            "HR": [1.5, 0.8, 2.2],
            "CI_lower": [1.1, 0.5, 1.4],
            "CI_upper": [2.1, 1.1, 3.0],
            "p_value": [0.03, 0.2, 0.01],
        },
        index=["gene1", "gene2", "gene3"],
    )
    ax = plot_feature_importance(hazard_ratios)
    assert ax is not None
    plt.close(ax.figure)
