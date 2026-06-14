"""survivalnet public package interface.

This module stays lightweight so data-preparation scripts can import
``survivalnet.workflow`` without pulling in optional modeling dependencies.
The heavier symbols are resolved lazily on first access.
"""

from __future__ import annotations

from importlib import import_module

from ._version import __version__

_LAZY_ATTRS = {
    "c_index": ("survivalnet.core", "c_index"),
    "fit_km": ("survivalnet.core", "fit_km"),
    "logrank_p_value": ("survivalnet.core", "logrank_p_value"),
    "run_logrank_test": ("survivalnet.core", "run_logrank_test"),
    "split_risk_group": ("survivalnet.core", "split_risk_group"),
    "load_clinical_data": ("survivalnet.io", "load_clinical_data"),
    "load_table": ("survivalnet.io", "load_table"),
    "normalize_survival_data": ("survivalnet.io", "normalize_survival_data"),
    "prepare_feature_matrix": ("survivalnet.io", "prepare_feature_matrix"),
    "prepare_survival_dataset": ("survivalnet.io", "prepare_survival_dataset"),
    "split_train_test": ("survivalnet.io", "split_train_test"),
    "split_train_val_test": ("survivalnet.io", "split_train_val_test"),
    "plot_grouped_km": ("survivalnet.visualize", "plot_grouped_km"),
    "plot_grouped_km_with_pvalue": ("survivalnet.visualize", "plot_grouped_km_with_pvalue"),
    "plot_km_curve": ("survivalnet.visualize", "plot_km_curve"),
    "plot_feature_importance": ("survivalnet.visualize", "plot_feature_importance"),
    "plot_risk_score_distribution": ("survivalnet.visualize", "plot_risk_score_distribution"),
    "CoxModel": ("survivalnet.models", "CoxModel"),
    "LassoCoxModel": ("survivalnet.models", "LassoCoxModel"),
    "DeepSurvModel": ("survivalnet.models", "DeepSurvModel"),
    "build_analysis_table": ("survivalnet.workflow", "build_analysis_table"),
    "load_input_tables": ("survivalnet.workflow", "load_input_tables"),
    "save_split_tables": ("survivalnet.workflow", "save_split_tables"),
    "summarize_model_results": ("survivalnet.workflow", "summarize_model_results"),
    "train_baseline_models": ("survivalnet.workflow", "train_baseline_models"),
}

__all__ = [
    "__version__",
    *sorted(_LAZY_ATTRS),
]


def __getattr__(name: str):
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_ATTRS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
