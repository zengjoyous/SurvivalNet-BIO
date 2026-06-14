"""DeepSurv neural-network survival model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from pycox.models import CoxPH

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from lifelines.utils import concordance_index

from torchtuples import optim
import torchtuples as tt

from ..exceptions import ModelNotFittedError
from ..io import prepare_feature_matrix


@dataclass
class DeepSurvModel:
    """
    DeepSurv survival model based on pycox.
    """

    hidden_dim1: int = 32
    hidden_dim2: int = 16

    dropout: float = 0.1

    learning_rate: float = 1e-3

    batch_norm: bool = False

    batch_size: int = 32

    epochs: int = 80

    patience: int = 8

    validation_split: float = 0.1

    log_transform: bool = True

    max_features: int | None = 120

    min_expression_rate: float = 0.05

    random_state: int = 42

    model: CoxPH | None = None

    scaler: StandardScaler | None = None

    duration_col: str | None = None

    event_col: str | None = None

    feature_cols: list[str] | None = None

    history_: object | None = None

    n_parameters_: int | None = None

    @staticmethod
    def _build_network(
        num_features: int,
        hidden_dim1: int,
        hidden_dim2: int,
        dropout: float,
        batch_norm: bool,
    ) -> torch.nn.Sequential:
        layers: list[torch.nn.Module] = [
            torch.nn.Linear(num_features, hidden_dim1),
            torch.nn.ReLU(),
        ]
        if batch_norm:
            layers.append(torch.nn.BatchNorm1d(hidden_dim1))
        if dropout > 0:
            layers.append(torch.nn.Dropout(dropout))

        layers.extend(
            [
                torch.nn.Linear(hidden_dim1, hidden_dim2),
                torch.nn.ReLU(),
            ]
        )
        if batch_norm:
            layers.append(torch.nn.BatchNorm1d(hidden_dim2))
        if dropout > 0:
            layers.append(torch.nn.Dropout(dropout))

        layers.append(torch.nn.Linear(hidden_dim2, 1))
        return torch.nn.Sequential(*layers)

    @staticmethod
    def _count_parameters(net: torch.nn.Module) -> int:
        return int(sum(param.numel() for param in net.parameters() if param.requires_grad))

    def fit(
        self,
        data: pd.DataFrame,
        duration_col: str,
        event_col: str,
    ) -> "DeepSurvModel":

        torch.manual_seed(self.random_state)

        np.random.seed(self.random_state)

        model_data, feature_cols = prepare_feature_matrix(
            data,
            duration_col,
            event_col,
            log_transform=self.log_transform,
            max_features=self.max_features,
            min_expression_rate=self.min_expression_rate,
        )

        self.feature_cols = feature_cols

        self.duration_col = duration_col

        self.event_col = event_col

        X = model_data[feature_cols].values.astype("float32")
        y_time = model_data[duration_col].values
        y_event = model_data[event_col].values

        stratify = y_event if len(pd.unique(y_event)) > 1 and pd.Series(y_event).value_counts().min() >= 2 else None
        try:
            (
                X_train_raw,
                X_val_raw,
                y_time_train,
                y_time_val,
                y_event_train,
                y_event_val,
            ) = train_test_split(
                X,
                y_time,
                y_event,
                test_size=self.validation_split,
                random_state=self.random_state,
                stratify=stratify,
            )
        except ValueError:
            (
                X_train_raw,
                X_val_raw,
                y_time_train,
                y_time_val,
                y_event_train,
                y_event_val,
            ) = train_test_split(
                X,
                y_time,
                y_event,
                test_size=self.validation_split,
                random_state=self.random_state,
                stratify=None,
            )

        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train_raw)
        X_val = self.scaler.transform(X_val_raw)

        num_features = X.shape[1]
        net = self._build_network(
            num_features,
            self.hidden_dim1,
            self.hidden_dim2,
            self.dropout,
            self.batch_norm,
        )
        self.n_parameters_ = self._count_parameters(net)

        self.model = CoxPH(
            net,
            optim.Adam,
        )

        self.model.optimizer.set_lr(
            self.learning_rate
        )

        self.model.fit(
            X_train,
            (
                y_time_train,
                y_event_train,
            ),
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=[
                tt.callbacks.EarlyStopping(patience=self.patience)
            ],
            val_data=(
                X_val,
                (
                    y_time_val,
                    y_event_val,
                ),
            ),
            verbose=False,
        )

        self.history_ = getattr(self.model, "log", None)

        return self

    def predict_risk_score(
        self,
        data: pd.DataFrame,
    ) -> pd.Series:

        if self.model is None:
            raise ModelNotFittedError(
                "DeepSurvModel has not been fitted yet."
            )

        if self.scaler is None:
            raise ModelNotFittedError(
                "Scaler has not been fitted."
            )

        if self.feature_cols is None:
            raise ModelNotFittedError(
                "Feature columns are unavailable."
            )

        X = data[self.feature_cols].copy()
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.fillna(X.median(numeric_only=True))
        X = X.values.astype("float32")

        X = self.scaler.transform(X)

        risk_scores = self.model.predict(X)

        return pd.Series(
            risk_scores.flatten(),
            index=data.index,
            name="deep_risk_score",
        )

    def score(
        self,
        data: pd.DataFrame,
    ) -> float:

        if self.duration_col is None:
            raise ModelNotFittedError(
                "Model has not been fitted."
            )

        if self.event_col is None:
            raise ModelNotFittedError(
                "Model has not been fitted."
            )

        scoring_data = data.copy()
        scoring_data = scoring_data.dropna(subset=[self.duration_col, self.event_col]).reset_index(drop=True)
        risk_scores = self.predict_risk_score(scoring_data)

        return float(
            concordance_index(
                scoring_data[self.duration_col],
                -risk_scores,
                scoring_data[self.event_col],
            )
        )
