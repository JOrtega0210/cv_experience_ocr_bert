"""Métricas de regresión y de bandas compartidas por todos los modelos."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score

from src.utils import BAND_LABELS, clip_target, to_band


def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, dtype=float), clip_target(y_pred)
    err = np.abs(y_pred - y_true)
    bt, bp = to_band(y_true), to_band(y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "acc_1": float((err <= 1).mean()),
        "acc_2": float((err <= 2).mean()),
        "acc_3": float((err <= 3).mean()),
        "band_accuracy": float(accuracy_score(bt, bp)),
        "band_macro_f1": float(f1_score(bt, bp, labels=BAND_LABELS, average="macro", zero_division=0)),
        "r2": float(r2_score(y_true, y_pred)),
        "mean_error": float((y_pred - y_true).mean()),
    }
