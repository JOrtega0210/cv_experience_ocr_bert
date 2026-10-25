"""Corre un modelo en todos los escenarios del proyecto con el mismo split y guarda predicciones y tiempos.

Escenarios (`setup`):
    main             entrena con `text` (OCR limpio) y ajusta hiperparámetros en val; predice val y test.
                     El mismo modelo predice además el texto de referencia del test (split = test_reference),
                     para medir cuánto cuesta el ruido del OCR (pregunta 2).
    train_raw        entrena y evalúa con el OCR crudo            } pregunta 3: sensibilidad a la limpieza
    train_clean_conf entrena y evalúa sin líneas de baja confianza }

Cada modelo implementa `fit(train_texts, y_train, val_texts, y_val) -> (predict, info)`, donde `predict(texts)`
devuelve años e `info` describe los hiperparámetros elegidos en validación.

Salidas:
    output/predictions/<model>.csv        model, setup, split, id, y_true, y_pred (formato largo)
    output/metrics/<model>.json           hiperparámetros, tiempos y métricas de val/test por escenario
    models/baselines/<model>.joblib       predictor del escenario main (lo usa la app; no se sube a GitHub)
"""

from __future__ import annotations

import time
from typing import Callable

import joblib
import numpy as np
import pandas as pd

from src.evaluation import regression_metrics
from src.split import load_splits
from src.utils import (
    BASELINES_DIR, METRICS_DIR, OUTPUT_DIR, PROCESSED_DIR, TARGET, clip_target, get_logger, save_json, set_seed,
)

log = get_logger("experiments")

PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
SETUPS = {"main": "text", "train_raw": "text_raw", "train_clean_conf": "text_clean_conf"}

FitFn = Callable[[list[str], np.ndarray, list[str], np.ndarray], tuple[Callable[[list[str]], np.ndarray], dict]]


def load_data() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    texts = pd.read_csv(PROCESSED_DIR / "cvs.csv", dtype={"id": str}).drop(columns=["occupation", TARGET])
    splits = {name: df.merge(texts, on="id", how="left") for name, df in load_splits().items()}
    reference = pd.read_csv(PROCESSED_DIR / "reference.csv", dtype={"id": str})
    return splits, reference


def run(model_name: str, fit: FitFn, setups: dict[str, str] | None = None, reference: bool = True,
        save_model: bool = True) -> dict:
    set_seed()
    splits, ref = load_data()
    setups = setups or SETUPS
    rows, report = [], {"model": model_name, "setups": {}}

    for setup, col in setups.items():
        train, val, test = splits["train"], splits["val"], splits["test"]
        start = time.perf_counter()
        predict, info = fit(train[col].tolist(), train[TARGET].to_numpy(), val[col].tolist(), val[TARGET].to_numpy())
        train_seconds = time.perf_counter() - start

        start = time.perf_counter()
        test_pred = clip_target(predict(test[col].tolist()))
        infer_ms = 1000 * (time.perf_counter() - start) / len(test)
        val_pred = clip_target(predict(val[col].tolist()))

        if save_model and setup == "main":
            BASELINES_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(predict, BASELINES_DIR / f"{model_name}.joblib")

        evals = {"val": (val, val_pred), "test": (test, test_pred)}
        if reference and setup == "main":
            test_ref = test[["id", TARGET]].merge(ref, on="id")
            evals["test_reference"] = (test_ref, clip_target(predict(test_ref["text"].tolist())))

        report["setups"][setup] = {
            "text_column": col, "info": info, "train_seconds": round(train_seconds, 2),
            "inference_ms_per_cv": round(infer_ms, 3),
            "metrics": {split: regression_metrics(df[TARGET], pred) for split, (df, pred) in evals.items()},
        }
        for split, (df, pred) in evals.items():
            rows.append(pd.DataFrame({"model": model_name, "setup": setup, "split": split, "id": df["id"],
                                      "y_true": df[TARGET], "y_pred": np.round(pred, 4)}))
        m = report["setups"][setup]["metrics"]
        log.info("%s | %s | val MAE %.3f | test MAE %.3f | train %.1f s | %.2f ms/CV",
                 model_name, setup, m["val"]["mae"], m["test"]["mae"], train_seconds, infer_ms)

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(rows).to_csv(PREDICTIONS_DIR / f"{model_name}.csv", index=False)
    save_json(report, METRICS_DIR / f"{model_name}.json")
    return report
