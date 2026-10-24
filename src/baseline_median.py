"""Baseline 0: predice siempre la mediana de años del set de entrenamiento.

Es el piso: un modelo que no le gana a la mediana no aprendió nada del texto.
"""

from __future__ import annotations

import numpy as np

from src.experiments import run


class MedianPredictor:
    def __init__(self, median: float):
        self.median = median

    def __call__(self, texts):
        return np.full(len(texts), self.median)


def fit(train_texts, y_train, val_texts, y_val):
    median = float(np.median(y_train))
    return MedianPredictor(median), {"median": round(median, 4)}


if __name__ == "__main__":
    # importar por nombre: así joblib guarda src.baseline_median.MedianPredictor y no __main__.MedianPredictor
    from src.baseline_median import fit as module_fit

    run("median", module_fit)
