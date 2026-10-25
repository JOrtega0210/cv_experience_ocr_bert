"""Baseline bonus: regla que suma la duración de los trabajos a partir de los rangos de fechas.

1. Extrae los rangos "YYYY.MM - YYYY.MM" (acepta cualquier guion y espacios, para funcionar también con el OCR crudo).
2. Descarta rangos imposibles (año de 1–3 dígitos que el OCR no pudo leer, fin anterior al inicio).
3. Une los rangos que se solapan y suma los meses.  años = meses / 12.
4. Si un CV no tiene ningún rango válido, predice la mediana del entrenamiento.

No aprende nada del texto: es la prueba honesta de si BERT hace algo más que aritmética de fechas.
"""

from __future__ import annotations

import re

import numpy as np

from src.experiments import run

RANGE_RE = re.compile(r"(?<![\d.])(\d{4})\.(\d{1,2})\s*[-‐-―−]\s*(\d{4})\.(\d{1,2})(?![\d.])")


def job_intervals(text: str) -> list[tuple[int, int]]:
    """Rangos válidos como (mes_inicio, mes_fin) en meses absolutos."""
    out = []
    for y1, m1, y2, m2 in RANGE_RE.findall(text):
        start, end = int(y1) * 12 + int(m1) - 1, int(y2) * 12 + int(m2) - 1
        if 1 <= int(m1) <= 12 and 1 <= int(m2) <= 12 and end >= start:
            out.append((start, end))
    return out


def merged_months(intervals: list[tuple[int, int]]) -> int:
    total, current = 0, None
    for start, end in sorted(intervals):
        if current is None or start > current[1]:
            if current:
                total += current[1] - current[0]
            current = [start, end]
        else:
            current[1] = max(current[1], end)
    return total + (current[1] - current[0] if current else 0)


def predict_years(text: str, fallback: float) -> float:
    intervals = job_intervals(text)
    return merged_months(intervals) / 12 if intervals else fallback


class DateRulePredictor:
    def __init__(self, fallback: float):
        self.fallback = fallback

    def __call__(self, texts):
        return np.array([predict_years(t, self.fallback) for t in texts])


def fit(train_texts, y_train, val_texts, y_val):
    fallback = float(np.median(y_train))
    n_without = sum(not job_intervals(t) for t in train_texts)
    return DateRulePredictor(fallback), {"fallback_median": round(fallback, 4), "train_cvs_without_dates": n_without}


if __name__ == "__main__":
    from src.baseline_rules import fit as module_fit  # ver nota en baseline_median

    run("date_rule", module_fit)
