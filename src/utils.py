"""Semillas, rutas, constantes del proyecto y funciones auxiliares compartidas."""

from __future__ import annotations

import json
import logging
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
IMAGES_DIR = RAW_DIR / "images"
OCR_DIR = DATA_DIR / "ocr"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

MODELS_DIR = ROOT / "models"
BASELINES_DIR = MODELS_DIR / "baselines"
BERT_DIR = MODELS_DIR / "bert"

OUTPUT_DIR = ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
METRICS_DIR = OUTPUT_DIR / "metrics"

ALL_DIRS = [
    IMAGES_DIR, OCR_DIR, PROCESSED_DIR, SPLITS_DIR,
    BASELINES_DIR, BERT_DIR, FIGURES_DIR, TABLES_DIR, METRICS_DIR,
]

# ---------------------------------------------------------------------------
# Constantes del problema
# ---------------------------------------------------------------------------
SEED = 42

HF_DATASET = "sumin-yu/PopResume"

# Ocupación (prefijo de los shards en data/images/) -> CVs a muestrear
OCCUPATIONS: dict[str, int] = {
    "registered_nurses": 800,
    "elementary_middle_school_teachers": 800,
    "software_developers": 800,
    "accountants_auditors": 800,
    "construction_laborers": 800,
}

TARGET = "exp_year"
TARGET_CAP = 20  # exp_year > 20 se lee como "20 o más"

# Bandas de experiencia para la evaluación como clasificación
BANDS: list[tuple[int, int]] = [(0, 2), (3, 5), (6, 10), (11, TARGET_CAP)]
BAND_LABELS = [f"{lo}-{hi}" for lo, hi in BANDS]

# Columnas que nunca pueden usarse como features
FORBIDDEN_FEATURES = ["age", "sex", "race", "name", "state", "region"]
FAIRNESS_COLUMNS = ["sex", "race"]  # solo para la pregunta 4

SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_seed(seed: int = SEED) -> None:
    """Fija las semillas de random, numpy y (si están instalados) torch y paddle."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    try:
        import paddle

        paddle.seed(seed)
    except ImportError:
        pass


def ensure_dirs() -> None:
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def clip_target(y, cap: int = TARGET_CAP):
    """Recorta valores (verdaderos o predichos) al rango [0, cap]."""
    return np.clip(np.asarray(y, dtype=float), 0, cap)


def to_band(y, bands: list[tuple[int, int]] = BANDS) -> np.ndarray:
    """Convierte años (posiblemente continuos) a la etiqueta de banda.

    Las predicciones se redondean al entero más cercano antes de asignar la banda,
    de modo que 2.4 cae en 0-2 y 2.6 en 3-5.
    """
    y = np.rint(clip_target(y)).astype(int)
    labels = np.empty(len(y), dtype=object)
    for (lo, hi), label in zip(bands, BAND_LABELS):
        labels[(y >= lo) & (y <= hi)] = label
    return labels


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


@contextmanager
def timer(label: str, logger: logging.Logger | None = None) -> Iterator[dict]:
    """Mide el tiempo de un bloque: `with timer("ocr") as t: ...; t["seconds"]`."""
    out: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield out
    finally:
        out["seconds"] = time.perf_counter() - start
        (logger or get_logger("timer")).info("%s: %.2f s", label, out["seconds"])


def device() -> str:
    """'cuda' si hay GPU disponible para torch, si no 'cpu'."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


if __name__ == "__main__":
    ensure_dirs()
    set_seed()
    print("ROOT:", ROOT)
    print("Device:", device())
    print("Bands of [0, 2.6, 5, 7, 25]:", to_band([0, 2.6, 5, 7, 25]))
