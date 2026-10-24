"""Baseline 1: TF-IDF + Ridge.

- n-gramas de palabras 1–2 (y opcionalmente n-gramas de caracteres 3–5 dentro de palabras), sublinear tf.
- Números y años se CONSERVAN como tokens: el patrón de tokens acepta "2019" y "07" por separado, de modo que el
  modelo puede asociar años antiguos a más experiencia. Lo que no puede hacer es restar fechas: cada año es solo
  una columna más. El escenario `no_digits` (pregunta 3) quita todos los dígitos para medir cuánto aportan.
- Se elige en validación la combinación de features y la fuerza de regularización alpha.
"""

from __future__ import annotations

import re

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

from src.experiments import SETUPS, run

ALPHAS = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
TOKEN_PATTERN = r"(?u)\b\w+\b"  # incluye tokens de 1 carácter y números ("07", "2019")


def lowercase(text: str) -> str:
    return text.lower()


def lowercase_no_digits(text: str) -> str:
    return re.sub(r"\d", " ", text.lower())


class TfidfPredictor:
    def __init__(self, vectorizers, model):
        self.vectorizers, self.model = vectorizers, model

    def __call__(self, texts):
        return self.model.predict(hstack([v.transform(texts) for v in self.vectorizers]).tocsr())


def _features(char: bool, keep_digits: bool):
    # Funciones con nombre (no lambdas) para que el modelo se pueda guardar con joblib
    pre = lowercase if keep_digits else lowercase_no_digits
    vecs = [TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=TOKEN_PATTERN,
                            preprocessor=pre)]
    if char:
        vecs.append(TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True,
                                    preprocessor=pre))
    return vecs


def make_fit(keep_digits: bool = True):
    def fit(train_texts, y_train, val_texts, y_val):
        best = None
        for char in [False, True]:
            vecs = _features(char, keep_digits)
            x_train = hstack([v.fit_transform(train_texts) for v in vecs]).tocsr()
            x_val = hstack([v.transform(val_texts) for v in vecs]).tocsr()
            for alpha in ALPHAS:
                model = Ridge(alpha=alpha).fit(x_train, y_train)
                mae = mean_absolute_error(y_val, np.clip(model.predict(x_val), 0, 20))
                if best is None or mae < best[0]:
                    best = (mae, char, alpha, vecs, model)
        mae, char, alpha, vecs, model = best
        n_features = sum(len(v.vocabulary_) for v in vecs)
        return TfidfPredictor(vecs, model), {"char_ngrams": char, "alpha": alpha, "keep_digits": keep_digits,
                         "n_features": n_features, "val_mae": round(mae, 4)}

    return fit


if __name__ == "__main__":
    from src.baseline_tfidf import make_fit as module_make_fit  # ver nota en baseline_median

    run("tfidf_ridge", module_make_fit(keep_digits=True))
    # Pregunta 3: mismo modelo sin dígitos (se guarda aparte para no pisar las predicciones principales)
    run("tfidf_ridge_no_digits", module_make_fit(keep_digits=False), setups={"main": SETUPS["main"]}, reference=False,
        save_model=False)
