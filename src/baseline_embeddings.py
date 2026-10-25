"""Baseline 2: embeddings de oraciones congelados + Ridge.

- Encoders candidatos (no se ajustan): all-MiniLM-L6-v2 (límite 256 tokens) y bge-small-en-v1.5 (512 tokens).
  Se elige en validación el encoder y alpha de Ridge.
- CVs más largos que el límite del encoder: se parten por líneas en bloques que entren en el límite, se codifica
  cada bloque y se promedian los vectores (ponderados por número de tokens). En PopResume ningún CV limpio supera
  los 177 tokens, así que el chunking no se activa con este dataset, pero sí con CVs subidos a la app.
- Los vectores se normalizan (L2), como los usa sentence-transformers para similitud.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

from src.experiments import run
from src.utils import device, get_logger

log = get_logger("embeddings")

ENCODERS = ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"]
ALPHAS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3]

_models: dict = {}
_cache: dict[str, dict[str, np.ndarray]] = {}


def _model(name: str):
    if name not in _models:
        from sentence_transformers import SentenceTransformer

        _models[name] = SentenceTransformer(name, device=device())
    return _models[name]


def chunk_text(text: str, tokenizer, max_tokens: int) -> list[str]:
    """Agrupa líneas consecutivas en bloques de como máximo `max_tokens` tokens."""
    chunks, current, current_len = [], [], 0
    for line in text.split("\n"):
        n = len(tokenizer.tokenize(line)) + 1
        if current and current_len + n > max_tokens:
            chunks.append("\n".join(current))
            current, current_len = [], 0
        current.append(line)
        current_len += n
    if current:
        chunks.append("\n".join(current))
    return chunks


def encode(texts: list[str], name: str) -> np.ndarray:
    cache = _cache.setdefault(name, {})
    todo = list(dict.fromkeys(t for t in texts if t not in cache))
    if todo:
        model = _model(name)
        limit = model.max_seq_length - 2  # [CLS] y [SEP]
        pieces, owner, weights = [], [], []
        for i, text in enumerate(todo):
            for chunk in chunk_text(text, model.tokenizer, limit):
                pieces.append(chunk)
                owner.append(i)
                weights.append(len(model.tokenizer.tokenize(chunk)))
        n_chunked = len(pieces) - len(todo)
        log.info("%s: codificando %d textos (%d bloques extra por chunking)", name.split("/")[-1], len(todo), n_chunked)
        vectors = model.encode(pieces, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        owner, weights = np.array(owner), np.array(weights, dtype=float)
        for i, text in enumerate(todo):
            mask = owner == i
            v = np.average(vectors[mask], axis=0, weights=weights[mask])
            cache[text] = v / np.linalg.norm(v)
    return np.vstack([cache[t] for t in texts])


class EmbeddingPredictor:
    """Guarda solo el nombre del encoder (se vuelve a descargar de Hugging Face) y la Ridge entrenada."""

    def __init__(self, encoder: str, model: Ridge):
        self.encoder, self.model = encoder, model

    def __call__(self, texts):
        return self.model.predict(encode(texts, self.encoder))


def fit(train_texts, y_train, val_texts, y_val):
    best = None
    for name in ENCODERS:
        x_train, x_val = encode(train_texts, name), encode(val_texts, name)
        for alpha in ALPHAS:
            model = Ridge(alpha=alpha).fit(x_train, y_train)
            mae = mean_absolute_error(y_val, np.clip(model.predict(x_val), 0, 20))
            if best is None or mae < best[0]:
                best = (mae, name, alpha, model)
    mae, name, alpha, model = best
    return EmbeddingPredictor(name, model), {
        "encoder": name, "alpha": alpha, "dim": int(model.coef_.shape[0]), "val_mae": round(mae, 4)}


if __name__ == "__main__":
    from src.baseline_embeddings import fit as module_fit  # ver nota en baseline_median

    run("embeddings_ridge", module_fit)
