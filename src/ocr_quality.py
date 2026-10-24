"""Calidad del OCR contra el texto de referencia de PopResume: CER, WER y lectura de fechas.

El texto de referencia (`data/raw/reference_text.parquet`) se usa SOLO aquí, para medir el error del OCR;
nunca es feature de un modelo.

Normalización (igual para OCR y referencia, antes de CER/WER):
    NFKC, minúsculas, todos los guiones (– — − ‐) -> "-", viñetas (• ·) -> "-", espacios alrededor de "|"
    unificados y todos los espacios/saltos de línea colapsados a un espacio. Así no se penalizan diferencias
    de maquetación (sangrías, líneas en blanco) que no cambian el contenido.

Fechas:
    - year_recall: de los años (19xx/20xx) que aparecen en la referencia, qué fracción aparece en el OCR
      (conteo multiconjunto: si la referencia tiene 2016 dos veces, el OCR debe tenerlo dos veces).
    - date_recall: lo mismo con el token de fecha completo, p. ej. "2016.05".
    - all_years_ok: el CV tiene todos sus años bien leídos.

Salidas:
    output/tables/ocr_quality_per_cv.csv
    output/tables/ocr_quality_by_occupation.csv
    output/tables/ocr_model_comparison.csv      (mobile vs server en la muestra de 50 CVs, si existe)
    output/tables/ocr_year_errors.csv           (años de la referencia que el OCR no reprodujo, con contexto)
    output/metrics/ocr_quality.json
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from pathlib import Path

import jiwer
import numpy as np
import pandas as pd

from src.utils import DATA_DIR, METRICS_DIR, OCR_DIR, RAW_DIR, TABLES_DIR, get_logger, load_json, save_json

log = get_logger("ocr_quality")

DASHES = "‐‑‒–—―−"
YEAR_RE = re.compile(r"(?<!\d)(?:19[5-9]\d|20[0-3]\d)(?!\d)")
DATE_RE = re.compile(r"(?<!\d)(?:19[5-9]\d|20[0-3]\d)[./-](?:0?[1-9]|1[0-2])(?!\d)")


def normalize_for_eval(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = re.sub(f"[{DASHES}]", "-", text)
    text = re.sub(r"[•·●▪]", "-", text)
    text = re.sub(r"\s*\|\s*", " | ", text)
    return re.sub(r"\s+", " ", text).strip()


def _multiset_recall(ref: Counter, hyp: Counter) -> float:
    total = sum(ref.values())
    if total == 0:
        return np.nan
    return sum(min(c, hyp[k]) for k, c in ref.items()) / total


def years(text: str) -> Counter:
    return Counter(YEAR_RE.findall(normalize_for_eval(text)))


def dates(text: str) -> Counter:
    # "2016-05" y "2016.05" se consideran el mismo token
    return Counter(d.replace("-", ".").replace("/", ".") for d in DATE_RE.findall(normalize_for_eval(text)))


def quality_row(hyp: str, ref: str) -> dict:
    h, r = normalize_for_eval(hyp), normalize_for_eval(ref)
    ref_years, hyp_years = years(ref), years(hyp)
    return {
        "cer": jiwer.cer(r, h) if r else np.nan,
        "wer": jiwer.wer(r, h) if r else np.nan,
        "n_ref_years": sum(ref_years.values()),
        "year_recall": _multiset_recall(ref_years, hyp_years),
        "date_recall": _multiset_recall(dates(ref), dates(hyp)),
        "all_years_ok": bool(_multiset_recall(ref_years, hyp_years) == 1.0) if ref_years else np.nan,
    }


def evaluate(hyp: pd.DataFrame, ref: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """hyp y ref tienen columnas id + text_col / text. Devuelve una fila de métricas por CV."""
    df = hyp[["id", text_col]].merge(ref[["id", "text"]].rename(columns={"text": "ref_text"}), on="id")
    metrics = pd.DataFrame([quality_row(h, r) for h, r in zip(df[text_col], df["ref_text"])])
    return pd.concat([df[["id"]].reset_index(drop=True), metrics], axis=1)


def load_ocr_texts(ocr_dir: Path = OCR_DIR) -> pd.DataFrame:
    rows = []
    for path in sorted(ocr_dir.glob("*.json")):
        r = load_json(path)
        rows.append({"id": r["id"], "text": r["text"], "mean_score": r["mean_score"], "n_lines": r["n_lines"]})
    return pd.DataFrame(rows)


def year_errors(hyp: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Años de la referencia que faltan en el OCR, con la línea de la referencia donde aparecían."""
    df = hyp.merge(ref.rename(columns={"text": "ref_text"}), on="id")
    rows = []
    for cv_id, h, r in zip(df["id"], df["text"], df["ref_text"]):
        missing = years(r) - years(h)
        for year in missing:
            ref_line = next((l for l in r.splitlines() if year in l), "")
            hyp_line = next((l for l in h.splitlines() if l[:15].lower() == ref_line.strip()[:15].lower()), "")
            rows.append({"id": cv_id, "year": year, "ref_line": ref_line.strip(), "ocr_line": hyp_line.strip()})
    return pd.DataFrame(rows, columns=["id", "year", "ref_line", "ocr_line"])


def summarize(per_cv: pd.DataFrame) -> dict:
    def stats(s: pd.Series) -> dict:
        return {"mean": round(float(s.mean()), 4), "median": round(float(s.median()), 4),
                "p90": round(float(s.quantile(0.9)), 4), "max": round(float(s.max()), 4)}

    return {
        "n_cvs": int(len(per_cv)),
        "cer": stats(per_cv["cer"]),
        "wer": stats(per_cv["wer"]),
        "year_recall_micro": round(float((per_cv["year_recall"] * per_cv["n_ref_years"]).sum()
                                         / per_cv["n_ref_years"].sum()), 4),
        "date_recall_mean": round(float(per_cv["date_recall"].mean()), 4),
        "share_cvs_all_years_ok": round(float(per_cv["all_years_ok"].mean()), 4),
    }


def main() -> None:
    ref = pd.read_parquet(RAW_DIR / "reference_text.parquet")
    meta = pd.read_csv(DATA_DIR / "sampled_ids.csv", dtype={"id": str})[["id", "occupation"]]

    ocr = load_ocr_texts()
    log.info("%d textos OCR, %d referencias", len(ocr), len(ref))
    per_cv = evaluate(ocr, ref).merge(meta, on="id").merge(ocr[["id", "mean_score", "n_lines"]], on="id")
    per_cv.round(4).to_csv(TABLES_DIR / "ocr_quality_per_cv.csv", index=False)

    by_occ = (
        per_cv.groupby("occupation")
        .agg(n=("id", "size"), cer_mean=("cer", "mean"), cer_median=("cer", "median"), wer_mean=("wer", "mean"),
             wer_median=("wer", "median"), year_recall=("year_recall", "mean"),
             date_recall=("date_recall", "mean"), share_all_years_ok=("all_years_ok", "mean"),
             mean_confidence=("mean_score", "mean"))
        .reset_index()
    )
    by_occ.round(4).to_csv(TABLES_DIR / "ocr_quality_by_occupation.csv", index=False)

    errors = year_errors(ocr, ref)
    errors.to_csv(TABLES_DIR / "ocr_year_errors.csv", index=False)

    summary = summarize(per_cv)
    summary["n_year_errors"] = int(len(errors))
    summary["corr_confidence_vs_cer"] = round(float(per_cv[["mean_score", "cer"]].corr().iloc[0, 1]), 4)

    # Comparación mobile vs server en la misma muestra (generada en el notebook de Colab)
    comparison = []
    for name in ["ocr_mobile_sample", "ocr_server_sample"]:
        sample_dir = DATA_DIR / name
        if not sample_dir.exists():
            continue
        sample = evaluate(load_ocr_texts(sample_dir), ref)
        ocr_summary_path = METRICS_DIR / f"ocr_summary_{name}.json"
        seconds = load_json(ocr_summary_path)["mean_seconds_per_cv"] if ocr_summary_path.exists() else np.nan
        first = load_json(next(sample_dir.glob("*.json")))
        comparison.append({"setup": name, "rec_model": first["rec_model"], "det_model": first["det_model"],
                           "n": len(sample), **{k: v for k, v in summarize(sample).items() if k != "n_cvs"},
                           "seconds_per_cv_gpu": seconds})
    if comparison:
        comp = pd.json_normalize(comparison)
        comp.to_csv(TABLES_DIR / "ocr_model_comparison.csv", index=False)
        summary["model_comparison"] = comp.to_dict(orient="records")

    save_json(summary, METRICS_DIR / "ocr_quality.json")
    log.info("CER medio %.4f | WER medio %.4f | años bien leídos %.4f | CVs con todos los años ok %.4f",
             summary["cer"]["mean"], summary["wer"]["mean"], summary["year_recall_micro"],
             summary["share_cvs_all_years_ok"])


if __name__ == "__main__":
    main()
