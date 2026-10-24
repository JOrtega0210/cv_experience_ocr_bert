"""Limpieza del texto OCR, eliminación de datos personales y enmascarado de respuestas directas.

Se generan tres variantes del texto por CV (pregunta 3, sensibilidad a la limpieza del OCR). Las tres pasan por
la eliminación de datos personales y el enmascarado, que son obligatorios:

    text_raw         OCR en orden de lectura, sin más cambios
    text_clean       + limpieza (texto principal que usan todos los modelos)
    text_clean_conf  + limpieza, descartando antes las líneas con confianza < CONF_THRESHOLD

Limpieza (`clean_text`):
    - NFKC; guiones tipográficos -> "-"; viñetas "•" y "-Texto" -> "- Texto"
    - une palabras cortadas con guion al final de línea y líneas que continúan la anterior (empiezan en minúscula)
    - rangos de fechas normalizados a "YYYY.MM - YYYY.MM"
    - reparación de años: PP-OCRv5 server a veces omite un dígito repetido (2008 -> "208", 2022 -> "222").
      Si un año de 1–3 dígitos tiene UNA sola reconstrucción válida (insertando un 0 o anteponiendo un 2) dentro de
      [MIN_YEAR, MAX_YEAR] y coherente con el rango (fin >= inicio), se repara; si es ambiguo se deja igual.

Datos personales (`remove_pii`): nombre -> [NAME], email -> [EMAIL], teléfono -> [PHONE], "ST, USA" -> [LOCATION].
El nombre se detecta como la primera línea en mayúsculas antes de la línea de contacto; además, si se conoce
(metadata), se borra cualquier aparición del nombre y apellido. El nombre NO se usa como feature.

Enmascarado (`mask_direct_answers`): frases que dan la respuesta ("12 years of experience", "age 34",
"born in 1985", "date of birth: ..."). Las fechas de trabajo y de graduación se mantienen: son la evidencia.

Salidas:
    data/processed/cvs.csv          id, occupation, exp_year, text_raw, text_clean, text_clean_conf, text
    data/processed/reference.csv    id, text (texto de referencia con la misma limpieza; solo para la pregunta 2)
    output/metrics/preprocessing_summary.json
    output/tables/cleaning_effect.csv
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

from src.ocr_extraction import reading_order
from src.ocr_quality import DASHES, evaluate
from src.utils import (
    DATA_DIR, METRICS_DIR, OCR_DIR, PROCESSED_DIR, RAW_DIR, TABLES_DIR, TARGET, get_logger, load_json, save_json,
)

log = get_logger("preprocessing")

CONF_THRESHOLD = 0.95  # ≈ percentil 1 de la confianza por línea en los 4,000 CVs
MIN_YEAR, MAX_YEAR = 1950, 2025

# ---------------------------------------------------------------------------
# Datos personales
# ---------------------------------------------------------------------------
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
# Segunda alternativa: con "+" inicial se toleran grupos de 2–4 dígitos (el OCR a veces omite un dígito)
PHONE_RE = re.compile(r"(?<![\w.])\+?1?[\s-]?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?![\w.])"
                      r"|\+\d{1,2}[\s-]?\d{2,4}[\s.-]\d{2,4}[\s.-]\d{2,4}(?![\w.])")
LOCATION_RE = re.compile(r"\b[A-Z]{2}\s*,\s*USA\b")

# ---------------------------------------------------------------------------
# Respuestas directas
# ---------------------------------------------------------------------------
NUM = r"\d{1,2}(?:\.\d)?\+?"
MASK_PATTERNS = {
    "total_experience": re.compile(
        rf"(?:\b(?:over|more than|nearly|almost|about|around|approximately)\s+)?\b{NUM}\s*(?:years?|yrs?)\.?"
        r"\s*(?:of\s+)?(?:[a-z-]+\s+){0,3}?experience\b", re.I),
    "experience_of_n_years": re.compile(rf"\bexperience\s*(?:of|:)\s*{NUM}\s*(?:years?|yrs?)\b", re.I),
    "age": re.compile(r"\bage[d]?\s*:?\s*\d{2}\b|\b\d{2}\s*(?:years?|yrs?)\s*old\b|\b\d{2}\s*y/?o\b", re.I),
    "date_of_birth": re.compile(r"\b(?:date of birth|d\.?o\.?b\.?|born(?:\s+(?:on|in))?)\b\s*:?\s*[^\n|;]*", re.I),
}
MASK_TOKEN = "[MASK]"


def remove_pii(text: str, names: list[str] | None = None) -> tuple[str, dict]:
    counts = {"name": 0, "email": 0, "phone": 0, "location": 0}
    lines = text.split("\n")
    # Nombre: primera línea en mayúsculas antes de la línea de contacto (email/teléfono)
    for i, line in enumerate(lines[:3]):
        if EMAIL_RE.search(line) or PHONE_RE.search(line):
            break
        if line.strip() and line.strip().upper() == line.strip() and re.search(r"[A-Z]", line):
            lines[i], counts["name"] = "[NAME]", 1
            break
    text = "\n".join(lines)
    # Email antes que el nombre: si no, "sarah.martin@..." queda como "[NAME].[NAME]@..." y ya no se reconoce
    for key, regex, token in [("email", EMAIL_RE, "[EMAIL]"), ("phone", PHONE_RE, "[PHONE]"),
                              ("location", LOCATION_RE, "[LOCATION]")]:
        text, counts[key] = regex.subn(token, text)
    for name in filter(None, names or []):
        text, n = re.subn(rf"\b{re.escape(name)}\b", "[NAME]", text, flags=re.I)
        counts["name"] += n
    return text, counts


def mask_direct_answers(text: str) -> tuple[str, dict]:
    counts = {}
    for key, regex in MASK_PATTERNS.items():
        text, counts[key] = regex.subn(MASK_TOKEN, text)
    return text, counts


# ---------------------------------------------------------------------------
# Limpieza
# ---------------------------------------------------------------------------
def _year_candidates(token: str) -> set[int]:
    if len(token) == 4:
        return {int(token)}
    options = {token[:i] + "0" + token[i:] for i in range(len(token) + 1)} | {"2" + token}
    return {int(o) for o in options if len(o) == 4 and MIN_YEAR <= int(o) <= MAX_YEAR}


def _repair_year(token: str, not_before: tuple[int, int] | None = None, month: int = 12) -> str:
    candidates = _year_candidates(token)
    if not_before:
        candidates = {c for c in candidates if (c, month) >= not_before}
    return str(candidates.pop()) if len(candidates) == 1 else token


RANGE_RE = re.compile(r"(?<![\d.])(\d{1,4})\.(\d{1,2})\s*-\s*(\d{1,4})\.(\d{1,2})(?![\d.])")
EDU_YEAR_RE = re.compile(r"(,\s*)(\d{1,4})\s*$", re.M)


def repair_dates(text: str) -> tuple[str, int]:
    repaired = 0

    def fix_range(m: re.Match) -> str:
        nonlocal repaired
        y1, m1, y2, m2 = m.groups()
        new_y1 = _repair_year(y1)
        start = (int(new_y1), int(m1)) if len(new_y1) == 4 else None
        new_y2 = _repair_year(y2, not_before=start, month=int(m2))
        repaired += (new_y1 != y1) + (new_y2 != y2)
        return f"{new_y1}.{int(m1):02d} - {new_y2}.{int(m2):02d}"

    def fix_edu(m: re.Match) -> str:
        nonlocal repaired
        new = _repair_year(m.group(2))
        repaired += new != m.group(2)
        return m.group(1) + new

    text = RANGE_RE.sub(fix_range, text)
    text = EDU_YEAR_RE.sub(fix_edu, text)
    return text, repaired


def clean_text(text: str) -> tuple[str, int]:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(f"[{DASHES}]", "-", text)
    text = re.sub(r"[ \t]*\|[ \t]*", " | ", text)
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    merged: list[str] = []
    for line in lines:
        line = re.sub(r"^[•·●▪]\s*", "- ", line)
        line = re.sub(r"^-(?=\S)", "- ", line)
        if merged and re.search(r"[a-z]-$", merged[-1]) and re.match(r"[a-z]", line):
            merged[-1] = merged[-1][:-1] + line              # palabra cortada con guion
        elif merged and re.match(r"[a-z]", line) and not merged[-1].isupper():
            merged[-1] = f"{merged[-1]} {line}"              # línea que continúa la anterior
        else:
            merged.append(line)
    text = "\n".join(merged)
    text, repaired = repair_dates(text)
    return re.sub(r"[ \t]+", " ", text).strip(), repaired


# ---------------------------------------------------------------------------
# Pipeline por CV
# ---------------------------------------------------------------------------
def text_from_record(record: dict, conf_threshold: float | None = None) -> str:
    """Rearma el texto en orden de lectura, opcionalmente sin las líneas de baja confianza."""
    lines = record["lines"]
    if conf_threshold is not None:
        lines = [l for l in lines if l["score"] >= conf_threshold]
    width = record.get("image_size", [max((l["box"][2] for l in lines), default=0)])[0]
    rows, _ = reading_order(lines, width)
    return "\n".join(" ".join(lines[i]["text"] for i in row) for row in rows)


def finalize(text: str, names: list[str] | None = None) -> tuple[str, dict]:
    """Pasos obligatorios para cualquier texto que entre a un modelo: datos personales + enmascarado."""
    text, pii = remove_pii(text, names)
    text, masked = mask_direct_answers(text)
    return text, {"pii": pii, "masked": masked}


def preprocess_record(record: dict, names: list[str] | None = None,
                      conf_threshold: float = CONF_THRESHOLD) -> dict:
    raw = record["text"]
    clean, repaired = clean_text(raw)
    clean_conf, _ = clean_text(text_from_record(record, conf_threshold))
    n_dropped = sum(l["score"] < conf_threshold for l in record["lines"])

    out = {"id": record["id"], "n_years_repaired": repaired, "n_lines_dropped_conf": n_dropped}
    for name, text in [("text_raw", raw), ("text_clean", clean), ("text_clean_conf", clean_conf)]:
        out[name], info = finalize(text, names)
        if name == "text_clean":
            out.update({f"pii_{k}": v for k, v in info["pii"].items()})
            out.update({f"mask_{k}": v for k, v in info["masked"].items()})
    out["text"] = out["text_clean"]
    return out


def main() -> None:
    meta = pd.read_csv(RAW_DIR / "metadata.csv", dtype={"id": str})
    names = {r.id: [r.first_name, r.last_name, f"{r.first_name}{r.last_name}"] for r in meta.itertuples()}

    rows = [preprocess_record(load_json(p), names.get(p.stem)) for p in sorted(OCR_DIR.glob("*.json"))]
    df = pd.DataFrame(rows).merge(meta[["id", "occupation", TARGET]], on="id")

    mask_cols = [c for c in df.columns if c.startswith("mask_")]
    df["n_masked"] = df[mask_cols].sum(axis=1)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["id", "occupation", TARGET, "text_raw", "text_clean", "text_clean_conf", "text"]
    df[cols].to_csv(PROCESSED_DIR / "cvs.csv", index=False)

    # Referencia con el mismo tratamiento (solo para la pregunta 2: modelos evaluados con texto sin errores de OCR)
    ref = pd.read_parquet(RAW_DIR / "reference_text.parquet")
    ref["text"] = [finalize(clean_text(t)[0], names.get(i))[0] for i, t in zip(ref["id"], ref["text"])]
    ref.to_csv(PROCESSED_DIR / "reference.csv", index=False)

    # Efecto de cada variante sobre la lectura de fechas (contra la referencia original)
    ref_raw = pd.read_parquet(RAW_DIR / "reference_text.parquet")
    effect = []
    for variant in ["text_raw", "text_clean", "text_clean_conf"]:
        q = evaluate(df.rename(columns={variant: "hyp"})[["id", "hyp"]], ref_raw, text_col="hyp")
        effect.append({"variant": variant,
                       "year_recall_micro": round(float((q.year_recall * q.n_ref_years).sum() / q.n_ref_years.sum()), 4),
                       "date_recall_mean": round(float(q.date_recall.mean()), 4),
                       "share_cvs_all_years_ok": round(float(q.all_years_ok.mean()), 4)})
    effect = pd.DataFrame(effect)
    effect.to_csv(TABLES_DIR / "cleaning_effect.csv", index=False)

    summary = {
        "n_cvs": int(len(df)),
        "conf_threshold": CONF_THRESHOLD,
        "lines_dropped_conf_total": int(df["n_lines_dropped_conf"].sum()),
        "cvs_with_lines_dropped_conf": int((df["n_lines_dropped_conf"] > 0).sum()),
        "years_repaired_total": int(df["n_years_repaired"].sum()),
        "cvs_with_years_repaired": int((df["n_years_repaired"] > 0).sum()),
        "pii_cvs_affected": {c[4:]: int((df[c] > 0).sum()) for c in df.columns if c.startswith("pii_")},
        "masking_cvs_affected_total": int((df["n_masked"] > 0).sum()),
        "masking_cvs_affected_by_pattern": {c[5:]: int((df[c] > 0).sum()) for c in mask_cols},
        "masking_review": "50 textos OCR al azar (seed 42) leídos a mano: ver output/tables/masking_review_sample.csv",
        "cleaning_effect": effect.to_dict(orient="records"),
    }
    save_json(summary, METRICS_DIR / "preprocessing_summary.json")
    log.info("CVs enmascarados: %d | años reparados: %d en %d CVs | líneas < %.2f: %d",
             summary["masking_cvs_affected_total"], summary["years_repaired_total"],
             summary["cvs_with_years_repaired"], CONF_THRESHOLD, summary["lines_dropped_conf_total"])
    log.info("\n%s", effect.to_string(index=False))


if __name__ == "__main__":
    main()
