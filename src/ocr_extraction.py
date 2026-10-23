"""PaddleOCR sobre cada imagen de CV, con caché (un JSON por CV) y reconstrucción del orden de lectura.

Por cada CV se guarda en `data/ocr/<id>.json`:
    - las líneas detectadas tal como las devuelve PaddleOCR: texto, confianza y bounding box [x1, y1, x2, y2]
    - el orden de lectura reconstruido (índices de las líneas) y el número de columnas detectado
    - el texto ya ordenado (`text`), que es la entrada de preprocessing.py
    - tiempo de OCR y modelos usados

Orden de lectura (ver `reading_order`):
    1. Se busca un "canal" vertical vacío en la zona central de la página (25 %–75 % del ancho).
       Si existe y hay suficientes líneas a ambos lados, la página es de dos columnas.
    2. Las líneas que cruzan el canal (nombre, títulos a lo ancho) cortan la página en bandas horizontales.
       Dentro de cada banda se lee primero la columna izquierda completa y luego la derecha.
    3. Dentro de una columna, las cajas cuyo centro vertical está a menos de media altura de línea se
       agrupan en la misma fila y se leen de izquierda a derecha (p. ej. "Cargo ......... 2016.05–2019.02").

Uso:
    python -m src.ocr_extraction                       # todas las imágenes de data/raw/images, salta las ya cacheadas
    python -m src.ocr_extraction --device gpu:0
    python -m src.ocr_extraction --rec-model PP-OCRv5_mobile_rec --det-model PP-OCRv5_mobile_det \
        --out-dir data/ocr_mobile --limit 50          # comparación de modelos en una muestra
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils import IMAGES_DIR, METRICS_DIR, OCR_DIR, TABLES_DIR, ensure_dirs, get_logger, load_json, save_json

log = get_logger("ocr")

DEFAULT_DET_MODEL = "PP-OCRv5_server_det"
DEFAULT_REC_MODEL = "PP-OCRv5_server_rec"


# ---------------------------------------------------------------------------
# PaddleOCR
# ---------------------------------------------------------------------------
def build_ocr(det_model: str = DEFAULT_DET_MODEL, rec_model: str = DEFAULT_REC_MODEL, device: str | None = None):
    from paddleocr import PaddleOCR

    kwargs = dict(
        lang="en",
        text_detection_model_name=det_model,
        text_recognition_model_name=rec_model,
        # Los CVs son PDFs renderizados: vienen derechos y planos, no hace falta corregir orientación
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    if device:
        kwargs["device"] = device
    if device is None or device.startswith("cpu"):
        # El backend oneDNN de Paddle 3.x falla en CPU con PP-OCRv5 (ConvertPirAttribute2RuntimeAttribute)
        kwargs["enable_mkldnn"] = False
    return PaddleOCR(**kwargs)


def run_ocr(ocr, image) -> list[dict]:
    """Corre PaddleOCR sobre una ruta o un array RGB y devuelve las líneas [{text, score, box}]."""
    if not isinstance(image, (str, Path)):
        image = np.asarray(image)[:, :, ::-1]  # PaddleOCR espera BGR, como cv2
    result = ocr.predict(str(image) if isinstance(image, Path) else image)[0]
    boxes = np.asarray(result["rec_boxes"]).reshape(-1, 4).tolist()
    return [
        {"text": t, "score": round(float(s), 4), "box": [int(v) for v in b]}
        for t, s, b in zip(result["rec_texts"], result["rec_scores"], boxes)
    ]


# ---------------------------------------------------------------------------
# Orden de lectura
# ---------------------------------------------------------------------------
def find_column_gutter(lines: list[dict], page_width: int, min_gap_frac: float = 0.03,
                       min_lines_per_side: int = 5) -> float | None:
    """Devuelve la coordenada x del canal entre dos columnas, o None si la página es de una columna."""
    if len(lines) < 2 * min_lines_per_side or page_width <= 0:
        return None
    # Las líneas muy anchas (títulos a lo ancho) no sirven para encontrar columnas
    narrow = [l["box"] for l in lines if (l["box"][2] - l["box"][0]) < 0.6 * page_width]
    covered = np.zeros(page_width + 1, dtype=bool)
    for x1, _, x2, _ in narrow:
        covered[max(0, x1):min(page_width, x2) + 1] = True

    lo, hi = int(0.25 * page_width), int(0.75 * page_width)
    best_start, best_len, run_start = None, 0, None
    for x in range(lo, hi + 1):
        if not covered[x]:
            run_start = x if run_start is None else run_start
            if x - run_start + 1 > best_len:
                best_start, best_len = run_start, x - run_start + 1
        else:
            run_start = None
    if best_start is None or best_len < min_gap_frac * page_width:
        return None

    gutter = best_start + best_len / 2
    left = sum(1 for b in narrow if b[2] <= gutter)
    right = sum(1 for b in narrow if b[0] >= gutter)
    return gutter if min(left, right) >= min_lines_per_side else None


def _order_rows(idx: list[int], lines: list[dict]) -> list[list[int]]:
    """Agrupa en filas las cajas con centro vertical cercano; filas de arriba abajo, cajas de izquierda a derecha."""
    if not idx:
        return []
    heights = [lines[i]["box"][3] - lines[i]["box"][1] for i in idx]
    tol = 0.5 * float(np.median(heights))
    center = lambda i: (lines[i]["box"][1] + lines[i]["box"][3]) / 2  # noqa: E731

    rows: list[list[int]] = []
    for i in sorted(idx, key=center):
        if rows and abs(center(i) - np.mean([center(j) for j in rows[-1]])) <= tol:
            rows[-1].append(i)
        else:
            rows.append([i])
    return [sorted(r, key=lambda i: lines[i]["box"][0]) for r in rows]


def reading_order(lines: list[dict], page_width: int) -> tuple[list[list[int]], int]:
    """Devuelve (filas de índices en orden de lectura, número de columnas detectado)."""
    gutter = find_column_gutter(lines, page_width)
    all_idx = list(range(len(lines)))
    if gutter is None:
        return _order_rows(all_idx, lines), 1

    spans = lambda i: lines[i]["box"][0] < gutter < lines[i]["box"][2]  # noqa: E731
    top = lambda i: lines[i]["box"][1]  # noqa: E731

    rows: list[list[int]] = []
    band: list[int] = []
    for i in sorted(all_idx, key=top):
        if spans(i):
            # Una línea a lo ancho cierra la banda actual: columna izquierda, luego derecha
            rows += _order_rows([j for j in band if lines[j]["box"][2] <= gutter], lines)
            rows += _order_rows([j for j in band if lines[j]["box"][2] > gutter], lines)
            rows.append([i])
            band = []
        else:
            band.append(i)
    rows += _order_rows([j for j in band if lines[j]["box"][2] <= gutter], lines)
    rows += _order_rows([j for j in band if lines[j]["box"][2] > gutter], lines)
    return rows, 2


def rows_to_text(rows: list[list[int]], lines: list[dict]) -> str:
    return "\n".join(" ".join(lines[i]["text"] for i in row) for row in rows)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def ocr_record(ocr, image_path: Path, cv_id: str, models: dict) -> dict:
    from PIL import Image

    with Image.open(image_path) as im:
        width, height = im.size
    start = time.perf_counter()
    lines = run_ocr(ocr, image_path)
    elapsed = time.perf_counter() - start
    rows, n_columns = reading_order(lines, width)
    scores = [l["score"] for l in lines]
    return {
        "id": cv_id,
        **models,
        "image_size": [width, height],
        "ocr_seconds": round(elapsed, 3),
        "n_lines": len(lines),
        "mean_score": round(float(np.mean(scores)), 4) if scores else None,
        "n_columns": n_columns,
        "lines": lines,
        "reading_order": rows,
        "text": rows_to_text(rows, lines),
    }


def hardware_info(device: str | None) -> dict:
    info = {"platform": platform.platform(), "processor": platform.processor(), "device": device or "auto",
            "gpu": None}
    if device and device.startswith("cpu"):
        return info
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        info["gpu"] = gpu or None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        info["gpu"] = None
    try:
        import paddle

        info["paddle_version"] = paddle.__version__
        info["paddle_cuda"] = paddle.device.is_compiled_with_cuda()
    except ImportError:
        pass
    return info


def summarize(out_dir: Path, ids: list[str], failed: dict[str, str], hardware: dict, tag: str) -> dict:
    rows = []
    for cv_id in ids:
        path = out_dir / f"{cv_id}.json"
        if path.exists():
            r = load_json(path)
            rows.append({k: r[k] for k in ["id", "n_lines", "mean_score", "n_columns", "ocr_seconds"]}
                        | {"min_score": min((l["score"] for l in r["lines"]), default=None)})
    columns = ["id", "n_lines", "mean_score", "n_columns", "ocr_seconds", "min_score"]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(TABLES_DIR / f"ocr_per_cv{tag}.csv", index=False)

    # La primera imagen incluye la carga del modelo en GPU: se excluye de las estadísticas de tiempo
    times = df["ocr_seconds"].sort_values().iloc[:-1] if len(df) > 1 else df["ocr_seconds"]
    summary = {
        "n_images": len(ids),
        "n_processed": int(len(df)),
        "n_failed": len(failed),
        "failed": failed,
        "n_empty": int((df["n_lines"] == 0).sum()) if len(df) else 0,
        "mean_seconds_per_cv": round(float(times.mean()), 3) if len(df) else None,
        "median_seconds_per_cv": round(float(times.median()), 3) if len(df) else None,
        "total_ocr_hours": round(float(df["ocr_seconds"].sum()) / 3600, 3) if len(df) else None,
        "mean_confidence": round(float(df["mean_score"].mean()), 4) if len(df) else None,
        "mean_lines_per_cv": round(float(df["n_lines"].mean()), 1) if len(df) else None,
        "n_two_column": int((df["n_columns"] == 2).sum()) if len(df) else 0,
        "hardware": hardware,
    }
    save_json(summary, METRICS_DIR / f"ocr_summary{tag}.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images-dir", type=Path, default=IMAGES_DIR)
    parser.add_argument("--out-dir", type=Path, default=OCR_DIR)
    parser.add_argument("--det-model", default=DEFAULT_DET_MODEL)
    parser.add_argument("--rec-model", default=DEFAULT_REC_MODEL)
    parser.add_argument("--device", default=None, help="cpu, gpu o gpu:0 (por defecto PaddleOCR decide)")
    parser.add_argument("--limit", type=int, default=None, help="procesar solo las primeras N imágenes")
    parser.add_argument("--overwrite", action="store_true", help="rehacer OCR aunque exista el JSON")
    args = parser.parse_args()

    ensure_dirs()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = "" if args.out_dir.resolve() == OCR_DIR.resolve() else f"_{args.out_dir.name}"

    images = sorted(args.images_dir.glob("*.png"))[: args.limit]
    ids = [p.stem for p in images]
    todo = [p for p in images if args.overwrite or not (args.out_dir / f"{p.stem}.json").exists()]
    log.info("%d imágenes, %d ya en caché, %d por procesar", len(images), len(images) - len(todo), len(todo))

    failed: dict[str, str] = {}
    if todo:
        models = {"det_model": args.det_model, "rec_model": args.rec_model}
        ocr = build_ocr(args.det_model, args.rec_model, args.device)
        for path in tqdm(todo, desc="OCR"):
            try:
                save_json(ocr_record(ocr, path, path.stem, models), args.out_dir / f"{path.stem}.json")
            except Exception as e:  # una imagen corrupta no debe detener las otras 3999
                failed[path.stem] = f"{type(e).__name__}: {e}"
                log.error("Falló %s: %s", path.stem, failed[path.stem])

    summary = summarize(args.out_dir, ids, failed, hardware_info(args.device), tag)
    log.info(
        "Procesados %d/%d | fallidos %d | %.2f s/CV | confianza media %.4f | dos columnas: %d",
        summary["n_processed"], summary["n_images"], summary["n_failed"],
        summary["mean_seconds_per_cv"] or 0, summary["mean_confidence"] or 0, summary["n_two_column"],
    )


if __name__ == "__main__":
    main()
