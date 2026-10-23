"""Muestreo estratificado por ocupación y descarga de imágenes de CV desde PopResume.

Por cada ocupación se hace streaming de sus shards de imágenes, se filtran solo los CVs sin foto
(`has_photo == False`), se barajan con semilla fija y se toman N. Cada imagen se guarda como
`data/raw/images/<id>.png`. Luego se descarga el config `text` y se guarda únicamente `id` y `text`
de los CVs muestreados: ese texto es solo una referencia para medir la calidad del OCR.

Salidas:
    data/raw/images/<id>.png         imágenes (no se suben a GitHub)
    data/raw/metadata.csv            todos los atributos de los CVs muestreados (no se sube)
    data/raw/reference_text.parquet  id + text de referencia (no se sube)
    data/sampled_ids.csv             ids muestreados + ocupación + target (se sube, reproducibilidad)
    output/tables/target_distribution.csv
    output/metrics/sampling_summary.json

Uso:
    python -m src.download_data                      # 800 CVs por ocupación
    python -m src.download_data --n 2 --buffer-size 10   # prueba rápida
    python -m src.download_data --skip-images        # solo regenera el texto de referencia
"""

from __future__ import annotations

import argparse
import io

import pandas as pd
from datasets import Image, load_dataset
from PIL import Image as PILImage
from tqdm import tqdm

from src.utils import (
    DATA_DIR, HF_DATASET, IMAGES_DIR, METRICS_DIR, OCCUPATIONS, RAW_DIR, SEED, TABLES_DIR,
    TARGET, TARGET_CAP, ensure_dirs, get_logger, save_json, timer,
)

log = get_logger("download")

METADATA_PATH = RAW_DIR / "metadata.csv"
REFERENCE_TEXT_PATH = RAW_DIR / "reference_text.parquet"
SAMPLED_IDS_PATH = DATA_DIR / "sampled_ids.csv"

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def save_image(image: dict, path) -> tuple[int, int]:
    """Guarda la imagen como PNG. Si los bytes ya son PNG se escriben tal cual (sin recodificar)."""
    data = image["bytes"]
    with PILImage.open(io.BytesIO(data)) as im:
        size = im.size
        if path.exists():
            return size
        if data[:8] == PNG_SIGNATURE:
            path.write_bytes(data)
        else:
            im.convert("RGB").save(path, format="PNG")
    return size


def sample_occupation(occupation: str, n: int, seed: int, buffer_size: int, save_images: bool) -> list[dict]:
    ds = load_dataset(
        "parquet",
        data_files=f"hf://datasets/{HF_DATASET}/data/images/{occupation}-*.parquet",
        split="train",
        streaming=True,
    )
    # decode=False: recibimos los bytes originales y evitamos decodificar imágenes que se descartan
    ds = ds.cast_column("image", Image(decode=False))
    ds = ds.filter(lambda r: not r["has_photo"]).shuffle(seed=seed, buffer_size=buffer_size)

    rows = []
    for r in tqdm(ds.take(n), total=n, desc=occupation):
        image = r.pop("image")
        r.pop("text", None)  # el texto de referencia se toma del config `text`
        if save_images:
            r["img_width"], r["img_height"] = save_image(image, IMAGES_DIR / f"{r['id']}.png")
        r["occupation"] = occupation
        rows.append(r)

    if len(rows) < n:
        log.warning("%s: solo se obtuvieron %d de %d CVs sin foto", occupation, len(rows), n)
    return rows


def fetch_reference_text(ids: set[str]) -> pd.DataFrame:
    ds = load_dataset(HF_DATASET, "text", split="train").select_columns(["id", "text"])
    df = ds.to_pandas()
    df = df[df["id"].isin(ids)].reset_index(drop=True)
    missing = ids - set(df["id"])
    if missing:
        log.warning("%d ids muestreados no tienen texto de referencia", len(missing))
    return df


def summarize_target(meta: pd.DataFrame) -> dict:
    """Distribución del target por ocupación y chequeo del tope (enunciado: avisar si > 15 % en el tope)."""
    by_occ = (
        meta.groupby("occupation")
        .agg(
            n=("id", "size"),
            mean_raw=(f"{TARGET}_raw", "mean"),
            median=(TARGET, "median"),
            max_raw=(f"{TARGET}_raw", "max"),
            n_at_cap=(TARGET, lambda s: int((s == TARGET_CAP).sum())),
            n_above_cap_raw=(f"{TARGET}_raw", lambda s: int((s > TARGET_CAP).sum())),
        )
        .reset_index()
    )
    total = pd.DataFrame([{
        "occupation": "TOTAL",
        "n": len(meta),
        "mean_raw": meta[f"{TARGET}_raw"].mean(),
        "median": meta[TARGET].median(),
        "max_raw": meta[f"{TARGET}_raw"].max(),
        "n_at_cap": int((meta[TARGET] == TARGET_CAP).sum()),
        "n_above_cap_raw": int((meta[f"{TARGET}_raw"] > TARGET_CAP).sum()),
    }])
    table = pd.concat([by_occ, total], ignore_index=True)
    table["pct_at_cap"] = 100 * table["n_at_cap"] / table["n"]
    table.round(2).to_csv(TABLES_DIR / "target_distribution.csv", index=False)

    pct = float(table.loc[table["occupation"] == "TOTAL", "pct_at_cap"].iloc[0])
    summary = {
        "n_total": int(len(meta)),
        "n_per_occupation": meta["occupation"].value_counts().to_dict(),
        "cap": TARGET_CAP,
        "n_at_cap": int((meta[TARGET] == TARGET_CAP).sum()),
        "pct_at_cap": round(pct, 2),
        "cap_above_15pct": pct > 15,
        "exp_year_value_counts": meta[TARGET].value_counts().sort_index().to_dict(),
    }
    save_json(summary, METRICS_DIR / "sampling_summary.json")
    log.info("CVs en el tope de %d años: %d (%.1f %%)", TARGET_CAP, summary["n_at_cap"], pct)
    if summary["cap_above_15pct"]:
        log.warning("Más del 15 %% de los CVs está en el tope: evaluar subir el tope a 25 o 30.")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=None, help="CVs por ocupación (por defecto, los de utils.OCCUPATIONS)")
    parser.add_argument("--occupations", nargs="+", default=list(OCCUPATIONS), choices=list(OCCUPATIONS))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--buffer-size", type=int, default=5000)
    parser.add_argument("--skip-images", action="store_true", help="usar data/sampled_ids.csv existente y no descargar imágenes")
    parser.add_argument("--skip-text", action="store_true", help="no descargar el texto de referencia")
    args = parser.parse_args()

    ensure_dirs()

    if args.skip_images:
        meta = pd.read_csv(SAMPLED_IDS_PATH, dtype={"id": str})
        log.info("Usando %d ids de %s", len(meta), SAMPLED_IDS_PATH)
    else:
        rows = []
        with timer("descarga de imágenes", log):
            for occ in args.occupations:
                n = args.n or OCCUPATIONS[occ]
                rows += sample_occupation(occ, n, args.seed, args.buffer_size, save_images=True)
        meta = pd.DataFrame(rows)
        meta["id"] = meta["id"].astype(str)
        meta[f"{TARGET}_raw"] = meta[TARGET]
        meta[TARGET] = meta[TARGET].clip(0, TARGET_CAP)
        meta["seed"] = args.seed
        meta.to_csv(METADATA_PATH, index=False)
        meta[["id", "occupation", f"{TARGET}_raw", TARGET, "seed"]].to_csv(SAMPLED_IDS_PATH, index=False)
        log.info("Guardados %d CVs en %s", len(meta), IMAGES_DIR)
        summarize_target(meta)

    if not args.skip_text:
        with timer("texto de referencia", log):
            ref = fetch_reference_text(set(meta["id"]))
        ref.to_parquet(REFERENCE_TEXT_PATH, index=False)
        log.info("Texto de referencia de %d CVs en %s", len(ref), REFERENCE_TEXT_PATH)


if __name__ == "__main__":
    main()
