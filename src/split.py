"""Split train / val / test (70 / 15 / 15) por ID de CV, estratificado por ocupación × banda de experiencia.

Todos los modelos leen estos mismos archivos. El test no se toca hasta la evaluación final.

Salidas: data/splits/{train,val,test}.csv con id, occupation, exp_year, band
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import DATA_DIR, METRICS_DIR, SEED, SPLIT_FRACTIONS, SPLITS_DIR, TARGET, get_logger, save_json, to_band

log = get_logger("split")


def make_splits(df: pd.DataFrame, seed: int = SEED) -> dict[str, pd.DataFrame]:
    df = df.copy()
    df["band"] = to_band(df[TARGET])
    strata = df["occupation"] + "|" + df["band"]

    test_frac = SPLIT_FRACTIONS["test"]
    val_frac = SPLIT_FRACTIONS["val"] / (1 - test_frac)
    rest, test = train_test_split(df, test_size=test_frac, stratify=strata, random_state=seed)
    train, val = train_test_split(rest, test_size=val_frac, stratify=strata.loc[rest.index], random_state=seed)
    return {name: part.sort_values("id").reset_index(drop=True)
            for name, part in {"train": train, "val": val, "test": test}.items()}


def load_splits() -> dict[str, pd.DataFrame]:
    return {name: pd.read_csv(SPLITS_DIR / f"{name}.csv", dtype={"id": str}) for name in SPLIT_FRACTIONS}


def main() -> None:
    sampled = pd.read_csv(DATA_DIR / "sampled_ids.csv", dtype={"id": str})
    # Solo entran CVs con OCR procesado (los fallidos, si hubiera, se excluyen y se reportan)
    processed = pd.read_csv(DATA_DIR / "processed" / "cvs.csv", dtype={"id": str}, usecols=["id"])
    df = sampled[sampled["id"].isin(processed["id"])][["id", "occupation", TARGET]]
    log.info("%d CVs con texto procesado (de %d muestreados)", len(df), len(sampled))

    splits = make_splits(df)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_csv(SPLITS_DIR / f"{name}.csv", index=False)

    table = pd.concat([p.assign(split=n) for n, p in splits.items()])
    counts = pd.crosstab([table["occupation"], table["band"]], table["split"])[list(splits)]
    counts.to_csv(METRICS_DIR.parent / "tables" / "split_counts.csv")
    summary = {
        "seed": SEED,
        "n": {n: len(p) for n, p in splits.items()},
        "mean_exp_year": {n: round(float(p[TARGET].mean()), 3) for n, p in splits.items()},
        "band_share": {n: p["band"].value_counts(normalize=True).round(4).to_dict() for n, p in splits.items()},
    }
    save_json(summary, METRICS_DIR / "split_summary.json")
    log.info("Tamaños: %s", summary["n"])


if __name__ == "__main__":
    main()
