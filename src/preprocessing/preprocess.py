"""Ajustement et chargement du centrage-réduction du classifieur binaire."""

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import (  # noqa: E402
    MODEL_FEATURE_COLUMNS,
    MODEL_INPUT_DIR,
    PREPROCESSING_DIR,
    SPLITS,
)


TOLERANCE = 1e-9


def preprocessor_path(variant="P1"):
    if variant != "P1":
        raise ValueError(f"Prétraitement binaire inconnu : {variant}")
    return PREPROCESSING_DIR / "P1.joblib"


def load_preprocessor(variant="P1"):
    return joblib.load(preprocessor_path(variant))


def fit_preprocessor(X_train):
    expected = (SPLITS["train"]["n_ecg"], len(MODEL_FEATURE_COLUMNS))
    if X_train.shape != expected:
        raise ValueError(
            f"Ajustement refusé : {X_train.shape} au lieu de {expected} (train)."
        )
    return StandardScaler().fit(X_train)


def main():
    PREPROCESSING_DIR.mkdir(parents=True, exist_ok=True)
    matrices = {}
    for split in ("train", "validation"):
        with np.load(MODEL_INPUT_DIR / f"{split}.npz", allow_pickle=False) as data:
            if data["feature_names"].tolist() != MODEL_FEATURE_COLUMNS:
                raise ValueError(f"{split} : colonnes inattendues.")
            matrices[split] = data["X"]

    scaler = fit_preprocessor(matrices["train"])
    transformed = {
        split: scaler.transform(values)
        for split, values in matrices.items()
    }
    parameters = pd.DataFrame({
        "feature": MODEL_FEATURE_COLUMNS,
        "mean": scaler.mean_,
        "scale": scaler.scale_,
    })
    checks = {
        "sorties finies": all(np.isfinite(values).all() for values in transformed.values()),
        "moyenne train proche de 0": bool(
            np.all(np.abs(transformed["train"].mean(axis=0)) < TOLERANCE)
        ),
        "écart-type train proche de 1": bool(
            np.all(np.abs(transformed["train"].std(axis=0) - 1) < TOLERANCE)
        ),
    }
    for label, passed in checks.items():
        print(f"[{'OK' if passed else 'ÉCHEC'}] {label}")
    if not all(checks.values()):
        raise SystemExit("Contrôles du prétraitement binaire échoués.")

    joblib.dump(scaler, preprocessor_path())
    parameters.to_csv(PREPROCESSING_DIR / "P1_parameters.csv", index=False)
    print(f"Prétraitement P1 enregistré : {preprocessor_path()}")


if __name__ == "__main__":
    main()
