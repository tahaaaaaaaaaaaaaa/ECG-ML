
"""Validation structurelle et numérique des caractéristiques statistiques.

Contrôles par partition : schéma exact (121 colonnes dans l'ordre attendu),
identifiants (uniques, identiques aux métadonnées et dans le même ordre),
valeurs finies, identités mathématiques, inégalités, convention de variance
nulle. Contrôle global : aucun ECG commun à deux partitions.

Ce script ne modifie aucun fichier. Il se termine avec un code non nul
si un contrôle échoue.
"""

from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd

# Permet aussi l'exécution directe du fichier (bouton « Run » de VS Code).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import (  # noqa: E402
    FEATURE_COLUMNS,
    FEATURES_DIR,
    LEADS,
    SPLITS,
    SPLITS_DIR,
    features_path,
    metadata_path,
)


EXPECTED_COLUMNS = ["ecg_id"] + FEATURE_COLUMNS


def leads(frame, feature):
    """Matrice (n_ecg, 12) d'une caractéristique pour toutes les dérivations."""
    return frame[[f"{lead}_{feature}" for lead in LEADS]].to_numpy(dtype=np.float64)


def read_csv_or_none(file_path):
    """Lit un CSV ; retourne None (ÉCHEC propre) s'il est vide ou illisible."""
    try:
        return pd.read_csv(file_path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        print(f"[ÉCHEC] Fichier illisible : {file_path} ({exc})")
        return None


def validate_split(split_name, features_dir, splits_dir):
    path = features_path(split_name, features_dir)
    meta_path = metadata_path(split_name, splits_dir)
    expected_rows = SPLITS[split_name]["n_ecg"]

    print("\n" + "=" * 60)
    print(f"VALIDATION : {split_name.upper()}")
    print("=" * 60)

    for file_path in (path, meta_path):
        if not file_path.is_file():
            print(f"[ÉCHEC] Fichier absent : {file_path}")
            return False, None

    df = read_csv_or_none(path)
    metadata = read_csv_or_none(meta_path)

    if df is None or metadata is None:
        return False, None

    print(f"Dimensions                 : {df.shape}")
    print(f"Nombre d'ECG attendu       : {expected_rows}")

    checks = {}
    checks["nombre de lignes correct"] = len(df) == expected_rows
    checks["121 colonnes, noms et ordre exacts"] = (
        list(df.columns) == EXPECTED_COLUMNS
    )

    if not checks["121 colonnes, noms et ordre exacts"]:
        for label, passed in checks.items():
            print(f"[{'OK' if passed else 'ÉCHEC'}] {label}")
        return False, None

    # Identifiants
    ids = df["ecg_id"]
    checks["identifiants non manquants"] = bool(ids.notna().all())
    checks["identifiants entiers"] = pd.api.types.is_integer_dtype(ids)
    checks["identifiants uniques"] = bool(ids.is_unique)
    checks["mêmes ECG que les métadonnées"] = (
        set(ids) == set(metadata["ecg_id"])
    )
    checks["même ordre que les métadonnées"] = (
        ids.tolist() == metadata["ecg_id"].tolist()
    )

    # Valeurs numériques
    numeric = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64)
    missing = int(numeric.isna().sum().sum())
    nonfinite = int((~np.isfinite(values)).sum())

    print(f"Valeurs manquantes         : {missing}")
    print(f"Valeurs non finies         : {nonfinite}")
    checks["valeurs numériques finies"] = missing == 0 and nonfinite == 0

    # Identités mathématiques (valeurs écrites avec 12 chiffres significatifs).
    # Lecture sur les valeurs converties : une cellule non numérique devient
    # NaN et fait échouer les contrôles proprement.
    mean, std, rms = leads(numeric, "mean"), leads(numeric, "std"), leads(numeric, "rms")
    median, iqr = leads(numeric, "median"), leads(numeric, "iqr")
    minimum, maximum = leads(numeric, "min"), leads(numeric, "max")
    line_length = leads(numeric, "line_length")
    skewness, kurt = leads(numeric, "skewness"), leads(numeric, "kurtosis")

    tol = 1e-9
    checks["rms² = std² + mean²"] = bool(np.allclose(
        rms ** 2, std ** 2 + mean ** 2, rtol=1e-8, atol=1e-12
    ))
    checks["min <= médiane <= max"] = bool(
        np.all(minimum <= median + tol) and np.all(median <= maximum + tol)
    )
    checks["min <= moyenne <= max"] = bool(
        np.all(minimum <= mean + tol) and np.all(mean <= maximum + tol)
    )
    checks["std, iqr, line_length >= 0"] = bool(
        np.all(std >= 0) and np.all(iqr >= 0) and np.all(line_length >= 0)
    )

    # Convention de variance nulle : asymétrie et aplatissement fixés à 0.
    zero_rows, zero_leads = np.nonzero(std == 0.0)
    zero_cases = [
        (int(ids.iloc[r]) if pd.notna(ids.iloc[r]) else None, LEADS[c])
        for r, c in zip(zero_rows, zero_leads)
    ]
    print(f"Dérivations à variance nulle : {zero_cases or 'aucune'}")
    checks["convention variance nulle (skewness = kurtosis = 0)"] = bool(
        np.all(skewness[std == 0.0] == 0.0) and np.all(kurt[std == 0.0] == 0.0)
    )

    for label, passed in checks.items():
        print(f"[{'OK' if passed else 'ÉCHEC'}] {label}")

    success = all(checks.values())
    print(f"\nRésultat : {'VALIDÉ' if success else 'NON VALIDÉ'}")
    return success, set(ids)


def main():
    parser = argparse.ArgumentParser(
        description="Validation des caractéristiques statistiques."
    )
    parser.add_argument(
        "--features-dir", type=Path, default=FEATURES_DIR,
        help="Dossier des fichiers *_features.csv (défaut : projet).",
    )
    parser.add_argument(
        "--splits-dir", type=Path, default=SPLITS_DIR,
        help="Dossier des fichiers *_metadata.csv (défaut : projet).",
    )
    args = parser.parse_args()

    results = {}
    id_sets = {}

    for split_name in SPLITS:
        success, ids = validate_split(
            split_name, args.features_dir, args.splits_dir
        )
        results[split_name] = success
        id_sets[split_name] = ids or set()

    print("\n" + "=" * 60)
    print("CONTRÔLE GLOBAL")
    print("=" * 60)

    names = list(id_sets)
    overlap_ok = True
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            common = len(id_sets[left] & id_sets[right])
            overlap_ok &= common == 0
            print(f"ECG communs {left}/{right} : {common}")

    print(f"[{'OK' if overlap_ok else 'ÉCHEC'}] aucun ECG commun entre partitions")

    if not (all(results.values()) and overlap_ok):
        failed = [name for name, ok in results.items() if not ok]
        raise SystemExit(
            f"Validation échouée ({', '.join(failed) or 'contrôle global'}). "
            "Ne pas poursuivre."
        )

    print("\nToutes les partitions sont VALIDÉES.")


if __name__ == "__main__":
    main()
