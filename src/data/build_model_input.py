
"""Construction des matrices d'apprentissage X / Y (phase 2 du planning).

Pour chaque partition, les caractéristiques et les métadonnées sont
associées par ecg_id (fusion un-à-un, ordre des métadonnées). Le script
enregistre un fichier .npz par partition et un manifeste JSON, puis
vérifie l'ensemble des contrôles de l'étape 2.2. Aucun ajustement n'est
réalisé : la matrice de test est construite mécaniquement et seules ses
dimensions sont affichées.
"""

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import subprocess
import sys

import numpy as np
import pandas as pd

# Permet aussi l'exécution directe du fichier (bouton « Run » de VS Code).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import (  # noqa: E402
    CLASSES,
    MODEL_FEATURE_COLUMNS,
    MODEL_INPUT_DIR,
    ROOT,
    SPLITS,
    features_path,
    metadata_path,
)


SPOT_CHECK_ECG = 12722  # ECG du train à dérivation constante (V5)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(paths=()):
    """Commit courant, suffixé si l'arbre de travail contient des modifications.

    `paths` restreint le contrôle des modifications à certains dossiers
    (par exemple le code et les configurations), afin que les sorties d'une
    exécution en cours ne soient pas comptées comme des modifications.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", *paths], cwd=ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "inconnu"
    return commit + ("+modifications-non-commitées" if dirty else "")


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


# ============================================================
# 2.1 CONSTRUCTION
# ============================================================

def build_split(split_name):
    metadata = pd.read_csv(metadata_path(split_name))
    features = pd.read_csv(features_path(split_name))

    # Fusion un-à-un par ecg_id ; une jointure à gauche conserve
    # l'ordre des lignes des métadonnées.
    merged = metadata.merge(
        features, on="ecg_id", how="left",
        validate="one_to_one", indicator=True,
    )

    if not merged["_merge"].eq("both").all():
        raise ValueError(f"{split_name} : ECG sans caractéristiques.")

    if len(features) != len(metadata):
        raise ValueError(f"{split_name} : caractéristiques hors métadonnées.")

    patient_id = merged["patient_id"].to_numpy(dtype=np.float64)

    if not np.all(patient_id == np.round(patient_id)):
        raise ValueError(f"{split_name} : patient_id non entier.")

    arrays = {
        "X": merged[MODEL_FEATURE_COLUMNS].to_numpy(dtype=np.float64),
        "Y": merged[CLASSES].to_numpy(dtype=np.int8),
        "ecg_id": merged["ecg_id"].to_numpy(dtype=np.int64),
        "patient_id": patient_id.astype(np.int64),
        "feature_names": np.array(MODEL_FEATURE_COLUMNS),
        "class_names": np.array(CLASSES),
    }

    return arrays, metadata, features


# ============================================================
# 2.2 CONTRÔLES
# ============================================================

def check_split(split_name, arrays, metadata, features):
    n_ecg = SPLITS[split_name]["n_ecg"]
    X, Y, ids = arrays["X"], arrays["Y"], arrays["ecg_id"]
    feature_names = arrays["feature_names"].tolist()

    checks = {
        f"X de dimensions ({n_ecg}, {len(MODEL_FEATURE_COLUMNS)})":
            X.shape == (n_ecg, len(MODEL_FEATURE_COLUMNS)),
        f"Y de dimensions ({n_ecg}, {len(CLASSES)})":
            Y.shape == (n_ecg, len(CLASSES)),
        "noms des caractéristiques = MODEL_FEATURE_COLUMNS":
            feature_names == MODEL_FEATURE_COLUMNS,
        "ordre des classes NORM, MI, STTC, CD, HYP":
            arrays["class_names"].tolist() == ["NORM", "MI", "STTC", "CD", "HYP"],
        "Y identique aux étiquettes des métadonnées (par ecg_id)":
            np.array_equal(
                Y, metadata.set_index("ecg_id").loc[ids, CLASSES].to_numpy()
            ),
        "X identique aux caractéristiques enregistrées (par ecg_id)":
            np.array_equal(
                X,
                features.set_index("ecg_id")
                .loc[ids, MODEL_FEATURE_COLUMNS]
                .to_numpy(dtype=np.float64),
            ),
        "au moins une classe positive par ECG": bool(np.all(Y.sum(axis=1) >= 1)),
        "valeurs de Y dans {0, 1}": bool(np.isin(Y, (0, 1)).all()),
        "toutes les valeurs de X finies": bool(np.isfinite(X).all()),
        "ecg_id dans l'ordre des métadonnées":
            ids.tolist() == metadata["ecg_id"].tolist(),
    }

    if split_name == "train":
        row = int(np.flatnonzero(ids == SPOT_CHECK_ECG)[0])
        csv_row = (
            features.loc[features["ecg_id"] == SPOT_CHECK_ECG, MODEL_FEATURE_COLUMNS]
            .to_numpy(dtype=np.float64)[0]
        )
        checks[f"contrôle ponctuel ECG {SPOT_CHECK_ECG} (ligne {row})"] = (
            np.array_equal(X[row], csv_row)
        )

    for label, passed in checks.items():
        print(f"[{'OK' if passed else 'ÉCHEC'}] {label}")

    return all(checks.values())


def check_disjoint(all_arrays):
    names = list(all_arrays)
    success = True

    for i, left in enumerate(names):
        for right in names[i + 1:]:
            common_ids = np.intersect1d(
                all_arrays[left]["ecg_id"], all_arrays[right]["ecg_id"]
            ).size
            common_patients = np.intersect1d(
                all_arrays[left]["patient_id"], all_arrays[right]["patient_id"]
            ).size
            passed = common_ids == 0 and common_patients == 0
            success &= passed
            print(
                f"[{'OK' if passed else 'ÉCHEC'}] {left}/{right} : "
                f"{common_ids} ECG et {common_patients} patients communs"
            )

    return success


def check_round_trip(path, arrays):
    """Relit le fichier .npz enregistré et le compare aux tableaux en mémoire."""
    with np.load(path, allow_pickle=False) as saved:
        return set(saved.files) == set(arrays) and all(
            np.array_equal(saved[key], value) for key, value in arrays.items()
        )


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main():
    MODEL_INPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "script": "src/data/build_model_input.py",
        "n_features": len(MODEL_FEATURE_COLUMNS),
        "source_classes": CLASSES,
        "binary_classes": ["NORMAL", "PATHOLOGICAL"],
        "binary_label_rule": (
            "PATHOLOGICAL si MI, STTC, CD ou HYP est présent ; "
            "sinon NORMAL si NORM est présent."
        ),
        "labels_in_npz": (
            "Y contient les indicateurs des classes source, dans l'ordre "
            "source_classes ; train_binary les convertit en classes binaires."
        ),
        "splits": {},
    }

    all_arrays = {}
    success = True

    for split_name in SPLITS:
        print("\n" + "=" * 60)
        print(f"PARTITION : {split_name.upper()}")
        print("=" * 60)

        arrays, metadata, features = build_split(split_name)
        success &= check_split(split_name, arrays, metadata, features)

        output_path = MODEL_INPUT_DIR / f"{split_name}.npz"
        np.savez(output_path, **arrays)

        round_trip = check_round_trip(output_path, arrays)
        print(f"[{'OK' if round_trip else 'ÉCHEC'}] relecture du fichier .npz identique")
        success &= round_trip

        print(f"X : {arrays['X'].shape}  Y : {arrays['Y'].shape}")

        entry = {
            "features_csv": {
                "path": relative(features_path(split_name)),
                "sha256": sha256(features_path(split_name)),
            },
            "metadata_csv": {
                "path": relative(metadata_path(split_name)),
                "sha256": sha256(metadata_path(split_name)),
            },
            "npz": {"path": relative(output_path), "sha256": sha256(output_path)},
            "X_shape": list(arrays["X"].shape),
            "Y_shape": list(arrays["Y"].shape),
        }

        # Prévalences rapportées pour le train et la validation uniquement.
        if split_name != "test":
            positives = arrays["Y"].sum(axis=0)
            entry["positives"] = dict(zip(CLASSES, positives.tolist()))
            entry["prevalence"] = {
                cls: round(float(count) / len(arrays["Y"]), 4)
                for cls, count in zip(CLASSES, positives)
            }
            print("Prévalence : " + ", ".join(
                f"{cls} {value:.1%}" for cls, value in entry["prevalence"].items()
            ))

        manifest["splits"][split_name] = entry
        all_arrays[split_name] = arrays

    print("\n" + "=" * 60)
    print("CONTRÔLE GLOBAL")
    print("=" * 60)
    success &= check_disjoint(all_arrays)

    manifest_path = MODEL_INPUT_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nManifeste : {manifest_path}")

    if not success:
        raise SystemExit("Contrôles de la phase 2 en échec. Ne pas poursuivre.")

    print("\nPhase 2 : tous les contrôles sont validés.")


if __name__ == "__main__":
    main()
