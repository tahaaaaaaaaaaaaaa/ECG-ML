
from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
import wfdb
from scipy.stats import skew, kurtosis

# Permet aussi l'exécution directe du fichier (bouton « Run » de VS Code).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import (  # noqa: E402
    DATA_ROOT,
    FEATURE_COLUMNS,
    FEATURES_DIR,
    FS as FS_EXPECTED,
    LEADS,
    N_SAMPLES as N_SAMPLES_EXPECTED,
    SPLITS,
    STAT_FEATURES as FEATURES,
    metadata_path as split_metadata_path,
)


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = FEATURES_DIR

N_FEATURES = len(LEADS) * len(FEATURES)


# ============================================================
# CHEMINS WFDB ET CHARGEMENT
# ============================================================

def resolve_record_path(row):
    filename_column = next(
        (
            col for col in ("filename_hr", "filename_500")
            if col in row.index and pd.notna(row[col])
        ),
        None,
    )

    if filename_column is None:
        raise ValueError(
            "Colonne filename_hr ou filename_500 absente."
        )

    path = DATA_ROOT / str(row[filename_column])
    path = Path(path)

    if path.suffix.lower() in {".hea", ".dat"}:
        path = path.with_suffix("")

    if not path.with_suffix(".hea").is_file():
        raise FileNotFoundError(
            f"Fichier .hea introuvable : "
            f"{path.with_suffix('.hea')}"
        )

    return path


def load_ecg(row):
    path = resolve_record_path(row)
    record = wfdb.rdrecord(str(path))

    signal = np.asarray(record.p_signal, dtype=np.float64)

    if signal.shape != (N_SAMPLES_EXPECTED, len(LEADS)):
        raise ValueError(
            f"Dimensions inattendues : {signal.shape}"
        )

    if not np.isclose(float(record.fs), FS_EXPECTED):
        raise ValueError(f"Fréquence inattendue : {record.fs} Hz")

    names = [str(s).strip().upper() for s in record.sig_name]
    expected_names = [s.upper() for s in LEADS]

    if names != expected_names:
        raise ValueError(
            f"Ordre des dérivations incorrect : {names}"
        )

    units = [str(u).strip().lower() for u in record.units]

    if units != ["mv"] * len(LEADS):
        raise ValueError(f"Unités inattendues : {record.units}")

    if not np.all(np.isfinite(signal)):
        raise ValueError("Le signal contient NaN ou Inf.")

    return signal


# ============================================================
# EXTRACTION
# ============================================================


def compute_lead_features(x):
    x = np.asarray(x, dtype=np.float64)

    if x.ndim != 1 or x.size < 2:
        raise ValueError("Dérivation invalide.")

    if not np.all(np.isfinite(x)):
        raise ValueError("Signal non fini.")

    q25, q75 = np.percentile(x, [25, 75])
    std_value = float(np.std(x, ddof=0))

    # Skewness et kurtosis ne sont pas définies
    # lorsque le signal est strictement constant.
    if std_value == 0.0:
        skewness_value = 0.0
        kurtosis_value = 0.0
    else:
        skewness_value = float(skew(x, bias=False))
        kurtosis_value = float(
            kurtosis(x, fisher=True, bias=False)
        )

    result = {
        "mean": float(np.mean(x)),
        "std": std_value,
        "rms": float(np.sqrt(np.mean(x ** 2))),
        "median": float(np.median(x)),
        "iqr": float(q75 - q25),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "skewness": skewness_value,
        "kurtosis": kurtosis_value,
        "line_length": float(np.sum(np.abs(np.diff(x)))),
    }

    bad_features = [
        name
        for name, value in result.items()
        if not np.isfinite(value)
    ]

    if bad_features:
        raise ValueError(
            f"Caractéristiques non finies : {bad_features}"
        )

    return result



def extract_features(signal):
    result = {}

    for index, lead in enumerate(LEADS):
        values = compute_lead_features(signal[:, index])

        for name in FEATURES:
            result[f"{lead}_{name}"] = values[name]

    if len(result) != N_FEATURES:
        raise RuntimeError(
            f"{len(result)} caractéristiques au lieu de {N_FEATURES}."
        )

    # Contrôles d'identités mathématiques.
    for lead in LEADS:
        mean = result[f"{lead}_mean"]
        std = result[f"{lead}_std"]
        rms = result[f"{lead}_rms"]
        minimum = result[f"{lead}_min"]
        maximum = result[f"{lead}_max"]

        if not np.isclose(
            rms ** 2, std ** 2 + mean ** 2,
            rtol=1e-7, atol=1e-10
        ):
            raise ValueError(f"Identité RMS incohérente : {lead}")

    return result


# ============================================================
# ORDRE DES LIGNES
# ============================================================

def reorder_like_metadata(output_path, ordered_ids):
    """Réécrit le fichier de sortie dans l'ordre des métadonnées.

    La reprise ajoute les ECG manquants en fin de fichier : sans cette
    étape, l'ordre des lignes diffère de celui des métadonnées. Les lignes
    sont déplacées telles quelles, octet par octet : aucune valeur n'est
    relue ni réécrite. Retourne True si le fichier a été réordonné.
    """
    output_path = Path(output_path)
    header, *rows = output_path.read_bytes().splitlines(keepends=True)

    line_end = b"\r\n" if header.endswith(b"\r\n") else b"\n"
    rows = [
        row if row.endswith(b"\n") else row + line_end
        for row in rows
        if row.strip()  # lignes vides ignorées, comme par pandas
    ]

    rows_by_id = {}
    for row in rows:
        ecg_id = int(row.split(b",", 1)[0])

        if ecg_id in rows_by_id:
            raise ValueError(f"ECG {ecg_id} présent deux fois.")

        rows_by_id[ecg_id] = row

    unknown_ids = set(rows_by_id) - set(ordered_ids)
    if unknown_ids:
        raise ValueError(
            f"{len(unknown_ids)} ECG absents des métadonnées."
        )

    ordered_rows = [
        rows_by_id[ecg_id]
        for ecg_id in ordered_ids
        if ecg_id in rows_by_id
    ]

    if ordered_rows == rows:
        return False

    temporary_path = output_path.with_name(output_path.name + ".tmp")

    try:
        temporary_path.write_bytes(header + b"".join(ordered_rows))
        temporary_path.replace(output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    return True


# ============================================================
# TRAITEMENT D'UNE PARTITION
# ============================================================

def process_split(split_name, max_records=None):
    config = SPLITS[split_name]

    metadata_path = split_metadata_path(split_name)
    output_path = OUTPUT_DIR / f"{split_name}_features.csv"
    error_path = OUTPUT_DIR / f"{split_name}_errors.csv"

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Métadonnées introuvables : {metadata_path}"
        )

    metadata = pd.read_csv(metadata_path)

    if "ecg_id" not in metadata.columns:
        raise ValueError("Colonne ecg_id absente.")

    if metadata["ecg_id"].isna().any():
        raise ValueError("Identifiant ECG manquant dans les métadonnées.")

    if metadata["ecg_id"].duplicated().any():
        raise ValueError("Identifiants ECG dupliqués dans les métadonnées.")

    if len(metadata) != config["n_ecg"]:
        raise ValueError(
            f"Effectif inattendu pour {split_name} : "
            f"{len(metadata)} au lieu de {config['n_ecg']}."
        )

    # Colonnes fixes et ordre reproductible.
    feature_columns = FEATURE_COLUMNS
    output_columns = ["ecg_id"] + feature_columns

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Reprise : les ECG déjà sauvegardés ne sont pas recalculés.
    if output_path.exists():
        existing = pd.read_csv(output_path)

        if list(existing.columns) != output_columns:
            raise ValueError(
                f"Schéma de sortie inattendu : {output_path}. "
                "Renommer ou déplacer ce fichier avant de recommencer."
            )

        if existing["ecg_id"].duplicated().any():
            raise ValueError("Doublons dans le fichier de sortie.")

        unknown_ids = set(existing["ecg_id"]) - set(metadata["ecg_id"])

        if unknown_ids:
            raise ValueError(
                "Le fichier existant contient des ECG hors partition."
            )

        completed_ids = set(existing["ecg_id"])

        # Vérifie que les résultats précédents sont complets.
        if not np.isfinite(
            existing[feature_columns].to_numpy(dtype=np.float64)
        ).all():
            raise ValueError(
                "Le fichier existant contient des valeurs non finies."
            )
    else:
        completed_ids = set()

    pending = metadata[
        ~metadata["ecg_id"].isin(completed_ids)
    ]

    if max_records is not None:
        pending = pending.head(max_records)

    print(f"\n{'=' * 60}")
    print(f"PARTITION : {split_name.upper()}")
    print(f"ECG attendus          : {len(metadata)}")
    print(f"ECG déjà sauvegardés  : {len(completed_ids)}")
    print(f"ECG à traiter         : {len(pending)}")
    print(f"Caractéristiques/ECG  : {N_FEATURES}")
    print(f"{'=' * 60}")

    batch = []
    errors = []
    processed_new = 0

    def save_batch():
        nonlocal batch

        if not batch:
            return

        chunk = pd.DataFrame(batch, columns=output_columns)

        # Première sauvegarde ou ajout à la suite.
        chunk.to_csv(
            output_path,
            mode="a" if output_path.exists() else "w",
            header=not output_path.exists(),
            index=False,
            float_format="%.12g",
        )

        batch = []

    for _, row in pending.iterrows():
        ecg_id = row["ecg_id"]

        try:
            signal = load_ecg(row)
            features = extract_features(signal)

            batch.append({
                "ecg_id": ecg_id,
                **features,
            })

        except Exception as exc:
            errors.append({
                "ecg_id": ecg_id,
                "error": str(exc),
            })

            print(f"[ERREUR] ECG {ecg_id} : {exc}")

        processed_new += 1

        if len(batch) >= 100:
            save_batch()

        if processed_new % 100 == 0:
            print(
                f"Progression : {processed_new}/{len(pending)} "
                f"ECG traités pendant cette exécution."
            )

    save_batch()

    # Journal des erreurs de cette exécution, toujours réécrit (même vide)
    # pour qu'un journal périmé ne survive pas à une exécution sans erreur.
    pd.DataFrame(errors, columns=["ecg_id", "error"]).to_csv(
        error_path,
        index=False,
        encoding="utf-8",
    )
    print(f"Journal des erreurs ({len(errors)}) : {error_path}")

    # Après une reprise, remettre les lignes dans l'ordre des métadonnées.
    if output_path.exists():
        if reorder_like_metadata(output_path, metadata["ecg_id"].tolist()):
            print("Lignes réordonnées selon les métadonnées.")

    # Vérification finale sur les données effectivement sauvegardées.
    if output_path.exists():
        final = pd.read_csv(output_path)

        if final["ecg_id"].duplicated().any():
            raise RuntimeError("Doublons ECG dans le résultat final.")

        if not np.isfinite(
            final[feature_columns].to_numpy(dtype=np.float64)
        ).all():
            raise RuntimeError("Valeurs non finies dans le résultat final.")

        if not set(final["ecg_id"]).issubset(set(metadata["ecg_id"])):
            raise RuntimeError("Des ECG ne correspondent pas à la partition.")

        expected_order = metadata.loc[
            metadata["ecg_id"].isin(final["ecg_id"]), "ecg_id"
        ].tolist()

        if final["ecg_id"].tolist() != expected_order:
            raise RuntimeError("Ordre des lignes différent des métadonnées.")

        missing_ids = set(metadata["ecg_id"]) - set(final["ecg_id"])

        print("\nBILAN")
        print(f"Fichier                 : {output_path}")
        print(f"Dimensions sauvegardées : {final.shape}")
        print(f"ECG manquants           : {len(missing_ids)}")
        print(f"Erreurs de cette exécution : {len(errors)}")

        if len(missing_ids) == 0:
            print("Partition complète : OUI")
        else:
            print(
                "Partition complète : NON. "
                "Corriger les erreurs puis relancer le même split."
            )

    return


def main():
    parser = argparse.ArgumentParser(
        description="Extraction complète des caractéristiques ECG."
    )

    parser.add_argument(
        "--split",
        choices=["train", "validation", "test", "all"],
        default="train",
        help="Partition à traiter (défaut : train).",
    )

    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Limite optionnelle pour un test contrôlé.",
    )

    args = parser.parse_args()

    if args.max_records is not None and args.max_records < 1:
        parser.error("--max-records doit être positif.")

    splits = (
        ["train", "validation", "test"]
        if args.split == "all"
        else [args.split]
    )

    for split_name in splits:
        process_split(split_name, args.max_records)


if __name__ == "__main__":
    main()
