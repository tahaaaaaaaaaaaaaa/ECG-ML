
"""Construction et vérification des partitions figées train / validation / test.

Par défaut, le script reconstruit les partitions en mémoire et les compare
aux fichiers existants de data/processed/splits, sans rien écrire : les
partitions sont figées et ne doivent pas être régénérées. L'écriture n'a
lieu que si les fichiers sont absents, ou avec l'option --overwrite.
"""

from ast import literal_eval
from pathlib import Path
import argparse
import sys

import pandas as pd

# Permet aussi l'exécution directe du fichier (bouton « Run » de VS Code).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import (  # noqa: E402
    CLASSES as TARGET_CLASSES,
    DATA_ROOT,
    DATABASE_FILE,
    METADATA_COLUMNS,
    SPLITS,
    SPLITS_DIR as SPLITS_ROOT,
    STATEMENTS_FILE,
)


REQUIRED_DATABASE_COLUMNS = {
    "ecg_id",
    "patient_id",
    "strat_fold",
    "filename_hr",
    "scp_codes",
}


# ============================================================
# 1. Vérification des fichiers et lecture des métadonnées
# ============================================================

def load_metadata():
    for file_path in [DATABASE_FILE, STATEMENTS_FILE]:
        if not file_path.is_file():
            raise FileNotFoundError(
                f"Fichier introuvable : {file_path}"
            )

    if not (DATA_ROOT / "records500").is_dir():
        raise FileNotFoundError(
            f"Dossier records500 introuvable : {DATA_ROOT / 'records500'}"
        )

    database = pd.read_csv(DATABASE_FILE)
    statements = pd.read_csv(STATEMENTS_FILE, index_col=0)

    missing_columns = REQUIRED_DATABASE_COLUMNS - set(database.columns)

    if missing_columns:
        raise ValueError(
            f"Colonnes manquantes dans ptbxl_database.csv : "
            f"{sorted(missing_columns)}"
        )

    if "diagnostic" not in statements.columns:
        raise ValueError(
            "La colonne 'diagnostic' est absente de scp_statements.csv."
        )

    if "diagnostic_class" not in statements.columns:
        raise ValueError(
            "La colonne 'diagnostic_class' est absente de scp_statements.csv."
        )

    print("=" * 65)
    print("1. METADONNEES")
    print("=" * 65)
    print(f"Nombre total d'ECG : {len(database)}")
    print(f"Nombre de patients distincts : {database['patient_id'].nunique()}")
    print(f"Colonnes disponibles : {len(database.columns)}")
    print(f"Nombre de codes SCP : {len(statements)}")

    print("\nEffectifs bruts par fold :")
    print(
        database["strat_fold"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    return database, statements


# ============================================================
# 2. Codes SCP diagnostiques et cinq labels binaires
# ============================================================

def build_scp_to_class(statements):
    # Seuls les codes SCP indiqués comme diagnostiques sont retenus.
    # Le code doit également appartenir à l'une des cinq classes cibles.
    diagnostic_statements = statements.loc[
        statements["diagnostic"].eq(1)
    ].copy()

    diagnostic_statements = diagnostic_statements.loc[
        diagnostic_statements["diagnostic_class"].isin(TARGET_CLASSES)
    ]

    scp_to_class = diagnostic_statements["diagnostic_class"].to_dict()

    if not scp_to_class:
        raise ValueError(
            "Aucun code SCP diagnostique associé aux classes cibles."
        )

    print("\nCodes SCP diagnostiques associés aux classes cibles :")
    for target in TARGET_CLASSES:
        count = sum(value == target for value in scp_to_class.values())
        print(f"  {target}: {count} codes SCP")

    return scp_to_class


def parse_scp_codes(value):
    """Convertit la représentation textuelle des codes SCP en dictionnaire."""
    if pd.isna(value):
        return {}

    if isinstance(value, dict):
        return value

    parsed = literal_eval(value)

    if not isinstance(parsed, dict):
        raise ValueError(
            f"Format inattendu pour scp_codes : {type(parsed)}"
        )

    return parsed


def extract_target_labels(value, scp_to_class):
    """Retourne les cinq labels diagnostiques binaires.

    Convention : tous les codes présents comptent, quelle que soit
    leur vraisemblance (likelihood), y compris 0.
    """
    codes = parse_scp_codes(value)

    present_classes = {
        scp_to_class[code]
        for code in codes
        if code in scp_to_class
    }

    return {
        target: int(target in present_classes)
        for target in TARGET_CLASSES
    }


def build_labels(database, scp_to_class):
    label_records = database["scp_codes"].apply(
        extract_target_labels, scp_to_class=scp_to_class
    )
    labels = pd.DataFrame(label_records.tolist(), index=database.index)

    # Garder un ordre de colonnes strictement défini.
    labels = labels[TARGET_CLASSES].astype("int8")

    data = pd.concat(
        [database.reset_index(drop=True), labels.reset_index(drop=True)],
        axis=1,
    )

    print("\n" + "=" * 65)
    print("2. CONSTRUCTION DES LABELS")
    print("=" * 65)

    print("\nEffectifs par classe avant sélection des folds :")
    print(data[TARGET_CLASSES].sum().to_string())

    no_target_mask = data[TARGET_CLASSES].sum(axis=1).eq(0)

    print(f"\nECG sans aucune classe cible : {no_target_mask.sum()}")
    print(f"ECG avec au moins une classe cible : {(~no_target_mask).sum()}")

    # Les ECG sans classe cible ne sont pas utilisables pour cette tâche
    # de classification supervisée à cinq sorties.
    return data.loc[~no_target_mask].copy()


# ============================================================
# 3. Partitions
# ============================================================

def build_splits(eligible):
    split_definitions = {
        name: definition["folds"] for name, definition in SPLITS.items()
    }

    selected_folds = [
        fold for folds in split_definitions.values() for fold in folds
    ]

    if len(selected_folds) != len(set(selected_folds)):
        raise ValueError("Un fold est présent dans plusieurs partitions.")

    splits = {}

    for split_name, folds in split_definitions.items():
        split = eligible.loc[
            eligible["strat_fold"].isin(folds)
        ].copy()

        if split.empty:
            raise ValueError(
                f"La partition {split_name} est vide."
            )

        splits[split_name] = split.reset_index(drop=True)

    return splits, split_definitions


def check_patient_overlap(splits):
    patient_sets = {
        name: set(split["patient_id"].dropna().unique())
        for name, split in splits.items()
    }

    for left, right in [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]:
        overlap = patient_sets[left] & patient_sets[right]

        print(
            f"Patients communs {left}/{right} : {len(overlap)}"
        )

        if overlap:
            raise ValueError(
                f"Fuite de patients entre {left} et {right}."
            )


def record_files_exist(relative_path):
    """Vérifie la présence des fichiers WFDB .hea et .dat."""
    if not isinstance(relative_path, str) or not relative_path.strip():
        return False

    # pathlib accepte les séparateurs « / » sur tous les systèmes.
    base_path = DATA_ROOT / relative_path

    return (
        base_path.with_suffix(".hea").is_file()
        and base_path.with_suffix(".dat").is_file()
    )


def check_record_files(splits):
    print("\n" + "=" * 65)
    print("3. VERIFICATION DES FICHIERS WFDB")
    print("=" * 65)

    for split_name, split in splits.items():
        paths_ok = split["filename_hr"].apply(record_files_exist)
        missing_count = int((~paths_ok).sum())

        print(
            f"{split_name}: {len(split)} ECG éligibles, "
            f"{missing_count} enregistrements introuvables"
        )

        if missing_count:
            print(
                split.loc[~paths_ok, ["ecg_id", "filename_hr"]]
                .head(10)
                .to_string(index=False)
            )

            raise FileNotFoundError(
                f"Des enregistrements 500 Hz manquent dans {split_name}."
            )


# ============================================================
# 4. Résumé, comparaison avec les fichiers figés, sauvegarde
# ============================================================

def build_outputs(splits, split_definitions):
    """Contenu CSV attendu de chaque fichier de sortie."""
    outputs = {}
    output_columns = METADATA_COLUMNS + TARGET_CLASSES

    for split_name, split in splits.items():
        outputs[SPLITS_ROOT / f"{split_name}_metadata.csv"] = (
            split[output_columns].to_csv(index=False)
        )

        print("\n" + "=" * 65)
        print(f"PARTITION : {split_name.upper()}")
        print("=" * 65)
        print(f"Folds : {split_definitions[split_name]}")
        print(f"Nombre d'ECG : {len(split)}")
        print(f"Nombre de patients : {split['patient_id'].nunique()}")
        print("\nEffectifs positifs par classe :")
        print(split[TARGET_CLASSES].sum().to_string())
        print("\nNombre de labels positifs par ECG :")
        print(
            split[TARGET_CLASSES]
            .sum(axis=1)
            .value_counts()
            .sort_index()
            .to_string()
        )

    # Résumé global des données sélectionnées
    summary = pd.DataFrame(
        {
            name: {
                "folds": ",".join(map(str, split_definitions[name])),
                "n_ecg": len(split),
                "n_patients": split["patient_id"].nunique(),
                **{
                    f"n_{target}": int(split[target].sum())
                    for target in TARGET_CLASSES
                },
            }
            for name, split in splits.items()
        }
    ).T

    outputs[SPLITS_ROOT / "split_summary.csv"] = summary.to_csv(
        index=True, index_label="partition"
    )

    return outputs


def normalize(text):
    """Neutralise les fins de ligne (CRLF sous Windows, LF ailleurs)."""
    return text.replace("\r\n", "\n")


def verify_or_write(outputs, overwrite):
    print("\n" + "=" * 65)
    print("4. COMPARAISON AVEC LES PARTITIONS FIGÉES")
    print("=" * 65)

    existing = [path for path in outputs if path.exists()]

    if existing and not overwrite:
        all_identical = True

        for path, content in outputs.items():
            if not path.exists():
                print(f"[ABSENT]    {path.name}")
                all_identical = False
                continue

            identical = (
                normalize(path.read_text(encoding="utf-8"))
                == normalize(content)
            )
            all_identical &= identical
            print(f"[{'IDENTIQUE' if identical else 'DIFFÉRENT'}] {path.name}")

        if not all_identical:
            raise SystemExit(
                "Les partitions reconstruites diffèrent des fichiers figés. "
                "Aucun fichier n'a été modifié."
            )

        print("\nPartitions figées reproduites à l'identique. "
              "Aucun fichier modifié.")
        return

    SPLITS_ROOT.mkdir(parents=True, exist_ok=True)

    for path, content in outputs.items():
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
        print(f"Fichier enregistré : {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruit et vérifie les partitions figées."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Réécrit les fichiers existants (à éviter : partitions figées).",
    )
    args = parser.parse_args()

    database, statements = load_metadata()
    scp_to_class = build_scp_to_class(statements)
    eligible = build_labels(database, scp_to_class)
    splits, split_definitions = build_splits(eligible)
    check_patient_overlap(splits)
    check_record_files(splits)
    outputs = build_outputs(splits, split_definitions)
    verify_or_write(outputs, args.overwrite)

    print("\nVérifications terminées.")


if __name__ == "__main__":
    main()
