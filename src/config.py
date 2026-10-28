"""Configuration centralisée du classifieur binaire ECG PTB-XL.

Tous les scripts importent ici leurs chemins, étiquettes source, caractéristiques
et partitions afin d'éviter les duplications.

Exécution recommandée depuis la racine du projet, sous forme de module :

    .venv\\Scripts\\python.exe -m src.features.validate_statistical_features
"""

from pathlib import Path


# ============================================================
# 1. Chemins
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = ROOT / "data" / "raw" / "ptb-xl"
DATABASE_FILE = DATA_ROOT / "ptbxl_database.csv"
STATEMENTS_FILE = DATA_ROOT / "scp_statements.csv"

PROCESSED_ROOT = ROOT / "data" / "processed"
SPLITS_DIR = PROCESSED_ROOT / "splits"
FEATURES_DIR = PROCESSED_ROOT / "features_statistical"
MODEL_INPUT_DIR = PROCESSED_ROOT / "model_input"

RESULTS_DIR = ROOT / "results"
PREPROCESSING_DIR = RESULTS_DIR / "binary" / "preprocessing"


# ============================================================
# 2. Étiquettes source utilisées pour créer les classes binaires
# ============================================================

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


# ============================================================
# 3. Partitions figées (ne jamais les modifier)
# ============================================================

SPLITS = {
    "train": {"folds": [1, 2, 3, 4, 5], "n_ecg": 10692},
    "validation": {"folds": [10], "n_ecg": 2158},
    "test": {"folds": [6], "n_ecg": 2129},
}

METADATA_COLUMNS = ["ecg_id", "patient_id", "strat_fold", "filename_hr"]


def metadata_path(split_name, splits_dir=SPLITS_DIR):
    return Path(splits_dir) / f"{split_name}_metadata.csv"


def features_path(split_name, features_dir=FEATURES_DIR):
    return Path(features_dir) / f"{split_name}_features.csv"


# ============================================================
# 4. Signaux et caractéristiques statistiques
# ============================================================

FS = 500
N_SAMPLES = 5000

# Notation clinique usuelle ; les en-têtes WFDB écrivent AVR, AVL, AVF
# (la comparaison avec les en-têtes est insensible à la casse).
LEADS = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]

STAT_FEATURES = [
    "mean", "std", "rms", "median", "iqr",
    "min", "max",
    "skewness", "kurtosis", "line_length",
]

FEATURE_COLUMNS = [
    f"{lead}_{feature}"
    for lead in LEADS
    for feature in STAT_FEATURES
]

MODEL_FEATURE_COLUMNS = FEATURE_COLUMNS
