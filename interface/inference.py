"""Inférence avec le protocole binaire, pour l'interface utilisateur.

Réutilise le pipeline de lecture des signaux (load_ecg), extraction des
120 caractéristiques (extract_features), prétraitement enregistré (P1) et
les cinq modèles binaires, avec leurs seuils réglés sur la validation.

Le test n'est jamais lu : seuls les noms de ses enregistrements servent à
refuser l'analyse d'un ECG du test (évaluation unique, phase 6). Les exemples
proposés viennent de la validation ; un ECG du train est signalé comme vu
pendant l'apprentissage.
"""

from pathlib import Path
import json
import re
import tempfile

import numpy as np
import pandas as pd
import torch
import wfdb
from torch import nn

from src.config import (
    DATA_ROOT,
    FEATURE_COLUMNS,
    LEADS,
    MODEL_FEATURE_COLUMNS,
    SPLITS,
    STAT_FEATURES,
    features_path,
    metadata_path,
)
from src.features.extract_statistical_features_full import extract_features, load_ecg
from src.models.mlp import count_parameters
from src.models.train_binary import (
    CLASS_NAMES as CLASSES,
    RUNS_DIR,
    SOURCE_CLASSES,
    make_model,
)
from src.preprocessing.preprocess import load_preprocessor


PROTOCOL_FILE = RUNS_DIR.parent / "protocol.json"
TEST_SUMMARY = RUNS_DIR.parent / "test_summary.json"

RECORD_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}")
DAT_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}\.dat")

HISTOGRAM_BINS = 40


class ProtocoleBinaire:
    """Prétraitement et modèles binaires, chargés une seule fois."""

    def __init__(self):
        self.protocol = json.loads(PROTOCOL_FILE.read_text(encoding="utf-8"))
        if self.protocol.get("task") != "binary":
            raise RuntimeError("Le protocole binaire n'est pas disponible.")

        self.preprocessor = load_preprocessor(self.protocol["preprocessing"])
        self.models = [self._load_run(run_id) for run_id in self.protocol["run_ids"]]

        # Noms d'enregistrement -> partition. Pour le test, seules les colonnes
        # d'identification sont lues (ni signaux, ni étiquettes).
        self.records = {}
        for split in SPLITS:
            columns = ["ecg_id", "filename_hr"] + (
                SOURCE_CLASSES if split != "test" else []
            )
            meta = pd.read_csv(metadata_path(split), usecols=columns)
            for row in meta.itertuples(index=False):
                if split == "test":
                    labels = None
                else:
                    pathological = any(getattr(row, c) for c in SOURCE_CLASSES[1:])
                    labels = {
                        CLASSES[0]: int(bool(row.NORM) and not pathological),
                        CLASSES[1]: int(pathological),
                    }
                self.records[Path(row.filename_hr).name] = {
                    "partition": split, "ecg_id": int(row.ecg_id), "etiquettes": labels,
                }

        self.validation = pd.read_csv(metadata_path("validation"))

    def _load_run(self, run_id):
        run_dir = RUNS_DIR / run_id
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        if (config["hidden_sizes"] != self.protocol["architecture"]["hidden_sizes"]
                or config["preprocessing"] != self.protocol["preprocessing"]):
            raise RuntimeError(f"{run_id} ne correspond pas au protocole binaire.")

        model = make_model(config)
        model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu"))
        model.eval()
        val_metrics = json.loads((run_dir / "val_metrics.json").read_text(encoding="utf-8"))
        threshold = float(self.protocol["thresholds"][run_id])
        return {
            "run_id": run_id,
            "seed": config["seed"],
            "model": model,
            "threshold": threshold,
            "thresholds": np.array([1 - threshold, threshold]),
            "history": pd.read_csv(run_dir / "history.csv"),
            "best_epoch": int(val_metrics["best_epoch"]),
            "epochs_run": int(val_metrics["epochs_run"]),
        }

    # ------------------------------------------------------------
    # Informations générales
    # ------------------------------------------------------------

    def informations(self):
        """Architecture, seuils, historiques et résultats publiés du test."""
        model = self.models[0]["model"]
        layers = [layer for layer in model.network if isinstance(layer, nn.Linear)]
        summary = json.loads(TEST_SUMMARY.read_text(encoding="utf-8"))
        global_precision_recall = precision_rappel_globaux(summary["per_seed"])
        return {
            "classes": CLASSES,
            "derivations": LEADS,
            "statistiques": STAT_FEATURES,
            "caracteristiques": FEATURE_COLUMNS,
            "entrees": MODEL_FEATURE_COLUMNS,
            "architecture": {
                "tailles": [layers[0].in_features] + [layer.out_features for layer in layers],
                "dropout": self.protocol["architecture"]["dropout"],
                "parametres": count_parameters(model),
                "parametres_par_couche": [
                    layer.weight.numel() + layer.bias.numel() for layer in layers
                ],
            },
            "entrainement": self.protocol["training"],
            "graines": [
                {
                    "graine": m["seed"],
                    "run_id": m["run_id"],
                    "seuils": m["thresholds"].tolist(),
                    "meilleure_epoque": m["best_epoch"],
                    "epoques": m["epochs_run"],
                    "historique": {
                        "epoch": m["history"]["epoch"].tolist(),
                        "train_loss": m["history"]["train_loss"].round(6).tolist(),
                        "val_loss": m["history"]["val_loss"].round(6).tolist(),
                        "val_macro_auroc": (
                            m["history"]["val_auroc_pathological"].round(6).tolist()
                        ),
                    },
                }
                for m in self.models
            ],
            "test": {
                "n_ecg": summary["n_ecg"],
                "n_patients": summary["n_patients"],
                "auroc_macro": summary["summary"]["auroc_macro"],
                "f1_macro": summary["summary"]["f1_macro"],
                "ic95": {
                    key: [summary["bootstrap"][key]["ci95_low"],
                          summary["bootstrap"][key]["ci95_high"]]
                    for key in ("auroc_macro", "f1_macro")
                },
                "auroc_par_classe": {
                    c: summary["per_class"][c]["auroc"]["mean"] for c in CLASSES
                },
                # Résultats enregistrés de l'évaluation unique (aucun recalcul).
                "par_classe": {
                    c: {key: summary["per_class"][c][key]
                        for key in ("auroc", "precision", "recall", "f1", "support")}
                    for c in CLASSES
                },
                "globales": {
                    **{key: summary["summary"][key] for key in ("f1_macro", "f1_micro")},
                    **resumer(global_precision_recall),
                },
                "par_graine": [
                    {
                        "graine": m["seed"],
                        "f1_macro": m["f1_macro"],
                        "f1_micro": m["f1_micro"],
                        **precision_rappel_globaux({run_id: m})[run_id],
                        "par_classe": {
                            c: {key: m["per_class"][c][key]
                                for key in ("precision", "recall", "f1", "support", "confusion_matrix")}
                            for c in CLASSES
                        },
                    }
                    for run_id, m in summary["per_seed"].items()
                ],
            },
        }

    def catalogue(self):
        """ECG de la validation proposés en exemple, avec leurs annotations."""
        return {
            "partition": "validation",
            "n_ecg": int(len(self.validation)),
            "donnees_brutes": (DATA_ROOT / f"{self.validation['filename_hr'].iloc[0]}.hea").is_file(),
            "ecg": [
                [
                    int(row.ecg_id),
                    [CLASSES[1] if any(getattr(row, c) for c in SOURCE_CLASSES[1:])
                     else CLASSES[0]],
                ]
                for row in self.validation.itertuples(index=False)
            ],
        }

    def poids(self, seed):
        """Poids et biais d'une graine (pour l'animation du réseau)."""
        entry = next((m for m in self.models if m["seed"] == seed), None)
        if entry is None:
            raise LookupError(f"Graine inconnue : {seed}")
        model = entry["model"]
        layers = [layer for layer in model.network if isinstance(layer, nn.Linear)]
        return {
            "graine": seed,
            "couches": [
                {
                    "W": np.round(layer.weight.detach().numpy().astype(np.float64), 7).tolist(),
                    "b": np.round(layer.bias.detach().numpy().astype(np.float64), 7).tolist(),
                }
                for layer in layers
            ],
        }

    def distributions(self):
        """Histogramme de chaque entrée sur le train (illustration de P1)."""
        train = pd.read_csv(features_path("train"), usecols=MODEL_FEATURE_COLUMNS)
        result = {}
        for column in MODEL_FEATURE_COLUMNS:
            values = train[column].to_numpy(dtype=np.float64)
            low, high = np.quantile(values, [0.005, 0.995])
            if high <= low:
                high = low + 1e-9
            counts, edges = np.histogram(values, bins=HISTOGRAM_BINS, range=(low, high))
            result[column] = {
                "bornes": [float(low), float(high)],
                "effectifs": counts.tolist(),
                "hors_bornes": [int((values < low).sum()), int((values > high).sum())],
            }
        return {"n_ecg": int(len(train)), "entrees": result}

    # ------------------------------------------------------------
    # Analyse d'un ECG
    # ------------------------------------------------------------

    def analyser_validation(self, ecg_id):
        rows = self.validation[self.validation["ecg_id"] == ecg_id]
        if rows.empty:
            raise LookupError(
                f"L'ECG {ecg_id} n'appartient pas à la partition de validation."
            )
        row = rows.iloc[0]
        signal = load_ecg(row)
        source = {
            "type": "base",
            "nom": Path(row["filename_hr"]).name,
            "ecg_id": int(ecg_id),
            "partition": "validation",
            "etiquettes": {
                CLASSES[0]: int(bool(row["NORM"]) and not any(row[c] for c in SOURCE_CLASSES[1:])),
                CLASSES[1]: int(any(row[c] for c in SOURCE_CLASSES[1:])),
            },
        }
        return self.analyser(signal, source)

    def analyser_fichiers(self, header_bytes, dat_bytes):
        """ECG importé au format WFDB (fichiers .hea et .dat)."""
        name, dat_name = parse_header(header_bytes)
        known = self.records.get(name)
        # Refus avant toute lecture du signal.
        if known and known["partition"] == "test":
            raise PermissionError(
                f"L'enregistrement {name} appartient à la partition de test (fold 6). "
                "Le protocole du projet n'autorise qu'une seule évaluation du test "
                "(phase 6) : l'interface refuse de l'analyser."
            )
        signal = read_uploaded_record(name, dat_name, header_bytes, dat_bytes)
        source = {"type": "fichier", "nom": name, "ecg_id": None,
                  "partition": None, "etiquettes": None}
        if known:
            source.update(known)
        return self.analyser(signal, source)

    def analyser(self, signal, source):
        features = extract_features(signal)
        x = np.array([[features[column] for column in MODEL_FEATURE_COLUMNS]])
        z = self.preprocessor.transform(x)
        if not np.isfinite(z).all():
            raise ValueError("Sorties du prétraitement non finies.")
        inputs = torch.as_tensor(np.asarray(z), dtype=torch.float32)

        seeds = []
        with torch.no_grad():
            for m in self.models:
                hidden, logits = forward_layers(m["model"], inputs)
                probabilities = torch.softmax(logits, dim=1).numpy()
                pathological = int(probabilities[0, 1] >= m["threshold"])
                decisions = np.array([1 - pathological, pathological], dtype=np.int8)
                seeds.append({
                    "graine": m["seed"],
                    "couches_cachees": [h[0].numpy().astype(np.float64).round(6).tolist()
                                        for h in hidden],
                    "logits": logits[0].numpy().astype(np.float64).tolist(),
                    "probabilites": probabilities[0].astype(np.float64).tolist(),
                    "seuils": m["thresholds"].tolist(),
                    "decisions": decisions.astype(int).tolist(),
                    "regle_vide": False,
                })

        return {
            "source": source,
            "signal": {
                "fs": 500,
                "derivations": LEADS,
                # Résolution de PTB-XL : 0,001 mV ; l'arrondi à 3 décimales est exact.
                "valeurs": np.round(signal.T, 3).tolist(),
            },
            "caracteristiques": {column: features[column] for column in FEATURE_COLUMNS},
            "entrees": {
                "brutes": x[0].tolist(),
                "moyennes": self.preprocessor.mean_.tolist(),
                "ecarts_types": self.preprocessor.scale_.tolist(),
                "normalisees": z[0].tolist(),
            },
            "graines": seeds,
            "accord": np.sum([s["decisions"] for s in seeds], axis=0).astype(int).tolist(),
        }


def precision_rappel_globaux(per_seed):
    """Précision et rappel macro et micro de chaque graine.

    Dérivés des métriques par classe et des matrices de confusion
    enregistrées : aucune prédiction n'est recalculée. Contrôle : le F1 macro
    et le F1 micro obtenus de la même façon doivent retrouver les valeurs
    enregistrées.
    """
    resultats = {}
    for run_id, m in per_seed.items():
        pc = m["per_class"]
        (_, fp), (fn, vp) = np.sum([pc[c]["confusion_matrix"] for c in CLASSES], axis=0)
        precision_micro, rappel_micro = vp / (vp + fp), vp / (vp + fn)
        f1_micro = 2 * precision_micro * rappel_micro / (precision_micro + rappel_micro)
        f1_macro = np.mean([pc[c]["f1"] for c in CLASSES])
        if abs(f1_micro - m["f1_micro"]) > 1e-12 or abs(f1_macro - m["f1_macro"]) > 1e-12:
            raise RuntimeError(f"{run_id} : F1 recalculés différents des valeurs enregistrées.")
        resultats[run_id] = {
            "precision_macro": float(np.mean([pc[c]["precision"] for c in CLASSES])),
            "recall_macro": float(np.mean([pc[c]["recall"] for c in CLASSES])),
            "precision_micro": float(precision_micro),
            "recall_micro": float(rappel_micro),
        }
    return resultats


def resumer(par_graine):
    """Moyenne et écart-type (ddof = 1) sur les graines, comme en phases 5 et 6."""
    cles = next(iter(par_graine.values()))
    return {
        key: {"mean": float(np.mean([v[key] for v in par_graine.values()])),
              "std": float(np.std([v[key] for v in par_graine.values()], ddof=1))}
        for key in cles
    }


def forward_layers(model, inputs):
    """Activations après chaque ReLU et logits (dropout inactif en évaluation)."""
    hidden = []
    x = inputs
    for layer in model.network:
        x = layer(x)
        if isinstance(layer, nn.ReLU):
            hidden.append(x)
    return hidden, x


def parse_header(header_bytes):
    """Nom de l'enregistrement et du fichier .dat désignés par l'en-tête WFDB."""
    try:
        header = header_bytes.decode("utf-8")
    except UnicodeDecodeError:
        header = header_bytes.decode("latin-1")

    lines = [line.split() for line in header.splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    try:
        name, n_signals = lines[0][0], int(lines[0][1])
        dat_names = {lines[1 + k][0] for k in range(n_signals)}
    except (IndexError, ValueError):
        raise ValueError("En-tête WFDB (.hea) illisible.") from None

    if not RECORD_NAME.fullmatch(name):
        raise ValueError(f"Nom d'enregistrement non pris en charge : {name!r}.")
    if len(dat_names) != 1 or not DAT_NAME.fullmatch(next(iter(dat_names))):
        raise ValueError("L'en-tête doit désigner un seul fichier de signal .dat.")
    return name, dat_names.pop()


def read_uploaded_record(name, dat_name, header_bytes, dat_bytes):
    """Écrit l'enregistrement dans un dossier temporaire et le lit avec load_ecg."""
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        (folder / f"{name}.hea").write_bytes(header_bytes)
        (folder / dat_name).write_bytes(dat_bytes)
        try:
            wfdb.rdrecord(str(folder / name))
        except Exception as error:  # noqa: BLE001 : message renvoyé à la page
            raise ValueError(
                "Enregistrement WFDB illisible : fichier .dat incomplet ou "
                f"incompatible avec l'en-tête ({error})."
            ) from None
        # Chemin absolu : resolve_record_path le conserve tel quel.
        return load_ecg(pd.Series({"filename_hr": str(folder / name)}))
