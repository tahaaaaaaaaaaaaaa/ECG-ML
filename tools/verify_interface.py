"""Vérifie la concordance du classifieur binaire et de l'interface (validation).

Le test n'est jamais lu. Pour chacun des ECG de validation, compare les
caractéristiques calculées depuis le signal brut, les entrées normalisées,
les probabilités et les décisions aux sorties du pipeline binaire enregistré.

  python tools/verify_interface.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from interface.inference import ProtocoleBinaire  # noqa: E402
from src.config import (  # noqa: E402
    FEATURE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    MODEL_INPUT_DIR,
    features_path,
)
from src.features.extract_statistical_features_full import load_ecg  # noqa: E402
from src.models.train_binary import CLASS_NAMES, binary_labels  # noqa: E402


TOLERANCE_FEATURES = 1e-9
TOLERANCE_PROBA = 1e-5
# CPU float32 can shift scores slightly between batch inference and one-ECG calls.
TOLERANCE_METRIQUE = 3e-6


def main():
    start = time.time()
    protocol = ProtocoleBinaire()
    meta = protocol.validation
    with np.load(MODEL_INPUT_DIR / "validation.npz", allow_pickle=False) as data:
        X_ref = data["X"]
        source_y = data["Y"]
        source_classes = data["class_names"].tolist()
    y = binary_labels(source_y, source_classes)
    features_ref = pd.read_csv(features_path("validation")).set_index("ecg_id")
    expected = protocol.protocol["validation"]

    count = len(meta)
    features = np.empty((count, len(FEATURE_COLUMNS)))
    inputs = np.empty((count, len(MODEL_FEATURE_COLUMNS)))
    probabilities = {m["seed"]: np.empty((count, len(CLASS_NAMES)))
                     for m in protocol.models}
    decisions = {m["seed"]: np.empty((count, len(CLASS_NAMES)), dtype=np.int8)
                 for m in protocol.models}
    for index, row in enumerate(meta.itertuples(index=False)):
        result = protocol.analyser(load_ecg(pd.Series(row._asdict())), {
            "type": "base", "nom": str(row.filename_hr), "ecg_id": int(row.ecg_id),
            "partition": "validation", "etiquettes": None,
        })
        features[index] = [result["caracteristiques"][column] for column in FEATURE_COLUMNS]
        inputs[index] = result["entrees"]["brutes"]
        for run in result["graines"]:
            probabilities[run["graine"]][index] = run["probabilites"]
            decisions[run["graine"]][index] = run["decisions"]
        if (index + 1) % 500 == 0:
            print(f"  {index + 1}/{count} ECG analysés")

    if not (meta["ecg_id"].to_numpy() == features_ref.index.to_numpy()).all():
        raise SystemExit("Ordre des ECG différent entre métadonnées et caractéristiques.")
    feature_values = features_ref[FEATURE_COLUMNS].to_numpy()
    feature_error = np.max(
        np.abs(features - feature_values) / np.maximum(np.abs(feature_values), 1e-12)
    )
    input_error = np.max(
        np.abs(inputs - X_ref) / np.maximum(np.abs(X_ref), 1e-12)
    )
    transformed = torch.as_tensor(
        protocol.preprocessor.transform(X_ref), dtype=torch.float32
    )

    rows = []
    for model_data in protocol.models:
        seed = model_data["seed"]
        with torch.no_grad():
            reference = torch.softmax(model_data["model"](transformed), dim=1).numpy()
        interface_probabilities = probabilities[seed]
        interface_decisions = decisions[seed]
        predicted = np.argmax(interface_decisions, axis=1)
        stored = expected[model_data["run_id"]]
        metrics = {
            "auroc_macro": float(roc_auc_score(y, interface_probabilities[:, 1])),
            "f1_macro": float(f1_score(y, predicted, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y, predicted)),
        }
        rows.append({
            "seed": seed,
            "probability_error": float(np.max(np.abs(interface_probabilities - reference))),
            "different_decisions": int(
                (interface_decisions[:, 1] !=
                 (interface_probabilities[:, 1] >= model_data["threshold"])).sum()
            ),
            "metric_error": max(abs(metrics[key] - stored[key]) for key in metrics),
            "auroc": metrics["auroc_macro"],
        })

    print(f"\nVÉRIFICATION DE L'INTERFACE ({count} ECG de validation, "
          f"{time.time() - start:.0f} s)")
    print(f"  caractéristiques : écart relatif max {feature_error:.1e}")
    print(f"  entrées : écart relatif max {input_error:.1e}")
    passed = feature_error <= TOLERANCE_FEATURES and input_error <= TOLERANCE_FEATURES
    for row in rows:
        ok = (
            row["probability_error"] <= TOLERANCE_PROBA
            and row["different_decisions"] == 0
            and row["metric_error"] <= TOLERANCE_METRIQUE
        )
        passed &= ok
        print(
            f"  graine {row['seed']} : probabilité max {row['probability_error']:.1e} ; "
            f"décisions différentes {row['different_decisions']} ; "
            f"écart métriques {row['metric_error']:.1e} (AUROC {row['auroc']:.4f}) "
            f"{'OK' if ok else 'ÉCHEC'}"
        )
    print("RÉSULTAT :", "identique au pipeline binaire" if passed else "ÉCHEC")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
