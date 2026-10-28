"""Entraînement et évaluation du classifieur binaire normal/pathologique."""

from copy import deepcopy
from pathlib import Path
import argparse
import json
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import MODEL_FEATURE_COLUMNS, MODEL_INPUT_DIR, PREPROCESSING_DIR
from src.models.mlp import MLP


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "binary.json"
OUTPUT_DIR = ROOT / "results" / "binary"
RUNS_DIR = OUTPUT_DIR / "runs"
CLASS_NAMES = ["NORMAL", "PATHOLOGICAL"]
SOURCE_CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def binary_labels(targets, class_names):
    """Encode pathology as positive; pathology takes precedence over NORM."""
    if class_names != SOURCE_CLASSES:
        raise ValueError(f"Étiquettes source inattendues : {class_names}")
    targets = np.asarray(targets)
    if targets.ndim != 2 or targets.shape[1] != len(SOURCE_CLASSES):
        raise ValueError(f"Dimensions d'étiquettes inattendues : {targets.shape}")
    if not np.isin(targets, (0, 1)).all():
        raise ValueError("Les étiquettes source doivent être binaires.")

    pathology = targets[:, 1:].any(axis=1)
    normal_only = (targets[:, 0] == 1) & ~pathology
    if not np.all(pathology | normal_only):
        raise ValueError("Un ECG ne peut être ni normal ni pathologique.")
    return np.where(pathology, 1, 0).astype(np.int64)


def load_split(split):
    with np.load(MODEL_INPUT_DIR / f"{split}.npz", allow_pickle=False) as data:
        if data["feature_names"].tolist() != MODEL_FEATURE_COLUMNS:
            raise ValueError(f"{split} : caractéristiques inattendues.")
        labels = binary_labels(data["Y"], data["class_names"].tolist())
        return data["X"], labels


def set_reproducibility(seed, num_threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(num_threads)


def make_model(config):
    return MLP(
        n_inputs=len(MODEL_FEATURE_COLUMNS),
        hidden_sizes=config["hidden_sizes"],
        n_outputs=2,
        dropout=config["dropout"],
    )


def probabilities(model, X):
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(X), dim=1).cpu().numpy()


def binary_metrics(y_true, scores, threshold):
    predicted = (scores[:, 1] >= threshold).astype(np.int64)
    truth = np.eye(2, dtype=np.int8)[y_true]
    prediction = np.eye(2, dtype=np.int8)[predicted]
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, prediction, average=None, zero_division=0
    )
    matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
    per_class = {}
    for index, name in enumerate(CLASS_NAMES):
        true_positive = int(matrix[index, index])
        false_positive = int(matrix[:, index].sum() - true_positive)
        false_negative = int(matrix[index, :].sum() - true_positive)
        true_negative = int(matrix.sum() - true_positive - false_positive - false_negative)
        per_class[name] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
            "confusion_matrix": [
                [true_negative, false_positive],
                [false_negative, true_positive],
            ],
            "auroc": float(roc_auc_score(truth[:, index], scores[:, index])),
            "average_precision": float(
                average_precision_score(truth[:, index], scores[:, index])
            ),
        }
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "auroc_macro": float(np.mean([v["auroc"] for v in per_class.values()])),
        "average_precision_macro": float(
            np.mean([v["average_precision"] for v in per_class.values()])
        ),
        "f1_macro": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, predicted, average="micro", zero_division=0)),
        "per_class": per_class,
    }


def run_id(config):
    return f"{config['name']}_seed{config['seed']}"


def train_run(config):
    identifier = run_id(config)
    run_dir = RUNS_DIR / identifier
    if run_dir.exists():
        raise FileExistsError(f"{run_dir} existe déjà : une exécution n'est jamais écrasée.")

    set_reproducibility(config["seed"], config["num_threads"])
    import joblib

    scaler = joblib.load(PREPROCESSING_DIR / f"{config['preprocessing']}.joblib")
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("validation")
    X_train = torch.as_tensor(scaler.transform(X_train), dtype=torch.float32)
    X_val = torch.as_tensor(scaler.transform(X_val), dtype=torch.float32)
    y_train_tensor = torch.as_tensor(y_train, dtype=torch.long)
    y_val_tensor = torch.as_tensor(y_val, dtype=torch.long)

    model = make_model(config)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    loader = DataLoader(
        TensorDataset(X_train, y_train_tensor),
        batch_size=config["batch_size"], shuffle=True,
        generator=torch.Generator().manual_seed(config["seed"]),
    )

    best_auroc, best_epoch, best_state = -np.inf, 0, None
    stale_epochs, history = 0, []
    start = time.perf_counter()
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)

        val_scores = probabilities(model, X_val)
        with torch.no_grad():
            val_loss = float(criterion(model(X_val), y_val_tensor))
        val_auroc = float(roc_auc_score(y_val, val_scores[:, 1]))
        history.append({
            "epoch": epoch,
            "train_loss": total_loss / len(X_train),
            "val_loss": val_loss,
            "val_auroc_pathological": val_auroc,
        })
        if val_auroc > best_auroc:
            best_auroc, best_epoch = val_auroc, epoch
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config["patience"]:
                break

    duration = time.perf_counter() - start
    model.load_state_dict(best_state)
    val_scores = probabilities(model, X_val)
    thresholds = np.round(np.arange(1, 100) / 100, 2)
    f1s = [
        f1_score(y_val, val_scores[:, 1] >= threshold, zero_division=0)
        for threshold in thresholds
    ]
    threshold = float(thresholds[int(np.argmax(f1s))])
    metrics = binary_metrics(y_val, val_scores, threshold)
    run_dir.mkdir(parents=True)
    torch.save(model.state_dict(), run_dir / "model.pt")
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    (run_dir / "config.json").write_text(
        json.dumps({**config, "run_id": identifier}, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "val_metrics.json").write_text(
        json.dumps({
            "run_id": identifier, "best_epoch": best_epoch,
            "epochs_run": len(history), "duration_s": round(duration, 1),
            "threshold": threshold, **metrics,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"{identifier}: époque {best_epoch}, AUROC pathologique "
          f"{best_auroc:.4f}, seuil {threshold:.2f}, "
          f"F1 pathologique {metrics['per_class']['PATHOLOGICAL']['f1']:.4f}, "
          f"{duration:.1f} s", flush=True)
    return identifier, threshold, metrics


def train():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_ids, thresholds, validations = [], {}, {}
    for seed in config["seeds"]:
        current = {**config["training"], "name": config["name"], "seed": seed}
        identifier, threshold, metrics = train_run(current)
        run_ids.append(identifier)
        thresholds[identifier] = threshold
        validations[identifier] = metrics

    protocol = {
        "status": "trained_validation_tuned",
        "task": "binary",
        "classes": CLASS_NAMES,
        "source_classes": SOURCE_CLASSES,
        "label_rule": "PATHOLOGICAL si MI, STTC, CD ou HYP est présent; sinon NORMAL si NORM est présent.",
        "architecture": {
            "hidden_sizes": config["training"]["hidden_sizes"],
            "dropout": config["training"]["dropout"],
            "n_outputs": 2,
            "output": "softmax",
        },
        "preprocessing": config["training"]["preprocessing"],
        "training": config["training"],
        "seeds": config["seeds"],
        "run_ids": run_ids,
        "thresholds": thresholds,
        "validation": validations,
    }
    (OUTPUT_DIR / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Protocole binaire enregistré : {OUTPUT_DIR / 'protocol.json'}")


def evaluate_test():
    protocol_path = OUTPUT_DIR / "protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError("Entraînez d'abord le modèle binaire.")
    output_path = OUTPUT_DIR / "test_summary.json"
    if output_path.exists():
        raise FileExistsError(f"{output_path} existe déjà : test déjà évalué.")

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    import joblib

    scaler = joblib.load(PREPROCESSING_DIR / f"{protocol['preprocessing']}.joblib")
    X_test, y_test = load_split("test")
    with np.load(MODEL_INPUT_DIR / "test.npz", allow_pickle=False) as test_data:
        patient_ids = test_data["patient_id"]
    X_test = torch.as_tensor(scaler.transform(X_test), dtype=torch.float32)
    results, all_scores, all_predictions = {}, {}, {}
    for identifier in protocol["run_ids"]:
        run_dir = RUNS_DIR / identifier
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        model = make_model(config)
        model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu"))
        scores = probabilities(model, X_test)
        threshold = protocol["thresholds"][identifier]
        predicted = (scores[:, 1] >= threshold).astype(np.int64)
        results[identifier] = {
            "seed": config["seed"], "threshold": threshold,
            **binary_metrics(y_test, scores, threshold),
        }
        all_scores[identifier] = scores
        all_predictions[identifier] = predicted

    keys = ("auroc_macro", "average_precision_macro", "f1_macro", "f1_micro", "accuracy")
    summary = {
        key: {
            "mean": float(np.mean([result[key] for result in results.values()])),
            "std": float(np.std([result[key] for result in results.values()], ddof=1)),
        }
        for key in keys
    }
    per_class = {
        name: {
            metric: {
                "mean": float(np.mean([result["per_class"][name][metric]
                                       for result in results.values()])),
                "std": float(np.std([result["per_class"][name][metric]
                                     for result in results.values()], ddof=1)),
            }
            for metric in (
                "auroc", "average_precision", "precision", "recall", "f1", "support"
            )
        }
        for name in CLASS_NAMES
    }
    rng = np.random.default_rng(42)
    patients, inverse = np.unique(patient_ids, return_inverse=True)
    rows_by_patient = [np.flatnonzero(inverse == i) for i in range(len(patients))]
    bootstrap = []
    for _ in range(1000):
        drawn = rng.integers(0, len(patients), size=len(patients))
        rows = np.concatenate([rows_by_patient[index] for index in drawn])
        if np.unique(y_test[rows]).size < 2:
            continue
        run_aurocs, run_f1s = [], []
        for identifier in protocol["run_ids"]:
            run_aurocs.append(roc_auc_score(y_test[rows], all_scores[identifier][rows, 1]))
            run_f1s.append(f1_score(
                y_test[rows], all_predictions[identifier][rows],
                average="macro", zero_division=0,
            ))
        bootstrap.append({
            "auroc_macro": float(np.mean(run_aurocs)),
            "f1_macro": float(np.mean(run_f1s)),
        })
    bootstrap_summary = {
        key: {
            "ci95_low": float(np.percentile([sample[key] for sample in bootstrap], 2.5)),
            "ci95_high": float(np.percentile([sample[key] for sample in bootstrap], 97.5)),
        }
        for key in ("auroc_macro", "f1_macro")
    }
    report = {
        "split": "test",
        "n_ecg": len(y_test),
        "n_patients": int(len(patients)),
        "classes": CLASS_NAMES,
        "summary": summary,
        "per_class": per_class,
        "per_seed": results,
        "bootstrap": bootstrap_summary,
    }
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Test binaire : AUROC macro {summary['auroc_macro']['mean']:.4f} ± "
          f"{summary['auroc_macro']['std']:.4f} ; F1 macro "
          f"{summary['f1_macro']['mean']:.4f} ± {summary['f1_macro']['std']:.4f}")
    for name in CLASS_NAMES:
        metrics = per_class[name]
        print(f"  {name:12} AUROC {metrics['auroc']['mean']:.4f} ; "
              f"F1 {metrics['f1']['mean']:.4f} ; "
              f"rappel {metrics['recall']['mean']:.4f} ; "
              f"précision {metrics['precision']['mean']:.4f}")


def export_test_roc():
    """Courbes ROC du test pour l'interface, qui ne lit jamais les données du test."""
    summary_path = OUTPUT_DIR / "test_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError("Évaluez d'abord le test (--evaluate-test).")
    output_path = OUTPUT_DIR / "test_roc.json"
    if output_path.exists():
        raise FileExistsError(f"{output_path} existe déjà : courbes déjà exportées.")

    protocol = json.loads((OUTPUT_DIR / "protocol.json").read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    import joblib

    scaler = joblib.load(PREPROCESSING_DIR / f"{protocol['preprocessing']}.joblib")
    X_test, y_test = load_split("test")
    X_test = torch.as_tensor(scaler.transform(X_test), dtype=torch.float32)
    seeds = {}
    for identifier in protocol["run_ids"]:
        run_dir = RUNS_DIR / identifier
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        model = make_model(config)
        model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu"))
        scores = probabilities(model, X_test)[:, 1]
        # Mêmes modèles, mêmes données que l'évaluation unique : rien n'est réglé ici.
        auroc = float(roc_auc_score(y_test, scores))
        if not np.isclose(auroc, summary["per_seed"][identifier]["auroc_macro"]):
            raise ValueError(f"{identifier} : AUROC différente de test_summary.json.")
        fpr, tpr, _ = roc_curve(y_test, scores)
        threshold = protocol["thresholds"][identifier]
        predicted = scores >= threshold
        seeds[str(config["seed"])] = {
            "run_id": identifier,
            "auroc": auroc,
            "threshold": threshold,
            "operating_point": {
                "fpr": float(predicted[y_test == 0].mean()),
                "tpr": float(predicted[y_test == 1].mean()),
            },
            "fpr": np.round(fpr, 5).tolist(),
            "tpr": np.round(tpr, 5).tolist(),
        }
    report = {
        "split": "test",
        "n_ecg": int(len(y_test)),
        "n_normal": int((y_test == 0).sum()),
        "n_pathological": int((y_test == 1).sum()),
        "seeds": seeds,
    }
    output_path.write_text(
        json.dumps(report, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Courbes ROC du test enregistrées : {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate-test", action="store_true",
                        help="Évalue une seule fois les modèles enregistrés sur le test.")
    parser.add_argument("--export-roc", action="store_true",
                        help="Exporte les courbes ROC du test pour l'interface.")
    args = parser.parse_args()
    if args.export_roc:
        export_test_roc()
    elif args.evaluate_test:
        evaluate_test()
    else:
        train()


if __name__ == "__main__":
    main()
