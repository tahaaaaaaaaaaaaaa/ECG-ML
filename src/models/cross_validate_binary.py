"""Validation croisée imbriquée du MLP binaire sur les folds PTB-XL 1 à 5."""

from copy import deepcopy
from pathlib import Path
import json
import time
import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import MODEL_FEATURE_COLUMNS, MODEL_INPUT_DIR, SPLITS_DIR
from src.models.train_binary import (
    OUTPUT_DIR,
    RUNS_DIR,
    SOURCE_CLASSES,
    binary_labels,
    binary_metrics,
    make_model,
    probabilities,
    set_reproducibility,
)


CROSS_VALIDATION_DIR = OUTPUT_DIR / "cross_validation"
OUTER_FOLDS = tuple(range(1, 6))
THRESHOLDS = np.round(np.arange(1, 100) / 100, 2)
METRICS = ("auroc_macro", "average_precision_macro", "accuracy", "f1_macro")


def load_training_data():
    with np.load(MODEL_INPUT_DIR / "train.npz", allow_pickle=False) as data:
        if data["feature_names"].tolist() != MODEL_FEATURE_COLUMNS:
            raise ValueError("Colonnes de caractéristiques train inattendues.")
        if data["class_names"].tolist() != SOURCE_CLASSES:
            raise ValueError("Classes sources train inattendues.")
        X = data["X"]
        y = binary_labels(data["Y"], data["class_names"].tolist())
        ecg_ids = data["ecg_id"]
        patient_ids = data["patient_id"]

    metadata = pd.read_csv(SPLITS_DIR / "train_metadata.csv")
    if not np.array_equal(metadata["ecg_id"].to_numpy(), ecg_ids):
        raise ValueError("Les métadonnées train ne sont pas alignées sur la matrice.")
    if not np.array_equal(metadata["patient_id"].to_numpy(), patient_ids):
        raise ValueError("Les patients des métadonnées et de la matrice diffèrent.")

    folds = metadata["strat_fold"].to_numpy(dtype=np.int64)
    if set(np.unique(folds)) != set(OUTER_FOLDS):
        raise ValueError(f"Folds train inattendus : {sorted(np.unique(folds))}")
    patient_fold_counts = metadata.groupby("patient_id")["strat_fold"].nunique()
    if (patient_fold_counts != 1).any():
        raise ValueError("Un patient apparaît dans plusieurs folds train.")
    return X, y, patient_ids, folds


def select_seeds():
    protocol = json.loads((OUTPUT_DIR / "protocol.json").read_text(encoding="utf-8"))
    ranked = []
    for run_id, metrics in protocol["validation"].items():
        seed = int(run_id.rsplit("seed", 1)[1])
        score = metrics["per_class"]["PATHOLOGICAL"]["auroc"]
        ranked.append((float(score), seed, run_id))
    if not ranked:
        raise ValueError("Aucune graine présente dans le protocole.")
    selected = sorted(ranked, key=lambda item: item[1])
    return [
        {"seed": seed, "run_id": run_id, "validation_auroc_pathological": score}
        for score, seed, run_id in selected
    ]


def fit_inner_model(config, seed, X_train, y_train, X_val, y_val):
    set_reproducibility(seed, config["num_threads"])
    X_train_tensor = torch.as_tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.as_tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.as_tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.as_tensor(y_val, dtype=torch.long)

    model = make_model(config)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=config["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    best_auc, best_epoch, best_state = -np.inf, 0, None
    stale_epochs, history = 0, []
    start = time.perf_counter()
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

        val_scores = probabilities(model, X_val_tensor)
        train_scores = probabilities(model, X_train_tensor)
        val_auc = float(roc_auc_score(y_val, val_scores[:, 1]))
        with torch.no_grad():
            train_loss = float(criterion(model(X_train_tensor), y_train_tensor))
            val_loss = float(criterion(model(X_val_tensor), y_val_tensor))
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": float(
                np.mean(train_scores.argmax(axis=1) == y_train)
            ),
            "val_accuracy": float(
                np.mean(val_scores.argmax(axis=1) == y_val)
            ),
            "val_auroc_pathological": val_auc,
        })
        if val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config["patience"]:
                break

    if best_state is None:
        raise RuntimeError("Aucun état de modèle valide n'a été sélectionné.")
    model.load_state_dict(best_state)
    duration = time.perf_counter() - start
    scores = probabilities(model, X_val_tensor)
    return model, scores, best_epoch, history, duration


def fit_fixed_epochs(config, seed, X_train, y_train, epochs):
    set_reproducibility(seed, config["num_threads"])
    X_train_tensor = torch.as_tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.as_tensor(y_train, dtype=torch.long)
    model = make_model(config)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=config["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
    return model


def inner_threshold(y_true, scores):
    f1_values = [
        f1_score(y_true, scores[:, 1] >= threshold, zero_division=0)
        for threshold in THRESHOLDS
    ]
    return float(THRESHOLDS[int(np.argmax(f1_values))])


def run_fold(config, base_seed, outer_fold, X, y, patient_ids, folds, output_dir):
    outer_train = np.flatnonzero(folds != outer_fold)
    outer_test = np.flatnonzero(folds == outer_fold)
    inner_folds = sorted(set(OUTER_FOLDS) - {outer_fold})
    inner_fold = inner_folds[(outer_fold - 1) % len(inner_folds)]
    inner_train = outer_train[folds[outer_train] != inner_fold]
    inner_val = outer_train[folds[outer_train] == inner_fold]

    train_patients = np.unique(patient_ids[inner_train])
    val_patients = np.unique(patient_ids[inner_val])
    test_patients = np.unique(patient_ids[outer_test])
    if (np.intersect1d(train_patients, val_patients).size
            or np.intersect1d(train_patients, test_patients).size
            or np.intersect1d(val_patients, test_patients).size):
        raise ValueError(f"Fuite de patients détectée dans le pli {outer_fold}.")
    if np.unique(y[inner_train]).size != 2 or np.unique(y[inner_val]).size != 2:
        raise ValueError(f"Les deux classes doivent figurer dans le pli interne {outer_fold}.")
    if np.unique(y[outer_test]).size != 2:
        raise ValueError(f"Les deux classes doivent figurer dans le pli externe {outer_fold}.")

    fold_seed = base_seed + 1009 * outer_fold
    inner_scaler = StandardScaler().fit(X[inner_train])
    X_inner_train = inner_scaler.transform(X[inner_train])
    X_inner_val = inner_scaler.transform(X[inner_val])
    _, val_scores, best_epoch, history, inner_duration = fit_inner_model(
        config, fold_seed, X_inner_train, y[inner_train],
        X_inner_val, y[inner_val],
    )
    threshold = inner_threshold(y[inner_val], val_scores)

    outer_scaler = StandardScaler().fit(X[outer_train])
    X_outer_train = outer_scaler.transform(X[outer_train])
    X_outer_test = outer_scaler.transform(X[outer_test])
    model = fit_fixed_epochs(
        config, fold_seed, X_outer_train, y[outer_train], best_epoch
    )
    test_scores = probabilities(
        model, torch.as_tensor(X_outer_test, dtype=torch.float32)
    )
    metrics = binary_metrics(y[outer_test], test_scores, threshold)

    fold_dir = output_dir / f"fold_{outer_fold}"
    fold_dir.mkdir()
    pd.DataFrame(history).to_csv(fold_dir / "inner_history.csv", index=False)
    return {
        "outer_fold": outer_fold,
        "inner_validation_fold": inner_fold,
        "n_train_ecg": int(len(outer_train)),
        "n_inner_train_ecg": int(len(inner_train)),
        "n_inner_validation_ecg": int(len(inner_val)),
        "n_test_ecg": int(len(outer_test)),
        "n_train_patients": int(len(np.unique(patient_ids[outer_train]))),
        "n_inner_train_patients": int(len(train_patients)),
        "n_inner_validation_patients": int(len(val_patients)),
        "n_test_patients": int(len(test_patients)),
        "best_epoch_from_inner_validation": best_epoch,
        "inner_early_stopping_auroc_pathological": float(
            max(item["val_auroc_pathological"] for item in history)
        ),
        "threshold_from_inner_validation": threshold,
        "inner_training_duration_s": round(inner_duration, 1),
        **metrics,
    }


def summarize(fold_results):
    summary = {}
    for metric in METRICS:
        values = [result[metric] for result in fold_results]
        summary[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
        }
    summary["PATHOLOGICAL"] = {}
    for metric in ("precision", "recall", "f1", "auroc", "average_precision"):
        values = [
            result["per_class"]["PATHOLOGICAL"][metric]
            for result in fold_results
        ]
        summary["PATHOLOGICAL"][metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
        }
    return summary


def apprentissages_internes(config, X, y, folds, seed):
    histories = []
    for outer_fold in OUTER_FOLDS:
        outer_train = np.flatnonzero(folds != outer_fold)
        inner_folds = sorted(set(OUTER_FOLDS) - {outer_fold})
        inner_fold = inner_folds[(outer_fold - 1) % len(inner_folds)]
        inner_train = outer_train[folds[outer_train] != inner_fold]
        inner_val = outer_train[folds[outer_train] == inner_fold]

        scaler = StandardScaler().fit(X[inner_train])
        X_train = scaler.transform(X[inner_train])
        X_val = scaler.transform(X[inner_val])
        _, _, _, history, _ = fit_inner_model(
            config,
            seed + 1009 * outer_fold,
            X_train,
            y[inner_train],
            X_val,
            y[inner_val],
        )
        histories.extend(
            {"outer_fold": outer_fold, **record} for record in history
        )
        print(
            f"courbes seed={seed} pli={outer_fold}: "
            f"{len(history)} époques mesurées",
            flush=True,
        )

    frame = pd.DataFrame(histories)
    columns = ("train_loss", "val_loss", "train_accuracy", "val_accuracy")
    grouped = frame.groupby("epoch", sort=True)
    epochs = []
    for epoch, values in grouped:
        entry = {
            "epoch": int(epoch),
            "fold_count": int(values["outer_fold"].nunique()),
        }
        for column in columns:
            entry[f"{column}_mean"] = float(values[column].mean())
            entry[f"{column}_std"] = float(values[column].std(ddof=1)) if len(values) > 1 else 0.0
        epochs.append(entry)
    return epochs, histories


def build_learning_curves():
    output_path = CROSS_VALIDATION_DIR / "learning_curves.json"
    if output_path.exists():
        raise FileExistsError(
            f"{output_path} existe déjà ; les courbes ne sont pas écrasées."
        )

    CROSS_VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    X, y, _, folds = load_training_data()
    protocol = json.loads((OUTPUT_DIR / "protocol.json").read_text(encoding="utf-8"))
    seeds = {}
    all_histories = []
    for run_id in protocol["run_ids"]:
        config_path = RUNS_DIR / run_id / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        seed = int(config["seed"])
        epochs, histories = apprentissages_internes(config, X, y, folds, seed)
        seeds[str(seed)] = {"fold_average": epochs}
        all_histories.extend(
            {"seed": seed, **record} for record in histories
        )

    all_frame = pd.DataFrame(all_histories)
    grouped = all_frame.groupby("epoch", sort=True)
    overall_epochs = []
    for epoch, values in grouped:
        entry = {
            "epoch": int(epoch),
            "fold_count": int(values[["seed", "outer_fold"]].drop_duplicates().shape[0]),
        }
        for column in ("train_loss", "val_loss", "train_accuracy", "val_accuracy"):
            entry[f"{column}_mean"] = float(values[column].mean())
            entry[f"{column}_std"] = (
                float(values[column].std(ddof=1)) if len(values) > 1 else 0.0
            )
        overall_epochs.append(entry)

    report_frame = pd.DataFrame(overall_epochs).rename(columns={
        "train_loss_mean": "train_loss",
        "val_loss_mean": "val_loss",
        "train_accuracy_mean": "train_accuracy",
        "val_accuracy_mean": "val_accuracy",
    })
    report_frame.to_csv(
        CROSS_VALIDATION_DIR / "learning_curves_report.csv", index=False
    )
    output = {
        "protocol": "mean_inner_learning_curves_across_nested_5_fold_cv",
        "metrics": {
            "loss": "CrossEntropyLoss après chaque époque, en mode évaluation",
            "accuracy": "Proportion de prédictions correctes au seuil softmax 0.5",
            "aggregation": (
                "Moyenne entre les plis disponibles à chaque époque ; les plis "
                "sont absents après leur arrêt anticipé."
            ),
        },
        "overall_average": overall_epochs,
        "seeds": seeds,
    }
    CROSS_VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Courbes d'apprentissage enregistrées : {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--learning-curves",
        action="store_true",
        help="Mesure et exporte les courbes de perte et d'exactitude des plis internes.",
    )
    args = parser.parse_args()
    if args.learning_curves:
        build_learning_curves()
        return

    selected = select_seeds()
    X, y, patient_ids, folds = load_training_data()
    protocol = json.loads((OUTPUT_DIR / "protocol.json").read_text(encoding="utf-8"))
    CROSS_VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = CROSS_VALIDATION_DIR / "summary.json"
    if summary_path.exists():
        report = json.loads(summary_path.read_text(encoding="utf-8"))
        if report["protocol"] != "nested_5_fold_patient_disjoint":
            raise ValueError("Le protocole existant de validation croisée diffère.")
        all_results = report["selected_seeds"]
    else:
        all_results = {}

    for selection in selected:
        seed = selection["seed"]
        if str(seed) in all_results:
            print(f"seed={seed}: résultats déjà présents, conservés.", flush=True)
            continue
        run_dir = RUNS_DIR / selection["run_id"]
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        seed_dir = CROSS_VALIDATION_DIR / f"seed_{seed}"
        if seed_dir.exists():
            raise FileExistsError(
                f"{seed_dir} existe sans entrée correspondante au résumé ; "
                "aucun résultat ne sera écrasé."
            )
        seed_dir.mkdir()
        fold_results = []
        for outer_fold in OUTER_FOLDS:
            result = run_fold(
                config, seed, outer_fold, X, y, patient_ids, folds, seed_dir
            )
            fold_results.append(result)
            print(
                f"seed={seed} fold={outer_fold}: "
                f"AUROC={result['auroc_macro']:.4f}, "
                f"F1 macro={result['f1_macro']:.4f}, "
                f"F1 PATHO={result['per_class']['PATHOLOGICAL']['f1']:.4f}, "
                f"seuil={result['threshold_from_inner_validation']:.2f}",
                flush=True,
            )
        all_results[str(seed)] = {
            "seed": seed,
            "evaluated_from_configured_seeds": True,
            "selection_score": selection["validation_auroc_pathological"],
            "source_run_id": selection["run_id"],
            "folds": fold_results,
            "summary": summarize(fold_results),
        }

    report = {
        "status": "completed",
        "task": "binary",
        "protocol": "nested_5_fold_patient_disjoint",
        "outer_folds": list(OUTER_FOLDS),
        "test_fold_reserved": protocol.get("split_folds", {}).get("test", [6]),
        "seed_selection": "all_seeds_in_binary_protocol",
        "inner_validation": (
            "Pour chaque pli externe, le fold suivant (cyclique) parmi les folds "
            "1 à 5 sert à l'arrêt anticipé et au réglage du seuil."
        ),
        "preprocessing": (
            "StandardScaler ajusté sur les seules données d'entraînement du "
            "pli correspondant."
        ),
        "selected_seeds": {
            key: all_results[key] for key in sorted(all_results, key=int)
        },
    }
    summary_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Résultats de validation croisée : {summary_path}")


if __name__ == "__main__":
    main()
