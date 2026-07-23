"""Helpers for loading finetune-experiment JSON results into a tidy DataFrame.

Only the subset of the upstream evaluation module used by the figure-generation
scripts under `figures/` is kept here:
    - constants: DEFAULT_LEAF_GROUPS, GROUP_ORDER, MODEL_FAMILY_DISPLAY,
      VARIANT_DISPLAY, METRIC_DISPLAY
    - functions: find_json_files, parse_model_info, load_and_process_json,
      load_all_results, get_grouped_datasets
"""

import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize


DEFAULT_LEAF_GROUPS = {
    "SRP398011": "Random Split",
    "SRP169576": "Random Split",
    "SRP148288": "Random Split",
    "SRP166333": "Random Split",
    "SRP285817": "Random Split",
    "ERP132245": "Random Split",
    "GSE226826": "Random Split",
    "GSE235495": "Replicate Split",
    "GSE273033": "Replicate Split",
    "GSE273033_to_ERP132245": "Cross Experiment Split",
    "SRP398011_to_GSE226826": "Cross Experiment Split",
}

GROUP_ORDER = ["Random Split", "Replicate Split", "Cross Experiment Split"]

MODEL_FAMILY_DISPLAY = {
    "logreg": "LR",
    "xgboost": "XGB",
    "mlp": "MLP",
    "scbert": "CNN",
}

VARIANT_DISPLAY = {
    "raw": "RAW",
    "CLS": "CLS",
    "WM": "WM",
    "ENS": "ENS",
    "PCA_1": "PCA (1D)",
    "PCA_2": "PCA (2D)",
    "PCA_3": "PCA (3D)",
    "finetuned": "CNN",
}

METRIC_DISPLAY = {
    "accuracy": "Accuracy",
    "f1_macro": "F1 Macro",
    "f1_weighted": "F1 Weighted",
    "aucroc": "AUROC",
}


def find_json_files(results_dir: str) -> list:
    """Recursively find all experiment_results*.json files."""
    pattern = os.path.join(results_dir, "**", "experiment_results*.json")
    return sorted(glob.glob(pattern, recursive=True))


def compute_metrics(y_true: list, y_pred: list, n_bootstrap: int = 100, random_state: int = 42) -> dict:
    """Compute classification metrics (accuracy, F1, AUROC) with bootstrap std."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true) != len(y_pred):
        print(f"  Warning: Length mismatch - y_true={len(y_true)}, y_pred={len(y_pred)}. Skipping.")
        return None

    n_samples = len(y_true) // n_bootstrap
    rng = np.random.RandomState(random_state)

    successes = (y_true == y_pred).astype(int)
    accuracy = float(np.mean(successes))
    accuracy_std = float(np.std(successes, ddof=1) / np.sqrt(n_samples))

    metrics = {
        "accuracy": accuracy,
        "accuracy_std": accuracy_std,
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    f1_macro_samples = []
    f1_weighted_samples = []
    aucroc_samples = []

    for _ in range(n_bootstrap):
        indices = rng.choice(len(y_true), size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]

        f1_macro_samples.append(f1_score(y_true_boot, y_pred_boot, average="macro", zero_division=0))
        f1_weighted_samples.append(f1_score(y_true_boot, y_pred_boot, average="weighted", zero_division=0))

        try:
            classes = np.unique(np.concatenate([y_true_boot, y_pred_boot]))
            if len(classes) == 2:
                aucroc_samples.append(roc_auc_score(y_true_boot, y_pred_boot))
            elif len(classes) > 2:
                y_true_bin = label_binarize(y_true_boot, classes=classes)
                y_pred_bin = label_binarize(y_pred_boot, classes=classes)
                aucroc_samples.append(
                    roc_auc_score(y_true_bin, y_pred_bin, average="macro", multi_class="ovr")
                )
            else:
                aucroc_samples.append(np.nan)
        except Exception:
            aucroc_samples.append(np.nan)

    metrics["f1_macro_std"] = float(np.nanstd(f1_macro_samples, ddof=1))
    metrics["f1_weighted_std"] = float(np.nanstd(f1_weighted_samples, ddof=1))

    try:
        classes = np.unique(np.concatenate([y_true, y_pred]))
        if len(classes) == 2:
            metrics["aucroc"] = float(roc_auc_score(y_true, y_pred))
        elif len(classes) > 2:
            y_true_bin = label_binarize(y_true, classes=classes)
            y_pred_bin = label_binarize(y_pred, classes=classes)
            metrics["aucroc"] = float(
                roc_auc_score(y_true_bin, y_pred_bin, average="macro", multi_class="ovr")
            )
        else:
            metrics["aucroc"] = np.nan
    except Exception:
        metrics["aucroc"] = np.nan

    metrics["aucroc_std"] = float(np.nanstd(aucroc_samples, ddof=1)) if aucroc_samples else np.nan

    return metrics


def extract_dataset_id_from_path(json_path: str) -> str:
    """Extract dataset id from a results JSON path."""
    path = Path(json_path)
    filename = path.stem
    if filename.startswith("experiment_results_"):
        return filename.replace("experiment_results_", "")
    return path.parent.name


def parse_model_info(battery_name: str, model_name: str) -> dict:
    """Parse battery/model names to (model_family, input_type, variant) tuple-like dict."""
    info = {
        "battery": battery_name,
        "model": model_name,
        "model_family": None,
        "input_type": None,
        "variant": None,
        "is_pca": False,
    }

    if battery_name == "scbert_classifier":
        info["model_family"] = "scbert"
        info["input_type"] = "raw"
        info["variant"] = "finetuned"
        return info

    if "logreg" in model_name.lower():
        info["model_family"] = "logreg"
    elif "xgboost" in model_name.lower():
        info["model_family"] = "xgboost"
    elif "mlp" in model_name.lower():
        info["model_family"] = "mlp"

    if battery_name == "battery_raw":
        info["input_type"] = "raw"
        info["variant"] = "raw"
    elif battery_name == "battery_embeddings_cls":
        info["input_type"] = "embeddings"
        info["variant"] = "CLS"
    elif battery_name == "battery_WM":
        info["input_type"] = "embeddings"
        info["variant"] = "WM"
    elif battery_name == "battery_embeddings_per_gene":
        info["input_type"] = "embeddings"
        if model_name.startswith("ensemble_"):
            info["variant"] = "ENS"
        elif model_name.startswith("PCA_"):
            parts = model_name.split("_")
            if len(parts) >= 2:
                info["variant"] = f"PCA_{parts[1]}"
                info["is_pca"] = True

    return info


def load_and_process_json(json_path: str, hide_pca: bool = False, n_bootstrap: int = 10) -> pd.DataFrame:
    """Load a single experiment_results*.json and compute per-model metrics.

    ``n_bootstrap`` controls the number of bootstrap resamples used to estimate the
    per-metric standard deviation (the ``*_std`` columns, e.g. the ``±std`` shown in the
    heatmap cells).
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    dataset_id = extract_dataset_id_from_path(json_path)

    records = []
    for train_count_str, experiment in data.items():
        train_count = experiment.get(
            "train_count",
            int(train_count_str) if train_count_str.isdigit() else None,
        )
        ground_truth = experiment.get("ground_truth")
        batteries = experiment.get("batteries", {})

        global_y_true = ground_truth.get("labels") if ground_truth else None

        for battery_name, battery_data in batteries.items():
            predictions = battery_data.get("predictions", {})
            battery_test_labels = battery_data.get("test_labels")
            y_true = battery_test_labels if battery_test_labels is not None else global_y_true
            if y_true is None:
                continue

            for model_name, y_pred in predictions.items():
                if not y_pred:
                    continue

                model_info = parse_model_info(battery_name, model_name)
                if model_info["is_pca"] and hide_pca:
                    continue
                if model_info["model_family"] is None:
                    continue

                metrics = compute_metrics(y_true, y_pred, n_bootstrap=n_bootstrap)
                if metrics is None:
                    continue

                records.append({
                    "dataset_id": dataset_id,
                    "train_count": train_count,
                    "model_family": model_info["model_family"],
                    "input_type": model_info["input_type"],
                    "variant": model_info["variant"],
                    "full_method": f"{model_info['model_family']}_{model_info['variant']}",
                    **metrics,
                })

    return pd.DataFrame(records)


def load_all_results(json_files: list, hide_pca: bool = False, n_bootstrap: int = 10) -> pd.DataFrame:
    """Load and concatenate multiple experiment_results*.json files.

    ``n_bootstrap`` is forwarded to :func:`load_and_process_json` to control the bootstrap
    resampling used for the ``*_std`` columns.
    """
    all_dfs = []
    for json_file in json_files:
        print(f"Loading: {json_file}")
        df = load_and_process_json(json_file, hide_pca=hide_pca, n_bootstrap=n_bootstrap)
        if not df.empty:
            all_dfs.append(df)
            print(f"  Found {len(df)} records across {df['train_count'].nunique()} train counts")

    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)


def get_grouped_datasets(datasets: list, dataset_groups: dict, include_empty_groups: bool = True) -> list:
    """Return [(dataset_id, group_name), ...] ordered by GROUP_ORDER."""
    grouped = defaultdict(list)
    ungrouped = []

    for ds in datasets:
        group = dataset_groups.get(ds)
        if group:
            grouped[group].append(ds)
        else:
            ungrouped.append(ds)

    ordered = []
    for group in GROUP_ORDER:
        if group in grouped:
            for ds in sorted(grouped[group]):
                ordered.append((ds, group))
        elif include_empty_groups:
            ordered.append((f"__placeholder__{group}", group))

    for ds in sorted(ungrouped):
        ordered.append((ds, None))

    return ordered
