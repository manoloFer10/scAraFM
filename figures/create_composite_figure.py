#!/usr/bin/env python
"""
Create a 4-panel composite figure for LEAF AUROC results.

Panels:
- a: Random Split slice (max train-count bars)
- b: Replicate Split slice (max train-count bars)
- c: Cross Experiment Split slice (max train-count bars)
- d: GSE273033 scenario trend with grouped model families

Run from the repo root so the `evaluate` package is importable:
    python -m figures.create_composite_figure --help

This script mirrors the loading/metric pipeline from
`evaluate/eval_finetune_json.py` and pulls JSON extraction utilities from
`evaluate/extract_results.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

# Ensure repository root is importable when running this file directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate.eval_finetune_json import (  # noqa: E402
    DEFAULT_LEAF_GROUPS,
    METRIC_DISPLAY,
    MODEL_FAMILY_DISPLAY,
    VARIANT_DISPLAY,
    find_json_files,
    load_all_results,
    load_and_process_json,
    parse_model_info,
)
from evaluate.extract_results import (  # noqa: E402
    convert_to_serializable,
    extract_experiment_results,
)


GROUP_ORDER = ["Random Split", "Replicate Split", "Cross Experiment Split"]
MODEL_FAMILY_ORDER = ["logreg", "xgboost", "mlp", "scbert"]
VARIANT_ORDER = ["raw", "CLS", "WM", "ENS", "PCA_1", "PCA_2", "PCA_3", "finetuned"]

VARIANT_COLORS = {
    "raw": "#0072B2",
    "ENS": "#D55E00",
    "WM": "#009E73",
    "CLS": "#CC79A7",
    "PCA_1": "#E69F00",
    "PCA_2": "#56B4E9",
    "PCA_3": "#F0E442",
    "finetuned": "#000000",
}

# Dedicated colors for panel d (kept distinct from variant colors in panels a-c).
PANEL_D_CONSERVING_COLOR = "#334155"  # deep purple
PANEL_D_POOLING_COLOR = "#FB7185"     # rose

BRACKET_GROUP = {"logreg": 0, "xgboost": 1, "mlp": 2, "scbert": 2}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create LEAF 4-panel composite AUROC figure")
    p.add_argument(
        "--final-leaf-dir",
        type=str,
        default="results/final_leaf",
        help="Directory containing LEAF JSON result folders",
    )
    p.add_argument(
        "--random-results-dir",
        type=str,
        default="results/GSE_RANDOM_SPLIT/GSE273033",
        help="Raw random-split finetune result directory for GSE273033",
    )
    p.add_argument(
        "--random-json-path",
        type=str,
        default="results/GSE_RANDOM_SPLIT/GSE273033/experiment_results.json",
        help="Path for extracted random-split JSON",
    )
    p.add_argument(
        "--replicate-json-path",
        type=str,
        default="results/final_leaf/GSE273033/experiment_results.json",
        help="Replicate-split JSON path for GSE273033",
    )
    p.add_argument(
        "--cross-json-path",
        type=str,
        default="results/final_leaf/GSE273033_2_ERP132245/experiment_results_GSE273033_to_ERP132245.json",
        help="Cross-split JSON path for GSE273033 -> ERP132245",
    )
    p.add_argument(
        "--metric",
        type=str,
        default="aucroc",
        choices=["accuracy", "f1_macro", "f1_weighted", "aucroc"],
        help="Metric used in all four panels",
    )
    p.add_argument(
        "--outpath",
        type=str,
        default="evaluation/figure_creation/composite_leaf_4panel_aucroc.png",
        help="Output figure path",
    )
    p.add_argument(
        "--force-random-extract",
        action="store_true",
        help="Re-extract random split JSON even if file already exists",
    )
    p.add_argument(
        "--panel-d-bootstrap",
        type=int,
        default=10,
        help="Number of bootstrap resamples per model for panel d violin distributions",
    )
    p.add_argument("--dpi", type=int, default=180, help="Figure resolution")
    return p.parse_args()


def ensure_random_json(results_dir: Path, output_json: Path, force: bool = False) -> Path:
    """Create random-split JSON from raw finetune outputs when needed."""
    if output_json.exists() and not force:
        return output_json

    if not results_dir.exists():
        raise FileNotFoundError(f"Random split directory not found: {results_dir}")

    print(f"Extracting random-split JSON from: {results_dir}")
    results = extract_experiment_results(str(results_dir))
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(convert_to_serializable(results), f, indent=2)
    print(f"Saved random-split JSON: {output_json}")
    return output_json


def select_max_train_count(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the row(s) at maximum train_count for each dataset."""
    train_counts = df.groupby("dataset_id")["train_count"].max().reset_index()
    train_counts.columns = ["dataset_id", "selected_train_count"]
    return df.merge(
        train_counts,
        left_on=["dataset_id", "train_count"],
        right_on=["dataset_id", "selected_train_count"],
    ).copy()


def get_group_datasets(
    filtered_df: pd.DataFrame,
    group_name: str,
    metric: str,
    dataset_groups: dict[str, str],
) -> list[str]:
    """Datasets in a given split group, sorted by mean metric (descending)."""
    candidate = []
    for ds in sorted(filtered_df["dataset_id"].unique()):
        if dataset_groups.get(ds) == group_name:
            candidate.append(ds)

    if not candidate:
        return []

    ds_perf = filtered_df.groupby("dataset_id")[metric].mean().to_dict()
    return sorted(candidate, key=lambda ds: ds_perf.get(ds, -np.inf), reverse=True)


def build_methods(df: pd.DataFrame) -> list[str]:
    """Ordered full_method list, matching eval_finetune_json ordering logic."""
    methods = []
    for model_family in MODEL_FAMILY_ORDER:
        for variant in VARIANT_ORDER:
            method = f"{model_family}_{variant}"
            if method in df["full_method"].unique():
                methods.append(method)
    return methods


def append_random_split_dataset(
    max_df: pd.DataFrame,
    random_json_path: Path,
    alias_dataset_id: str = "GSE273033_random",
) -> pd.DataFrame:
    """Append max-train-count rows for random-split GSE273033 as a distinct dataset id."""
    if not random_json_path.exists():
        return max_df

    random_df = load_and_process_json(str(random_json_path), hide_pca=False)
    if random_df.empty:
        return max_df

    random_df = random_df[random_df["train_count"] == random_df["train_count"].max()].copy()
    random_df["dataset_id"] = alias_dataset_id

    return pd.concat([max_df, random_df], ignore_index=True)


def format_dataset_label(dataset_id: str) -> str:
    if "_" in dataset_id:
        return "\n".join(dataset_id.split("_"))
    return dataset_id


def plot_group_bar_panel(
    ax: plt.Axes,
    filtered_df: pd.DataFrame,
    datasets: list[str],
    metric: str,
    title: str,
) -> set[str]:
    """Plot one group slice of the max-train-count bar chart."""
    if not datasets:
        ax.text(0.5, 0.5, "No datasets", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return set()

    panel_df = filtered_df[filtered_df["dataset_id"].isin(datasets)].copy()
    methods = build_methods(panel_df)
    if not methods:
        ax.text(0.5, 0.5, "No methods", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return set()

    n_datasets = len(datasets)
    n_methods = len(methods)
    bar_width = 0.8 / n_methods
    x = np.arange(n_datasets)
    gap_size = 1.0
    n_bracket_groups = 3

    family_positions = {
        ds: {mf: [np.inf, -np.inf] for mf in MODEL_FAMILY_ORDER} for ds in datasets
    }
    dataset_max_vals = {ds: 0.0 for ds in datasets}
    variants_present = set(panel_df["variant"].unique())

    for idx, method in enumerate(methods):
        model_family, variant = method.split("_", 1)
        color = VARIANT_COLORS.get(variant, "#666666")
        family_idx = BRACKET_GROUP.get(model_family, n_bracket_groups - 1)

        values = []
        errors = []
        for dataset_id in datasets:
            method_data = panel_df[
                (panel_df["dataset_id"] == dataset_id) & (panel_df["full_method"] == method)
            ]
            if not method_data.empty:
                values.append(float(method_data[metric].values[0]))
                std_col = f"{metric}_std"
                if std_col in method_data.columns and not pd.isna(method_data[std_col].values[0]):
                    errors.append(float(method_data[std_col].values[0]))
                else:
                    errors.append(0.0)
            else:
                values.append(0.0)
                errors.append(0.0)

            dataset_max_vals[dataset_id] = max(dataset_max_vals[dataset_id], values[-1])

        family_shift = (family_idx - (n_bracket_groups - 1) / 2) * gap_size * bar_width
        positions = x + (idx - n_methods / 2 + 0.5) * bar_width + family_shift

        ax.bar(
            positions,
            values,
            bar_width,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.85,
            yerr=errors if any(e > 0 for e in errors) else None,
            capsize=2,
        )

        for ds_idx, dataset_id in enumerate(datasets):
            pos = positions[ds_idx]
            fam_min, fam_max = family_positions[dataset_id][model_family]
            family_positions[dataset_id][model_family][0] = min(fam_min, pos)
            family_positions[dataset_id][model_family][1] = max(fam_max, pos)

    # Merge CNN into MLP bracket group for annotation.
    for ds in datasets:
        mlp_pos = family_positions[ds]["mlp"]
        scbert_pos = family_positions[ds]["scbert"]
        if np.isfinite(scbert_pos[0]):
            mlp_pos[0] = min(mlp_pos[0], scbert_pos[0])
            mlp_pos[1] = max(mlp_pos[1], scbert_pos[1])

    y_max_overall = max(dataset_max_vals.values()) if dataset_max_vals else 1.0
    ax.set_ylim(0, max(1.15, y_max_overall + 0.2))

    for dataset_id in datasets:
        base_y = dataset_max_vals.get(dataset_id, 0.0) + 0.04
        bracket_height = 0.025
        for model_family in ["logreg", "xgboost", "mlp"]:
            fam_min, fam_max = family_positions[dataset_id][model_family]
            if not np.isfinite(fam_min) or not np.isfinite(fam_max):
                continue
            left = fam_min - bar_width * 0.2
            right = fam_max + bar_width * 0.2
            top = base_y + bracket_height

            ax.plot([left, left], [base_y, top], color="black", linewidth=1.0)
            ax.plot([left, right], [top, top], color="black", linewidth=1.0)
            ax.plot([right, right], [base_y, top], color="black", linewidth=1.0)

            ax.text(
                (left + right) / 2,
                top + 0.005,
                MODEL_FAMILY_DISPLAY.get(model_family, model_family.upper()),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([format_dataset_label(ds) for ds in datasets], ha="center", fontsize=9)
    ax.set_xlim(-0.5, n_datasets - 0.5)
    ax.set_xlabel("Dataset", fontsize=11)
    ax.set_ylabel(METRIC_DISPLAY.get(metric, metric), fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_title(title, fontsize=13, fontweight="bold")

    return variants_present


def compute_group_stat(df: pd.DataFrame, variants: set[str], metric: str) -> tuple[float, float, int]:
    subset = df[df["variant"].isin(variants)]
    if subset.empty:
        return np.nan, np.nan, 0

    vals = subset[metric].astype(float)
    mean = float(vals.mean())
    std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return mean, std, int(len(vals))


def compute_metric_value(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    """Compute one metric value using the same conventions as eval_finetune_json."""
    if metric == "accuracy":
        return float(np.mean(y_true == y_pred))

    if metric == "f1_macro":
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    if metric == "f1_weighted":
        return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    if metric == "aucroc":
        try:
            classes = np.unique(np.concatenate([y_true, y_pred]))
            if len(classes) == 2:
                return float(roc_auc_score(y_true, y_pred))
            if len(classes) > 2:
                y_true_bin = label_binarize(y_true, classes=classes)
                y_pred_bin = label_binarize(y_pred, classes=classes)
                return float(
                    roc_auc_score(y_true_bin, y_pred_bin, average="macro", multi_class="ovr")
                )
            return np.nan
        except Exception:
            return np.nan

    raise ValueError(f"Unsupported metric: {metric}")


def bootstrap_metric_samples(
    y_true: list,
    y_pred: list,
    metric: str,
    n_bootstrap: int,
    random_state: int,
) -> np.ndarray:
    """Return bootstrap metric samples for one model prediction vector."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if len(y_true_arr) == 0 or len(y_true_arr) != len(y_pred_arr):
        return np.array([], dtype=float)

    rng = np.random.RandomState(random_state)
    n_samples = len(y_true_arr)
    samples = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        yt = y_true_arr[idx]
        yp = y_pred_arr[idx]
        value = compute_metric_value(yt, yp, metric)
        if not np.isnan(value):
            samples.append(float(value))

    return np.asarray(samples, dtype=float)


def extract_panel_d_bootstrap_values(
    json_path: Path,
    metric: str,
    conserving_variants: set[str],
    pooling_variants: set[str],
    n_bootstrap: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract bootstrap samples by variant-group from max-train-count experiment in one JSON."""
    with open(json_path, "r") as f:
        data = json.load(f)

    max_train_count = None
    max_experiment = None
    for train_count_str, experiment in data.items():
        train_count = experiment.get("train_count")
        if train_count is None and isinstance(train_count_str, str) and train_count_str.isdigit():
            train_count = int(train_count_str)
        if train_count is None:
            continue
        if max_train_count is None or train_count > max_train_count:
            max_train_count = train_count
            max_experiment = experiment

    if max_experiment is None:
        return np.array([], dtype=float), np.array([], dtype=float)

    ground_truth = max_experiment.get("ground_truth")
    global_y_true = ground_truth.get("labels") if ground_truth else None
    batteries = max_experiment.get("batteries", {})

    conserving_points = []
    pooling_points = []
    model_counter = 0

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
            variant = model_info.get("variant")
            if model_info.get("model_family") is None or variant is None:
                continue

            samples = bootstrap_metric_samples(
                y_true=y_true,
                y_pred=y_pred,
                metric=metric,
                n_bootstrap=n_bootstrap,
                random_state=42 + model_counter,
            )
            model_counter += 1

            if samples.size == 0:
                continue

            if variant in conserving_variants:
                conserving_points.append(samples)
            elif variant in pooling_variants:
                pooling_points.append(samples)

    cons = np.concatenate(conserving_points) if conserving_points else np.array([], dtype=float)
    pool = np.concatenate(pooling_points) if pooling_points else np.array([], dtype=float)
    return cons, pool


def plot_panel_d(
    ax: plt.Axes,
    random_json: Path,
    replicate_json: Path,
    cross_json: Path,
    metric: str,
    n_bootstrap: int,
) -> pd.DataFrame:
    """Plot scenario trend panel for grouped classifier categories."""
    scenario_paths = [
        ("Random Split", random_json),
        ("Replicate Split", replicate_json),
        ("Cross Experiment Split", cross_json),
    ]

    conserving_variants = {"ENS", "PCA_1", "PCA_2", "PCA_3", "finetuned"}
    pooling_variants = {"WM", "CLS"}

    rows = []
    conserving_values = []
    pooling_values = []

    for scenario, path in scenario_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing scenario JSON: {path}")

        scenario_df = load_and_process_json(str(path), hide_pca=False)
        if scenario_df.empty:
            raise RuntimeError(f"No usable rows in: {path}")

        scenario_df = scenario_df[scenario_df["train_count"] == scenario_df["train_count"].max()].copy()

        cons_subset = scenario_df[scenario_df["variant"].isin(conserving_variants)][metric].astype(float)
        pool_subset = scenario_df[scenario_df["variant"].isin(pooling_variants)][metric].astype(float)

        cons_boot, pool_boot = extract_panel_d_bootstrap_values(
            json_path=path,
            metric=metric,
            conserving_variants=conserving_variants,
            pooling_variants=pooling_variants,
            n_bootstrap=n_bootstrap,
        )

        # Fallback to per-model point estimates if bootstrap extraction has no valid samples.
        conserving_values.append(cons_boot if cons_boot.size > 0 else cons_subset.values)
        pooling_values.append(pool_boot if pool_boot.size > 0 else pool_subset.values)

        c_mean, c_std, c_n = compute_group_stat(scenario_df, conserving_variants, metric)
        p_mean, p_std, p_n = compute_group_stat(scenario_df, pooling_variants, metric)

        rows.append(
            {
                "scenario": scenario,
                "group": "Conserving gene identity",
                "mean": c_mean,
                "std": c_std,
                "n_models": c_n,
            }
        )
        rows.append(
            {
                "scenario": scenario,
                "group": "Summarized embeddings",
                "mean": p_mean,
                "std": p_std,
                "n_models": p_n,
            }
        )

    stats = pd.DataFrame(rows)

    x_labels = ["Random Split", "Replicate Split", "Cross Experiment Split"]
    x = np.arange(len(x_labels))
    pos_offset = 0.12
    cons_positions = x - pos_offset
    pool_positions = x + pos_offset

    conserving_color = PANEL_D_CONSERVING_COLOR
    pooling_color = PANEL_D_POOLING_COLOR

    # Colored violins for distributions per scenario/group.
    cons_violins = ax.violinplot(
        conserving_values,
        positions=cons_positions,
        widths=0.2,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body in cons_violins["bodies"]:
        body.set_facecolor(conserving_color)
        body.set_edgecolor("black")
        body.set_alpha(0.75)

    pool_violins = ax.violinplot(
        pooling_values,
        positions=pool_positions,
        widths=0.2,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body in pool_violins["bodies"]:
        body.set_facecolor(pooling_color)
        body.set_edgecolor("black")
        body.set_alpha(0.75)

    cons_means = (
        stats[stats["group"] == "Conserving gene identity"].set_index("scenario").loc[x_labels]["mean"].values
    )
    pool_means = (
        stats[stats["group"] == "Summarized embeddings"].set_index("scenario").loc[x_labels]["mean"].values
    )

    # Gray dashed trend lines across scenarios.
    ax.plot(cons_positions, cons_means, linestyle="--", color="#777777", linewidth=1.8, zorder=3)
    ax.plot(pool_positions, pool_means, linestyle="--", color="#777777", linewidth=1.8, zorder=3)

    # Mean markers in the same colors as violins.
    ax.scatter(cons_positions, cons_means, color=conserving_color, s=36, zorder=4)
    ax.scatter(pool_positions, pool_means, color=pooling_color, s=36, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=0, ha="center", fontsize=9)
    ax.set_ylabel(METRIC_DISPLAY.get(metric, metric), fontsize=11)
    ax.set_xlabel("Scenario", fontsize=11)
    ax.set_title("Fixed dataset across experimental scenarios", fontsize=14)
    ax.grid(alpha=0.3, linestyle="--", axis="both")
    ax.set_ylim(0.45, 1.02)

    legend_handles = [
        Patch(facecolor=conserving_color, edgecolor="black", label="Conserving gene identity"),
        Patch(facecolor=pooling_color, edgecolor="black", label="Summarized embeddings"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9, framealpha=0.9)

    return stats


def main() -> None:
    args = parse_args()

    final_leaf_dir = Path(args.final_leaf_dir)
    random_results_dir = Path(args.random_results_dir)
    random_json_path = Path(args.random_json_path)
    replicate_json_path = Path(args.replicate_json_path)
    cross_json_path = Path(args.cross_json_path)
    outpath = Path(args.outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    # 1) Ensure random split JSON exists.
    random_json_path = ensure_random_json(
        random_results_dir,
        random_json_path,
        force=args.force_random_extract,
    )

    # 2) Load LEAF results with the same JSON loading/metric path as eval_finetune_json.py.
    leaf_json_files = find_json_files(str(final_leaf_dir))
    if not leaf_json_files:
        raise FileNotFoundError(f"No experiment_results*.json files under {final_leaf_dir}")

    results_df = load_all_results(leaf_json_files, hide_pca=False)
    if results_df.empty:
        raise RuntimeError("No valid rows loaded from final_leaf JSON files")

    # 3) Build max-train-count data used by source barplot.
    max_df = select_max_train_count(results_df)

    # Add GSE273033 random split into panel-a universe as a dedicated dataset key.
    max_df = append_random_split_dataset(
        max_df=max_df,
        random_json_path=random_json_path,
        alias_dataset_id="GSE273033_random",
    )

    dataset_groups = dict(DEFAULT_LEAF_GROUPS)
    dataset_groups["GSE273033_random"] = "Random Split"

    random_datasets = get_group_datasets(
        max_df,
        "Random Split",
        metric=args.metric,
        dataset_groups=dataset_groups,
    )
    replicate_datasets = get_group_datasets(
        max_df,
        "Replicate Split",
        metric=args.metric,
        dataset_groups=dataset_groups,
    )
    cross_datasets = get_group_datasets(
        max_df,
        "Cross Experiment Split",
        metric=args.metric,
        dataset_groups=dataset_groups,
    )

    # 4) Create composite 4-panel figure.
    # Top row: a:b = 4:1 width. Bottom row stays 1:1.
    fig = plt.figure(figsize=(18, 12), constrained_layout=False)
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.2)
    top = outer[0].subgridspec(1, 2, width_ratios=[4, 1], wspace=0.12)
    bottom = outer[1].subgridspec(1, 2, width_ratios=[1, 1], wspace=0.12)

    ax_a = fig.add_subplot(top[0, 0])
    ax_b = fig.add_subplot(top[0, 1])
    ax_c = fig.add_subplot(bottom[0, 0])
    ax_d = fig.add_subplot(bottom[0, 1])

    variants_a = plot_group_bar_panel(ax_a, max_df, random_datasets, args.metric, "Random Split")
    variants_b = plot_group_bar_panel(ax_b, max_df, replicate_datasets, args.metric, "Replicate Split")
    variants_c = plot_group_bar_panel(
        ax_c,
        max_df,
        cross_datasets,
        args.metric,
        "Cross Experiment Split",
    )

    # Reduce redundant y-labels in the top-right panel.
    ax_b.set_ylabel("")

    stats_df = plot_panel_d(
        ax_d,
        random_json=random_json_path,
        replicate_json=replicate_json_path,
        cross_json=cross_json_path,
        metric=args.metric,
        n_bootstrap=args.panel_d_bootstrap,
    )

    # Panel tags.
    for label, ax in zip(["a", "b", "c", "d"], [ax_a, ax_b, ax_c, ax_d]):
        ax.text(
            -0.11,
            1.06,
            label,
            transform=ax.transAxes,
            fontsize=18,
            fontweight="bold",
            va="top",
            ha="left",
        )

    # Shared variant legend for bar panels.
    variants_union = [
        v for v in VARIANT_ORDER if v in variants_a or v in variants_b or v in variants_c
    ]
    variant_handles = [
        Patch(
            facecolor=VARIANT_COLORS.get(v, "#666666"),
            edgecolor="black",
            label=VARIANT_DISPLAY.get(v, v),
        )
        for v in variants_union
    ]
    if variant_handles:
        fig.legend(
            handles=variant_handles,
            title="Variant (Panels a-c)",
            loc="center right",
            bbox_to_anchor=(0.995, 0.63),
            fontsize=7,
            title_fontsize=9,
            framealpha=0.9,
        )

    # plt.suptitle("LEAF Composite Figure: Split Slices and GSE273033 Scenario Trend", fontsize=16, y=0.99)
    fig.subplots_adjust(left=0.06, right=0.9, bottom=0.06, top=0.96)
    fig.savefig(outpath, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    # Save panel-d source table for traceability.
    stats_path = outpath.with_name(outpath.stem + "_panel_d_stats.csv")
    stats_df.to_csv(stats_path, index=False)

    print(f"Saved composite figure: {outpath}")
    print(f"Saved panel-d stats: {stats_path}")


if __name__ == "__main__":
    main()