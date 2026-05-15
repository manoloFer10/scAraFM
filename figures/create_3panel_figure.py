#!/usr/bin/env python
"""
Create a 3-panel composite figure for LEAF AUROC results.

Panels:
- a: Max-train-count grouped barplot (same style as create_specific_train_count_barplot)
- b: GSE273033 scenario trend (same as plot_panel_d)
- c: MLP/CNN train-count trends for GSE273033 scenarios (same style as create_subplot_matrix)

Run from the repo root so the `evaluate` and `figures` packages are importable:
    python -m figures.create_3panel_figure --help

`ensure_random_json` and `plot_panel_d` are imported from the sibling
`create_composite_figure` module; the rest of the JSON ingestion helpers come
from `evaluate.eval_finetune_json`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# Ensure repository root is importable when running this file directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate.eval_finetune_json import (  # noqa: E402
    DEFAULT_LEAF_GROUPS,
    GROUP_ORDER,
    METRIC_DISPLAY,
    MODEL_FAMILY_DISPLAY,
    VARIANT_DISPLAY,
    find_json_files,
    get_grouped_datasets,
    load_all_results,
    load_and_process_json,
)
from figures.create_composite_figure import (  # noqa: E402
    ensure_random_json,
    plot_panel_d,
)


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

VARIANT_MARKERS = {
    "raw": "o",
    "ENS": "^",
    "WM": "s",
    "CLS": "D",
    "PCA_1": "v",
    "PCA_2": "<",
    "PCA_3": ">",
    "finetuned": "*",
}

BRACKET_GROUP = {"logreg": 0, "xgboost": 1, "mlp": 2, "scbert": 2}

SCENARIO_TITLE_COLORS = {
    "Random Split": "#4CAF50",
    "Replicate Split": "#2196F3",
    "Cross Experiment Split": "#FF5722",
}

BREAK_DIAG_SIZE = 0.012


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create LEAF 3-panel composite figure")
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
        default="results/final_leaf(jsons)/results/final_leaf/GSE273033/experiment_results.json",
        help="Replicate-split JSON path for GSE273033",
    )
    p.add_argument(
        "--cross-json-path",
        type=str,
        default="results/final_leaf(jsons)/results/final_leaf/GSE273033_2_ERP132245/experiment_results_GSE273033_to_ERP132245.json",
        help="Cross-split JSON path for GSE273033 -> ERP132245",
    )
    p.add_argument(
        "--replicate-dataset-id",
        type=str,
        default="GSE273033",
        help="Replicate scenario dataset id in the combined results table",
    )
    p.add_argument(
        "--cross-dataset-id",
        type=str,
        default="GSE273033_to_ERP132245",
        help="Cross scenario dataset id in the combined results table",
    )
    p.add_argument(
        "--metric",
        type=str,
        default="aucroc",
        choices=["accuracy", "f1_macro", "f1_weighted", "aucroc"],
        help="Metric used in all panels",
    )
    p.add_argument(
        "--outpath",
        type=str,
        default="evaluation/figure_creation/composite_leaf_3panel_aucroc.png",
        help="Output figure path",
    )
    p.add_argument(
        "--force-random-extract",
        action="store_true",
        help="Re-extract random split JSON even if file already exists",
    )
    p.add_argument(
        "--panel-b-bootstrap",
        type=int,
        default=10,
        help="Number of bootstrap resamples per model for panel b violins",
    )
    p.add_argument("--dpi", type=int, default=300, help="Figure resolution")
    return p.parse_args()


def format_dataset_label(dataset_id: str) -> str:
    if dataset_id == "GSE273033_random":
        dataset_id = "GSE273033"
    if "GSE273033" in dataset_id or "ERP132245" in dataset_id:
        dataset_id += '\n(Drought Stress)'
    if "GSE226826" in dataset_id or "SRP398011" in dataset_id:
        dataset_id += '\n(Infection with P. Syringae)'
    if "_" in dataset_id:
        return " ".join(dataset_id.split("_"))
    return dataset_id


def apply_broken_y_axes_pair(
    ax_top: plt.Axes,
    ax_bottom: plt.Axes,
    low_max: float = 0.05,
    high_min: float = 0.4,
    break_size: float = BREAK_DIAG_SIZE,
) -> None:
    """Configure a classic broken-y pair: [0, low_max] and [high_min, y_max]."""
    _, y_max = ax_top.get_ylim()
    if y_max <= high_min:
        y_max = high_min + 0.01

    ax_top.set_ylim(high_min, y_max)
    ax_bottom.set_ylim(0.0, low_max)

    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)
    ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_bottom.xaxis.tick_bottom()

    ax_bottom.set_yticks([0.0])

    kwargs_top = {
        "transform": ax_top.transAxes,
        "color": "black",
        "clip_on": False,
        "linewidth": 1.2,
    }
    kwargs_bottom = {
        "transform": ax_bottom.transAxes,
        "color": "black",
        "clip_on": False,
        "linewidth": 1.2,
    }

    # Top axis marks on its bottom edge.
    ax_top.plot((-break_size, +break_size), (-break_size, +break_size), **kwargs_top)
    ax_top.plot((1 - break_size, 1 + break_size), (-break_size, +break_size), **kwargs_top)
    # Bottom axis marks on its top edge.
    ax_bottom.plot((-break_size, +break_size), (1 - break_size, 1 + break_size), **kwargs_bottom)
    ax_bottom.plot(
        (1 - break_size, 1 + break_size),
        (1 - break_size, 1 + break_size),
        **kwargs_bottom,
    )


def append_random_split_full_dataset(
    results_df: pd.DataFrame,
    random_json_path: Path,
    alias_dataset_id: str = "GSE273033_random",
) -> pd.DataFrame:
    """Append all train-count rows for random-split GSE273033 as a distinct dataset id."""
    if not random_json_path.exists():
        return results_df

    random_df = load_and_process_json(str(random_json_path), hide_pca=False)
    if random_df.empty:
        return results_df

    random_df = random_df.copy()
    random_df["dataset_id"] = alias_dataset_id
    return pd.concat([results_df, random_df], ignore_index=True)


def _select_train_count_rows(results_df: pd.DataFrame, train_count: int | str) -> pd.DataFrame:
    if train_count != "max":
        train_counts = (
            results_df.groupby("dataset_id")["train_count"]
            .apply(lambda x: train_count if train_count in x.values else x.max())
            .reset_index()
        )
        train_counts.columns = ["dataset_id", "selected_train_count"]
    else:
        train_counts = results_df.groupby("dataset_id")["train_count"].max().reset_index()
        train_counts.columns = ["dataset_id", "selected_train_count"]

    return results_df.merge(
        train_counts,
        left_on=["dataset_id", "train_count"],
        right_on=["dataset_id", "selected_train_count"],
    ).copy()


def plot_specific_train_count_barplot_on_ax(
    ax: plt.Axes,
    results_df: pd.DataFrame,
    metric: str,
    dataset_groups: dict[str, str] | None,
    train_count: int | str = "max",
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    show_title: bool = True,
    show_legend: bool = True,
    show_group_labels: bool = True,
    show_family_brackets: bool = True,
) -> set[str]:
    """Axis-level version of create_specific_train_count_barplot."""
    filtered_df = _select_train_count_rows(results_df, train_count=train_count)
    if filtered_df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return set()

    all_datasets = sorted(filtered_df["dataset_id"].unique())
    dataset_perf = filtered_df.groupby("dataset_id")[metric].mean().to_dict()

    if dataset_groups:
        grouped_datasets = get_grouped_datasets(all_datasets, dataset_groups, include_empty_groups=False)
        dataset_to_group = {ds: group for ds, group in grouped_datasets}

        ordered_datasets = []
        for group in GROUP_ORDER:
            group_ds = [
                ds
                for ds, g in grouped_datasets
                if g == group and not ds.startswith("__placeholder__")
            ]
            group_ds_sorted = sorted(
                group_ds,
                key=lambda ds: dataset_perf.get(ds, -np.inf),
                reverse=True,
            )
            ordered_datasets.extend(group_ds_sorted)

        ungrouped_ds = [
            ds for ds, g in grouped_datasets if g is None and not ds.startswith("__placeholder__")
        ]
        ordered_datasets.extend(
            sorted(ungrouped_ds, key=lambda ds: dataset_perf.get(ds, -np.inf), reverse=True)
        )
        datasets = ordered_datasets
    else:
        datasets = sorted(all_datasets, key=lambda ds: dataset_perf.get(ds, -np.inf), reverse=True)
        dataset_to_group = {}

    all_methods = []
    for mf in MODEL_FAMILY_ORDER:
        for var in VARIANT_ORDER:
            method = f"{mf}_{var}"
            if method in filtered_df["full_method"].unique():
                all_methods.append(method)

    n_datasets = len(datasets)
    n_methods = len(all_methods)
    if n_methods == 0 or n_datasets == 0:
        ax.text(0.5, 0.5, "No methods", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return set()

    bar_width = 0.8 / n_methods
    x = np.arange(n_datasets)
    n_bracket_groups = 3
    gap_size = 1

    family_positions = {
        ds: {mf: [np.inf, -np.inf] for mf in MODEL_FAMILY_ORDER} for ds in datasets
    }
    dataset_max_vals = {ds: 0.0 for ds in datasets}

    for idx, method in enumerate(all_methods):
        model_family, variant = method.split("_", 1)
        color = VARIANT_COLORS.get(variant, "#666666")
        family_idx = BRACKET_GROUP.get(model_family, n_bracket_groups - 1)

        values = []
        errors = []
        for dataset_id in datasets:
            method_data = filtered_df[
                (filtered_df["dataset_id"] == dataset_id)
                & (filtered_df["full_method"] == method)
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
            yerr=errors if any(e > 0 for e in errors) else None,
            capsize=2,
            alpha=0.85,
        )

        for ds_idx, dataset_id in enumerate(datasets):
            pos = positions[ds_idx]
            fam_min, fam_max = family_positions[dataset_id][model_family]
            family_positions[dataset_id][model_family][0] = min(fam_min, pos)
            family_positions[dataset_id][model_family][1] = max(fam_max, pos)

    if dataset_groups and dataset_to_group:
        current_group = None
        group_starts = []
        group_ends = []
        group_names = []

        for i, ds in enumerate(datasets):
            group = dataset_to_group.get(ds)
            if group != current_group:
                if current_group is not None:
                    group_ends.append(i - 1)
                group_starts.append(i)
                group_names.append(group)
                current_group = group
        group_ends.append(len(datasets) - 1)

        for i in range(1, len(group_starts)):
            sep_x = (group_starts[i] + group_ends[i - 1]) / 2
            ax.axvline(x=sep_x, color="gray", linestyle="--", linewidth=1, alpha=0.5)

        if show_group_labels:
            for start, end, name in zip(group_starts, group_ends, group_names):
                if not name:
                    continue
                mid_x = (start + end) / 2
                ax.text(
                    mid_x,
                    0.94,
                    name,
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                    color=SCENARIO_TITLE_COLORS.get(name, "#666666"),
                    transform=ax.get_xaxis_transform(),
                )

    for ds in datasets:
        mlp_pos = family_positions[ds]["mlp"]
        scbert_pos = family_positions[ds]["scbert"]
        if np.isfinite(scbert_pos[0]):
            mlp_pos[0] = min(mlp_pos[0], scbert_pos[0])
            mlp_pos[1] = max(mlp_pos[1], scbert_pos[1])

    y_max_overall = max(dataset_max_vals.values()) if dataset_max_vals else 1.0

    if show_family_brackets:
        for dataset_id in datasets:
            base_y = dataset_max_vals.get(dataset_id, 0.0) + 0.02
            bracket_height = 0.02
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
                    fontsize=8,
                    fontweight="bold",
                )

    ax.set_xticks(x)
    ax.set_xticklabels([format_dataset_label(ds) for ds in datasets], ha="center", fontsize=9)
    if show_ylabel:
        ax.set_ylabel(METRIC_DISPLAY.get(metric, metric), fontsize=11)
    else:
        ax.set_ylabel("")
    if show_xlabel:
        ax.set_xlabel("Dataset", fontsize=11)
    else:
        ax.set_xlabel("")
    ax.set_ylim(0, 1.07)
    ax.set_xlim(-0.5, n_datasets - 0.5)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=1, alpha=0.5)

    variants_in_plot = [v for v in VARIANT_ORDER if v in filtered_df["variant"].unique()]
    legend_handles = [
        Patch(
            facecolor=VARIANT_COLORS.get(v, "#666666"),
            edgecolor="black",
            label=VARIANT_DISPLAY.get(v, v),
        )
        for v in variants_in_plot
    ]
    if show_legend:
        ax.legend(
            handles=legend_handles,
            title="Representation variants",
            fontsize=9,
            title_fontsize=10,
            loc="upper left",
            bbox_to_anchor=(0.865, 0.90),
            ncol=2,
            framealpha=0.9,
        )

    if show_title:
        ax.set_title(f"Leaf perturbation prediction tasks overview", fontsize=14)
    else:
        ax.set_title("")

    # for i, dataset_id in enumerate(datasets):
    #     max_tc = filtered_df[filtered_df["dataset_id"] == dataset_id]["train_count"].values[0]
    #     offset = -40 if "_" in dataset_id else -18
    #     ax.annotate(
    #         f"n={int(max_tc)}",
    #         xy=(i, 0),
    #         xytext=(0, offset),
    #         textcoords="offset points",
    #         ha="center",
    #         va="top",
    #         fontsize=7,
    #         color="#666666",
    #     )

    return set(variants_in_plot)


def plot_panel_c_mlp_subplots(
    axes: list[plt.Axes],
    results_df: pd.DataFrame,
    metric: str,
    scenario_dataset_ids: list[str],
    scenario_labels: list[str],
    show_subplot_titles: bool = True,
    show_xlabel: bool = True,
    show_ylabel_first: bool = True,
    show_legend: bool = False,
) -> tuple[list, list]:
    """Create MLP/CNN-only scenario subplots following create_subplot_matrix style."""
    legend_handles = None
    legend_labels = None

    for i, (ax, dataset_id, label) in enumerate(zip(axes, scenario_dataset_ids, scenario_labels)):
        subset = results_df[
            (results_df["dataset_id"] == dataset_id)
            & (results_df["model_family"].isin(["mlp", "scbert"]))
        ].copy()

        if subset.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            if show_subplot_titles:
                ax.text(
                    0.5,
                    0.97,
                    label,
                    ha="center",
                    va="top",
                    transform=ax.transAxes,
                    fontsize=10,
                    fontweight="bold",
                    color=SCENARIO_TITLE_COLORS.get(label, "#333333"),
                )
            ax.set_xscale("log")
            ax.set_xlim(left=500, right=100000)
            ax.set_ylim(0, 1.05)
            continue

        subset = subset.sort_values("train_count")
        variants_present = [v for v in VARIANT_ORDER if v in subset["variant"].unique()]

        for variant in variants_present:
            var_data = subset[subset["variant"] == variant]
            if var_data.empty:
                continue

            color = VARIANT_COLORS.get(variant, "#666666")
            marker = VARIANT_MARKERS.get(variant, "x")
            std_col = f"{metric}_std"
            has_std = std_col in var_data.columns and not var_data[std_col].isna().all()

            if has_std:
                ax.errorbar(
                    var_data["train_count"],
                    var_data[metric],
                    yerr=var_data[std_col],
                    marker=marker,
                    linestyle="-",
                    color=color,
                    label=VARIANT_DISPLAY.get(variant, variant),
                    markersize=6,
                    linewidth=1.5,
                    capsize=3,
                    capthick=1,
                    elinewidth=1,
                    alpha=0.9,
                )
            else:
                ax.plot(
                    var_data["train_count"],
                    var_data[metric],
                    marker=marker,
                    linestyle="-",
                    color=color,
                    label=VARIANT_DISPLAY.get(variant, variant),
                    markersize=6,
                    linewidth=1.5,
                )

        ax.set_xscale("log")
        ax.set_xlim(left=500, right=60000)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3, which="both", axis="both")
        if show_subplot_titles:
            ax.text(
                0.5,
                0.98,
                label,
                ha="center",
                va="top",
                transform=ax.transAxes,
                fontsize=10,
                fontweight="bold",
                color=SCENARIO_TITLE_COLORS.get(label, "#333333"),
            )
        if show_xlabel:
            ax.set_xlabel("Training samples", fontsize=9)
        else:
            ax.set_xlabel("")
        if i == 0 and show_ylabel_first:
            ax.set_ylabel(METRIC_DISPLAY.get(metric, metric), fontsize=10)
        else:
            ax.set_ylabel("")

        if legend_handles is None and legend_labels is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

        if show_legend and legend_handles and legend_labels:
            ax.legend(
                handles=legend_handles,
                title="Representation variants",
                fontsize=6,
                title_fontsize=8,
                loc="lower center",
                #bbox_to_anchor=(0.865, 0.90),
                ncol=4,
                framealpha=0.9,
            )

    return legend_handles or [], legend_labels or []


def main() -> None:
    args = parse_args()

    final_leaf_dir = Path(args.final_leaf_dir)
    random_results_dir = Path(args.random_results_dir)
    random_json_path = Path(args.random_json_path)
    replicate_json_path = Path(args.replicate_json_path)
    cross_json_path = Path(args.cross_json_path)
    outpath = Path(args.outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    random_json_path = ensure_random_json(
        random_results_dir,
        random_json_path,
        force=args.force_random_extract,
    )

    leaf_json_files = find_json_files(str(final_leaf_dir))
    if not leaf_json_files:
        raise FileNotFoundError(f"No experiment_results*.json files under {final_leaf_dir}")

    results_df = load_all_results(leaf_json_files, hide_pca=False)
    if results_df.empty:
        raise RuntimeError("No valid rows loaded from final_leaf JSON files")

    random_alias = "GSE273033_random"
    results_df = append_random_split_full_dataset(
        results_df=results_df,
        random_json_path=random_json_path,
        alias_dataset_id=random_alias,
    )

    dataset_groups = dict(DEFAULT_LEAF_GROUPS)
    dataset_groups[random_alias] = "Random Split"

    fig = plt.figure(figsize=(18, 10), constrained_layout=False)
    grid = fig.add_gridspec(2, 3, height_ratios=[1.25, 1.0], hspace=0.28, wspace=0.2)

    panel_a_grid = grid[0, :].subgridspec(2, 1, height_ratios=[6, 1], hspace=0.04)
    ax_a_top = fig.add_subplot(panel_a_grid[0, 0])
    ax_a_bottom = fig.add_subplot(panel_a_grid[1, 0], sharex=ax_a_top)

    panel_b_grid = grid[1, 0].subgridspec(2, 1, height_ratios=[6, 1], hspace=0.04)
    ax_b_top = fig.add_subplot(panel_b_grid[0, 0])
    ax_b_bottom = fig.add_subplot(panel_b_grid[1, 0], sharex=ax_b_top)

    panel_c_grid = grid[1, 1:].subgridspec(1, 3, wspace=0.15)
    ax_c_top_list = []
    ax_c_bottom_list = []
    for i in range(3):
        c_pair_grid = panel_c_grid[0, i].subgridspec(2, 1, height_ratios=[6, 1], hspace=0.04)
        ax_c_top = fig.add_subplot(c_pair_grid[0, 0])
        ax_c_bottom = fig.add_subplot(c_pair_grid[1, 0], sharex=ax_c_top)
        ax_c_top_list.append(ax_c_top)
        ax_c_bottom_list.append(ax_c_bottom)

    plot_specific_train_count_barplot_on_ax(
        ax=ax_a_top,
        results_df=results_df,
        metric=args.metric,
        dataset_groups=dataset_groups,
        train_count="max",
        show_xlabel=False,
        show_ylabel=True,
        show_title=True,
        show_legend=True,
        show_group_labels=True,
        show_family_brackets=True,
    )
    plot_specific_train_count_barplot_on_ax(
        ax=ax_a_bottom,
        results_df=results_df,
        metric=args.metric,
        dataset_groups=dataset_groups,
        train_count="max",
        show_xlabel=True,
        show_ylabel=False,
        show_title=False,
        show_legend=False,
        show_group_labels=False,
        show_family_brackets=False,
    )

    stats_df = plot_panel_d(
        ax_b_top,
        random_json=random_json_path,
        replicate_json=replicate_json_path,
        cross_json=cross_json_path,
        metric=args.metric,
        n_bootstrap=args.panel_b_bootstrap,
    )
    _ = plot_panel_d(
        ax_b_bottom,
        random_json=random_json_path,
        replicate_json=replicate_json_path,
        cross_json=cross_json_path,
        metric=args.metric,
        n_bootstrap=args.panel_b_bootstrap,
    )
    panel_b_title = ax_b_top.get_title()
    ax_b_top.set_title("")
    ax_b_top.set_xlabel("")
    ax_b_bottom.set_title("")
    ax_b_bottom.set_ylabel("")
    b_bottom_legend = ax_b_bottom.get_legend()
    if b_bottom_legend is not None:
        b_bottom_legend.remove()

    panel_c_handles, panel_c_labels = plot_panel_c_mlp_subplots(
        axes=ax_c_top_list,
        results_df=results_df,
        metric=args.metric,
        scenario_dataset_ids=[
            random_alias,
            args.replicate_dataset_id,
            args.cross_dataset_id,
        ],
        scenario_labels=["Random Split", "Replicate Split", "Cross Experiment Split"],
        show_subplot_titles=True,
        show_xlabel=False,
        show_ylabel_first=True,
        show_legend=False,
    )
    _ = plot_panel_c_mlp_subplots(
        axes=ax_c_bottom_list,
        results_df=results_df,
        metric=args.metric,
        scenario_dataset_ids=[
            random_alias,
            args.replicate_dataset_id,
            args.cross_dataset_id,
        ],
        scenario_labels=["Random Split", "Replicate Split", "Cross Experiment Split"],
        show_subplot_titles=False,
        show_xlabel=True,
        show_ylabel_first=False,
        show_legend=True,
    )

    apply_broken_y_axes_pair(ax_a_top, ax_a_bottom, low_max=0.05, high_min=0.4, break_size=0.004)
    apply_broken_y_axes_pair(ax_b_top, ax_b_bottom, low_max=0.05, high_min=0.4)
    for ax_top, ax_bottom in zip(ax_c_top_list, ax_c_bottom_list):
        apply_broken_y_axes_pair(ax_top, ax_bottom, low_max=0.05, high_min=0.4)

    for ax in ax_c_top_list[1:]:
        ax.tick_params(axis="y", labelleft=False)
    for ax in ax_c_bottom_list[1:]:
        ax.tick_params(axis="y", labelleft=False)

    # if panel_c_handles and panel_c_labels:
    #     ax_c_top_list[1].legend(
    #         panel_c_handles,
    #         panel_c_labels,
    #         title="Variant",
    #         loc="upper center",
    #         bbox_to_anchor=(0.5, -0.26),
    #         ncol=2,
    #         fontsize=7,
    #         title_fontsize=9,
    #         framealpha=0.9,
    #     )

    fig.subplots_adjust(left=0.06, right=0.9, bottom=0.08, top=0.95)

    a_bbox = ax_a_top.get_position()
    b_bbox = ax_b_top.get_position()
    c_bbox_left = ax_c_top_list[0].get_position()
    c_bbox_right = ax_c_top_list[2].get_position()
    b_title_x = (b_bbox.x0 + b_bbox.x1) / 2
    c_title_x = (c_bbox_left.x0 + c_bbox_right.x1) / 2

    # Keep row titles close to plots and perfectly aligned across panels b and c.
    lower_title_y = max(b_bbox.y1, c_bbox_left.y1, c_bbox_right.y1) + 0.008
    fig.text(
        b_title_x,
        lower_title_y,
        panel_b_title,
        ha="center",
        va="bottom",
        fontsize=14,
    )

    fig.text(
        c_title_x,
        lower_title_y,
        "Training sample performance across experimental scenarios for GSE273033",
        ha="center",
        va="bottom",
        fontsize=14,
    )

    # Use one consistent offset rule for all panel tags.
    tag_x_offset = 0.025
    tag_y_offset = 0.008
    panel_tag_specs = [
        ("a", a_bbox.x0 - tag_x_offset, a_bbox.y1 + tag_y_offset),
        ("b", b_bbox.x0 - tag_x_offset, b_bbox.y1 + tag_y_offset),
        ("c", c_bbox_left.x0 - tag_x_offset, c_bbox_left.y1 + tag_y_offset),
    ]
    for label, x_pos, y_pos in panel_tag_specs:
        fig.text(
            x_pos,
            y_pos,
            label,
            fontsize=18,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

    fig.savefig(outpath, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".svg"), dpi=args.dpi, format="svg", bbox_inches="tight")
    plt.close(fig)

    stats_path = outpath.with_name(outpath.stem + "_panel_b_stats.csv")
    stats_df.to_csv(stats_path, index=False)

    print(f"Saved composite figure: {outpath}")
    print(f"Saved panel-b stats: {stats_path}")


if __name__ == "__main__":
    main()
