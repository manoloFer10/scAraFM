#!/usr/bin/env python
"""
Create the training-sweep subplot-matrix figure with bootstrapped error bars.

Layout (ported from scPlantFoundation's ``create_subplot_matrix``):
- Rows    = datasets (optionally grouped by split type, with side brackets)
- Columns = model families (LR, XGB, MLP — scBERT/CNN merged into the MLP column)
- Each cell = metric vs. train_count, one line per representation variant.

The "bootstrapping" matches :mod:`figures.create_3panel_figure`: results are loaded
with ``n_bootstrap`` forwarded through
:func:`evaluate.eval_finetune_json.load_all_results` →
:func:`evaluate.eval_finetune_json.compute_metrics`, which populates the
``<metric>_std`` columns. Those stds are drawn as error bars on each point, so the
shaded uncertainty is bootstrap-resampled rather than coming from raw fold spread.

Run from the repo root so the `evaluate` and `figures` packages are importable:
    python -m figures.create_subplot_matrix_figure --help

`ensure_random_json` is imported from `create_composite_figure`;
`append_random_split_full_dataset` / `format_dataset_label` from `create_3panel_figure`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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
)
from figures.create_composite_figure import ensure_random_json  # noqa: E402
from figures.create_3panel_figure import (  # noqa: E402
    append_random_split_full_dataset,
    format_dataset_label,
)


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

GROUP_COLORS = {
    "Random Split": "#4CAF50",
    "Replicate Split": "#2196F3",
    "Cross Experiment Split": "#FF5722",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create the bootstrapped training-sweep subplot-matrix figure"
    )
    p.add_argument(
        "--results-dir",
        type=str,
        default="results/final_leaf",
        help="Directory containing experiment_results*.json files",
    )
    p.add_argument(
        "--metric",
        type=str,
        default="aucroc",
        choices=["accuracy", "f1_macro", "f1_weighted", "aucroc"],
        help="Metric to plot",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default="evaluation/figure_creation",
        help="Directory to write the figure into",
    )
    p.add_argument(
        "--n-bootstrap",
        type=int,
        default=10,
        help="Number of bootstrap resamples per model (drives the <metric>_std error bars)",
    )
    p.add_argument(
        "--no-groups",
        action="store_true",
        help="Disable split-type row grouping/brackets (plain dataset ordering)",
    )
    p.add_argument(
        "--random-results-dir",
        type=str,
        default="results/GSE_RANDOM_SPLIT/GSE273033",
        help="Raw random-split finetune result directory for GSE273033 (set empty to skip)",
    )
    p.add_argument(
        "--random-json-path",
        type=str,
        default="results/GSE_RANDOM_SPLIT/GSE273033/experiment_results.json",
        help="Path for extracted random-split JSON to append as a separate dataset row",
    )
    p.add_argument(
        "--force-random-extract",
        action="store_true",
        help="Re-extract random split JSON even if it already exists",
    )
    p.add_argument("--dpi", type=int, default=150, help="Figure resolution")
    return p.parse_args()


def create_subplot_matrix(
    results_df,
    metric: str,
    outdir: Path,
    dataset_groups: dict | None = None,
    figsize_per_subplot: tuple = (4, 3.5),
    n_bootstrap: int = 10,
    dpi: int = 150,
) -> Path | None:
    """Subplot matrix of metric-vs-train_count, one cell per (dataset, model family).

    ``<metric>_std`` columns (bootstrap-resampled upstream via ``n_bootstrap``) are
    drawn as error bars. ``n_bootstrap`` is accepted only so it can be recorded in the
    output filename; the actual resampling happens at load time.
    """
    if results_df.empty:
        print(f"No data to plot for metric: {metric}")
        return None

    all_datasets = sorted(results_df["dataset_id"].unique())

    if dataset_groups:
        grouped_datasets = get_grouped_datasets(
            all_datasets, dataset_groups, include_empty_groups=False
        )
        datasets = [ds for ds, _ in grouped_datasets]
    else:
        datasets = all_datasets
        grouped_datasets = [(ds, None) for ds in datasets]

    # scbert (CNN) is merged into the mlp column.
    has_scbert = "scbert" in results_df["model_family"].unique()
    model_families = [
        mf
        for mf in ["logreg", "xgboost", "mlp"]
        if mf in results_df["model_family"].unique() or (mf == "mlp" and has_scbert)
    ]

    n_rows = len(datasets)
    n_cols = len(model_families)
    if n_rows == 0 or n_cols == 0:
        print("No rows/columns to plot")
        return None

    if dataset_groups:
        groups_present = []
        for _, group in grouped_datasets:
            if group and (not groups_present or groups_present[-1] != group):
                groups_present.append(group)
        n_groups = len(groups_present)
        header_height = 0.4
    else:
        n_groups = 0
        header_height = 0

    fig_height = figsize_per_subplot[1] * n_rows + header_height * n_groups
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_subplot[0] * n_cols, fig_height),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    current_group = None
    group_start_rows: dict[str, int] = {}
    group_end_rows: dict[str, int] = {}

    for i, (dataset_id, group) in enumerate(grouped_datasets):
        if group:
            if group != current_group:
                group_start_rows[group] = i
                current_group = group
            group_end_rows[group] = i

        is_placeholder = dataset_id.startswith("__placeholder__")

        for j, model_family in enumerate(model_families):
            ax = axes[i, j]

            if is_placeholder:
                ax.text(
                    0.5,
                    0.5,
                    "No data available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=10,
                    color="#999999",
                    fontstyle="italic",
                )
                if i == 0:
                    ax.set_title(
                        MODEL_FAMILY_DISPLAY.get(model_family, model_family.upper()),
                        fontsize=12,
                        fontweight="bold",
                    )
                if j == 0:
                    ax.set_ylabel(f"(No datasets)\n{METRIC_DISPLAY[metric]}", fontsize=10, color="#999999")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_color("#CCCCCC")
                continue

            if model_family == "mlp":
                subset = results_df[
                    (results_df["dataset_id"] == dataset_id)
                    & (results_df["model_family"].isin(["mlp", "scbert"]))
                ].copy()
            else:
                subset = results_df[
                    (results_df["dataset_id"] == dataset_id)
                    & (results_df["model_family"] == model_family)
                ].copy()

            if subset.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                if i == 0:
                    ax.set_title(
                        MODEL_FAMILY_DISPLAY.get(model_family, model_family.upper()),
                        fontsize=12,
                        fontweight="bold",
                    )
                if j == 0:
                    ax.set_ylabel(format_dataset_label(dataset_id), fontsize=10)
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
            ax.set_xlim(left=500, right=45000)
            ax.set_ylim(0, 1.05)
            ax.grid(alpha=0.3, which="both", axis="both")

            if i == 0:
                ax.set_title(
                    MODEL_FAMILY_DISPLAY.get(model_family, model_family.upper()),
                    fontsize=12,
                    fontweight="bold",
                )
            if j == 0:
                ax.set_ylabel(f"{format_dataset_label(dataset_id)}\n{METRIC_DISPLAY[metric]}", fontsize=10)
            if i == n_rows - 1:
                ax.set_xlabel("Train count", fontsize=10)

    # Side brackets per split-type group.
    if dataset_groups and group_start_rows:
        plt.tight_layout(rect=[0.08, 0.05, 1, 0.95])
        fig.canvas.draw()

        for group in GROUP_ORDER:
            if group not in group_start_rows:
                continue

            start_row = group_start_rows[group]
            end_row = group_end_rows[group]
            bbox_top = axes[start_row, 0].get_position()
            bbox_bottom = axes[end_row, 0].get_position()

            y_top = bbox_top.y1
            y_bottom = bbox_bottom.y0
            mid_y = (y_top + y_bottom) / 2
            bracket_x = 0.02
            text_x = 0.01
            cap_length = 0.008
            color = GROUP_COLORS.get(group, "#666666")

            fig.add_artist(
                plt.Line2D(
                    [bracket_x, bracket_x],
                    [y_bottom, y_top],
                    transform=fig.transFigure,
                    color=color,
                    linewidth=3,
                    solid_capstyle="round",
                )
            )
            fig.add_artist(
                plt.Line2D(
                    [bracket_x, bracket_x + cap_length],
                    [y_top, y_top],
                    transform=fig.transFigure,
                    color=color,
                    linewidth=3,
                    solid_capstyle="round",
                )
            )
            fig.add_artist(
                plt.Line2D(
                    [bracket_x, bracket_x + cap_length],
                    [y_bottom, y_bottom],
                    transform=fig.transFigure,
                    color=color,
                    linewidth=3,
                    solid_capstyle="round",
                )
            )
            fig.text(
                text_x,
                mid_y,
                group,
                fontsize=10,
                fontweight="bold",
                rotation=90,
                va="center",
                ha="center",
                color=color,
            )
    else:
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # Build the legend from every variant present across all subplots (including the
    # scbert/CNN "finetuned" variant that only appears in the merged MLP column), so
    # nothing is dropped just because the first subplot lacked it.
    variants_in_plot = [v for v in VARIANT_ORDER if v in results_df["variant"].unique()]
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=VARIANT_COLORS.get(v, "#666666"),
            marker=VARIANT_MARKERS.get(v, "x"),
            linestyle="-",
            markersize=6,
            linewidth=1.5,
        )
        for v in variants_in_plot
    ]
    legend_labels = [VARIANT_DISPLAY.get(v, v) for v in variants_in_plot]

    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            title="Variant",
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=len(legend_labels),
            fontsize=7,
            title_fontsize=9,
            framealpha=0.9,
        )

    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"subplot_matrix_{metric}_bootstrap{n_bootstrap}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def main() -> None:
    args = parse_args()

    results_dir = Path(args.results_dir)
    outdir = Path(args.outdir)

    json_files = find_json_files(str(results_dir))
    if not json_files:
        raise FileNotFoundError(f"No experiment_results*.json files under {results_dir}")

    # Bootstrapping enters here: n_bootstrap is forwarded to compute_metrics so the
    # <metric>_std columns are bootstrap-resampled (same path as create_3panel_figure).
    results_df = load_all_results(json_files, hide_pca=False, n_bootstrap=args.n_bootstrap)
    if results_df.empty:
        raise RuntimeError(f"No valid rows loaded from JSON files under {results_dir}")

    dataset_groups = None if args.no_groups else dict(DEFAULT_LEAF_GROUPS)

    # Optionally append the random-split GSE273033 sweep as its own grouped row.
    if args.random_results_dir and not args.no_groups:
        random_results_dir = Path(args.random_results_dir)
        random_json_path = Path(args.random_json_path)
        if random_results_dir.exists():
            random_json_path = ensure_random_json(
                random_results_dir,
                random_json_path,
                force=args.force_random_extract,
            )
            random_alias = "GSE273033_random"
            results_df = append_random_split_full_dataset(
                results_df=results_df,
                random_json_path=random_json_path,
                alias_dataset_id=random_alias,
                n_bootstrap=args.n_bootstrap,
            )
            if dataset_groups is not None:
                dataset_groups[random_alias] = "Random Split"

    create_subplot_matrix(
        results_df=results_df,
        metric=args.metric,
        outdir=outdir,
        dataset_groups=dataset_groups,
        n_bootstrap=args.n_bootstrap,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
