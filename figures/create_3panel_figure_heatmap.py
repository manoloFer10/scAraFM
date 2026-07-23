#!/usr/bin/env python
"""
Create a 3-panel composite figure for LEAF AUROC results — *heatmap* panel-a variant.

This is identical to ``figures.create_3panel_figure`` except that **panel a** is a heatmap
instead of a barplot:

- rows    = representation variants (raw, CLS, WM, ENS, PCA 1/2/3, finetuned),
- columns = model heads (LR, XGB, MLP), repeated per dataset, datasets grouped by
  experiment setup (Random / Replicate / Cross split) — same visual ordering as the bar
  panel,
- each cell shows the AUROC (color + printed number), with the standard deviation printed
  beneath it in smaller type as ``±std``.

The CNN/scbert ``finetuned`` result keeps its own row; it is placed under the **MLP**
column, and the LR / XGB columns of that row are marked ``-``.

Panels b and c are unchanged.

Run from the repo root so the `evaluate` and `figures` packages are importable:
    python -m figures.create_3panel_figure_heatmap --help
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.transforms import blended_transform_factory

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
from figures.create_composite_figure import (  # noqa: E402
    ensure_random_json,
    plot_panel_d,
)
from figures.create_3panel_figure import (  # noqa: E402
    SCENARIO_TITLE_COLORS,
    VARIANT_ORDER,
    append_random_split_full_dataset,
    apply_broken_y_axes_pair,
    format_dataset_label,
    plot_panel_c_mlp_subplots,
    _select_train_count_rows,
)

# Model heads shown as heatmap columns (CNN is folded into the MLP column for the
# `finetuned` row, mirroring the bar panel's bracket grouping).
HEATMAP_HEADS = ["logreg", "xgboost", "mlp"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create LEAF 3-panel composite figure (heatmap panel a)")
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
        default="evaluation/figure_creation/composite_leaf_3panel_aucroc_heatmap.png",
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
        default=1000,
        help="Number of bootstrap resamples per model for panel b violins",
    )
    p.add_argument(
        "--std-bootstrap",
        type=int,
        default=100,
        help="Number of bootstrap resamples used for the per-cell heatmap ±std",
    )
    p.add_argument("--dpi", type=int, default=300, help="Figure resolution")
    return p.parse_args()


def _format_dataset_label_italic(dataset_id: str) -> str:
    """Like ``format_dataset_label`` but renders the species name in italics."""
    label = format_dataset_label(dataset_id)
    if "P. Syringae" in label:
        label = label.replace("P. Syringae", r"$\mathit{P.\ Syringae}$")
    return label


def _draw_group_heatmap(
    ax: plt.Axes,
    group_name: str,
    datasets: list[str],
    rows: list[str],
    value_lookup: dict[tuple[str, str], float],
    std_lookup: dict[tuple[str, str], float],
    metric: str,
    vmin: float,
    vmax: float,
    cmap,
    show_ylabels: bool,
):
    """Draw one experiment partition (one experiment setup) of the panel-a heatmap."""
    columns = [(ds, mf) for ds in datasets for mf in HEATMAP_HEADS]
    n_cols = len(columns)
    n_rows = len(rows)

    value_matrix = np.full((n_rows, n_cols), np.nan)
    std_matrix = np.full((n_rows, n_cols), np.nan)
    dash_mask = np.zeros((n_rows, n_cols), dtype=bool)

    for r, variant in enumerate(rows):
        for c, (ds, mf) in enumerate(columns):
            if variant == "finetuned":
                # CNN/scbert result lives only in the MLP column; LR/XGB get a dash.
                if mf != "mlp":
                    dash_mask[r, c] = True
                    continue
                key = (ds, "scbert_finetuned")
            else:
                key = (ds, f"{mf}_{variant}")

            if key in value_lookup:
                value_matrix[r, c] = value_lookup[key]
                std_matrix[r, c] = std_lookup.get(key, np.nan)

    masked = np.ma.masked_invalid(value_matrix)
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")

    # Per-cell std rings + value text.
    for r in range(n_rows):
        for c in range(n_cols):
            if dash_mask[r, c]:
                ax.text(c, r, "-", ha="center", va="center", fontsize=10, color="#888888")
                continue
            val = value_matrix[r, c]
            if not np.isfinite(val):
                continue

            norm = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            txt_color = "white" if norm < 0.5 else "black"
            ax.text(
                c, r - 0.12, f"{val:.2f}",
                ha="center", va="center",
                fontsize=7.5, color=txt_color, zorder=4,
            )
            std = std_matrix[r, c]
            if np.isfinite(std):
                ax.text(
                    c, r + 0.24, f"±{std:.2f}",
                    ha="center", va="center",
                    fontsize=5.5, color=txt_color, zorder=4,
                )

    # Separators: thin black between datasets, faint white between model columns.
    for c in range(1, n_cols):
        x = c - 0.5
        if columns[c - 1][0] != columns[c][0]:
            ax.axvline(x=x, color="black", linewidth=1.2)
        else:
            ax.axvline(x=x, color="white", linewidth=0.4, alpha=0.6)
    for r in range(1, n_rows):
        ax.axhline(y=r - 0.5, color="white", linewidth=0.4, alpha=0.6)

    # Ticks / labels.
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(
        [MODEL_FAMILY_DISPLAY.get(mf, mf.upper()) for _ds, mf in columns],
        fontsize=7,
    )
    if show_ylabels:
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels([VARIANT_DISPLAY.get(v, v) for v in rows], fontsize=9)
        ax.set_ylabel("Representation variant", fontsize=11)
    else:
        ax.set_yticks([])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.5, n_cols - 0.5)

    # Per-dataset headers above this partition (x in data, y in axes coords).
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for i, ds in enumerate(datasets):
        center = i * len(HEATMAP_HEADS) + (len(HEATMAP_HEADS) - 1) / 2.0
        ax.text(
            center, 1.012, _format_dataset_label_italic(ds),
            ha="center", va="bottom", fontsize=8, transform=trans,
        )

    # Colored bracket below the partition, spanning all datasets of this experiment
    # grouping, with the scenario name centered beneath it.
    group_color = SCENARIO_TITLE_COLORS.get(group_name, "#444444")
    bracket_y = -0.10
    bracket_tick = 0.025
    x0, x1 = -0.4, n_cols - 0.6
    ax.plot(
        [x0, x0, x1, x1],
        [bracket_y + bracket_tick, bracket_y, bracket_y, bracket_y + bracket_tick],
        transform=trans, color=group_color, linewidth=2.0, clip_on=False, zorder=5,
    )
    ax.text(
        (n_cols - 1) / 2.0, bracket_y - 0.035, group_name,
        ha="center", va="top", fontsize=11, fontweight="bold",
        color=group_color, transform=trans,
    )

    return im


def plot_panel_a_heatmap(
    fig: plt.Figure,
    subspec,
    results_df: pd.DataFrame,
    metric: str,
    dataset_groups: dict[str, str] | None,
) -> list[plt.Axes]:
    """Panel a as 3 partitions (one per experiment setup), each a variants x heads heatmap.

    Returns the list of partition axes (left to right) so the caller can position the panel
    tag and title.
    """
    filtered_df = _select_train_count_rows(results_df, train_count="max")
    if filtered_df.empty:
        ax = fig.add_subplot(subspec)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return [ax]

    all_datasets = sorted(filtered_df["dataset_id"].unique())
    dataset_perf = filtered_df.groupby("dataset_id")[metric].mean().to_dict()
    grouped_datasets = get_grouped_datasets(all_datasets, dataset_groups or {}, include_empty_groups=False)

    # One partition per experiment setup (GROUP_ORDER), datasets sorted by perf descending.
    groups_present: list[tuple[str, list[str]]] = []
    for group in GROUP_ORDER:
        group_ds = [
            ds for ds, g in grouped_datasets
            if g == group and not ds.startswith("__placeholder__")
        ]
        group_ds = sorted(group_ds, key=lambda d: dataset_perf.get(d, -np.inf), reverse=True)
        if group_ds:
            groups_present.append((group, group_ds))

    ungrouped = sorted(
        [ds for ds, g in grouped_datasets if g is None and not ds.startswith("__placeholder__")],
        key=lambda d: dataset_perf.get(d, -np.inf), reverse=True,
    )
    if ungrouped:
        groups_present.append(("Other", ungrouped))

    rows = list(VARIANT_ORDER)

    # Lookup (dataset_id, full_method) -> value / std.
    std_col = f"{metric}_std"
    value_lookup: dict[tuple[str, str], float] = {}
    std_lookup: dict[tuple[str, str], float] = {}
    for _, row in filtered_df.iterrows():
        key = (row["dataset_id"], row["full_method"])
        value_lookup[key] = float(row[metric])
        std_lookup[key] = (
            float(row[std_col]) if std_col in filtered_df.columns and not pd.isna(row[std_col]) else np.nan
        )

    # Shared color scale across all partitions.
    finite_vals = filtered_df[metric].astype(float)
    finite_vals = finite_vals[np.isfinite(finite_vals)]
    vmin = float(np.floor(finite_vals.min() * 20) / 20) if len(finite_vals) else 0.0
    vmax = float(np.ceil(finite_vals.max() * 20) / 20) if len(finite_vals) else 1.0
    if vmax <= vmin:
        vmax = vmin + 0.05

    cmap = plt.get_cmap("cividis").copy()
    cmap.set_bad(color="#f0f0f0")

    n_groups = len(groups_present)
    heat_ratios = [float(len(ds)) for _g, ds in groups_present]
    total_heat = sum(heat_ratios)
    # Dedicate gridspec columns to a spacer + the shared colorbar so the whole panel
    # (heatmaps + gap + colorbar) stays inside grid[0, :] and lines up with the width
    # of panels b/c below, instead of spilling past them.
    spacer_ratio = total_heat * 0.02
    cbar_ratio = total_heat * 0.015
    gs = subspec.subgridspec(
        1,
        n_groups + 2,
        width_ratios=heat_ratios + [spacer_ratio, cbar_ratio],
        wspace=0.06,
    )

    axes: list[plt.Axes] = []
    for gi, (group_name, datasets) in enumerate(groups_present):
        ax = fig.add_subplot(gs[0, gi])
        _draw_group_heatmap(
            ax=ax,
            group_name=group_name,
            datasets=datasets,
            rows=rows,
            value_lookup=value_lookup,
            std_lookup=std_lookup,
            metric=metric,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            show_ylabels=(gi == 0),
        )
        axes.append(ax)

    # Shared colorbar in its own gridspec column. Using `cax=` (rather than `ax=axes`)
    # draws into this fixed column without stealing width from the heatmap axes, so the
    # colorbar stays within grid[0, :] after subplots_adjust.
    cbar_ax = fig.add_subplot(gs[0, n_groups + 1])
    sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(METRIC_DISPLAY.get(metric, metric), fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    return axes


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

    results_df = load_all_results(leaf_json_files, hide_pca=False, n_bootstrap=args.std_bootstrap)
    if results_df.empty:
        raise RuntimeError("No valid rows loaded from final_leaf JSON files")

    random_alias = "GSE273033_random"
    results_df = append_random_split_full_dataset(
        results_df=results_df,
        random_json_path=random_json_path,
        alias_dataset_id=random_alias,
        n_bootstrap=args.std_bootstrap,
    )

    dataset_groups = dict(DEFAULT_LEAF_GROUPS)
    dataset_groups[random_alias] = "Random Split"

    fig = plt.figure(figsize=(18, 10), constrained_layout=False)
    grid = fig.add_gridspec(2, 3, height_ratios=[1.25, 1.0], hspace=0.32, wspace=0.2)

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

    axes_a = plot_panel_a_heatmap(
        fig=fig,
        subspec=grid[0, :],
        results_df=results_df,
        metric=args.metric,
        dataset_groups=dataset_groups,
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

    plot_panel_c_mlp_subplots(
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

    apply_broken_y_axes_pair(ax_b_top, ax_b_bottom, low_max=0.05, high_min=0.4)
    for ax_top, ax_bottom in zip(ax_c_top_list, ax_c_bottom_list):
        apply_broken_y_axes_pair(ax_top, ax_bottom, low_max=0.05, high_min=0.4)

    for ax in ax_c_top_list[1:]:
        ax.tick_params(axis="y", labelleft=False)
    for ax in ax_c_bottom_list[1:]:
        ax.tick_params(axis="y", labelleft=False)

    fig.subplots_adjust(left=0.06, right=0.9, bottom=0.08, top=0.88)

    a_left_bbox = axes_a[0].get_position()
    a_right_bbox = axes_a[-1].get_position()
    b_bbox = ax_b_top.get_position()
    c_bbox_left = ax_c_top_list[0].get_position()
    c_bbox_right = ax_c_top_list[2].get_position()
    b_title_x = (b_bbox.x0 + b_bbox.x1) / 2
    c_title_x = (c_bbox_left.x0 + c_bbox_right.x1) / 2

    # Panel-a title sits above the group/dataset headers so it never overlaps them.
    a_header_top = a_left_bbox.y1 + 0.07 * (a_left_bbox.y1 - a_left_bbox.y0)
    a_title_x = (a_left_bbox.x0 + a_right_bbox.x1) / 2
    a_title_y = a_header_top + 0.03
    tag_y_offset = 0.008
    fig.text(
        a_title_x, a_title_y + tag_y_offset,
        "Leaf perturbation prediction tasks overview",
        ha="center", va="bottom", fontsize=14,
    )

    lower_title_y = max(b_bbox.y1, c_bbox_left.y1, c_bbox_right.y1) + 0.008
    fig.text(b_title_x, lower_title_y, panel_b_title, ha="center", va="bottom", fontsize=14)
    fig.text(
        c_title_x,
        lower_title_y,
        "Training sample performance across experimental scenarios for GSE273033",
        ha="center",
        va="bottom",
        fontsize=14,
    )

    tag_x_offset = 0.025
    panel_tag_specs = [
        ("a", a_left_bbox.x0 - tag_x_offset, a_title_y),
        ("b", b_bbox.x0 - tag_x_offset, b_bbox.y1 + tag_y_offset),
        ("c", c_bbox_left.x0 - tag_x_offset, c_bbox_left.y1 + tag_y_offset),
    ]
    for label, x_pos, y_pos in panel_tag_specs:
        fig.text(
            x_pos, y_pos, label,
            fontsize=18, fontweight="bold", va="bottom", ha="left",
        )

    fig.savefig(outpath, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".svg"), dpi=args.dpi, format="svg", bbox_inches="tight")
    plt.close(fig)

    stats_path = outpath.with_name(outpath.stem + "_panel_b_stats.csv")
    stats_df.to_csv(stats_path, index=False)

    print(f"Saved composite figure: {outpath.resolve()}")
    print(f"Saved panel-b stats: {stats_path.resolve()}")


if __name__ == "__main__":
    main()
