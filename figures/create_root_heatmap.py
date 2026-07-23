#!/usr/bin/env python
"""
Create the panel-a *heatmap* (variants x model heads) for ROOT AUROC results, as a
standalone single-panel figure.

This reuses :func:`figures.create_3panel_figure_heatmap.plot_panel_a_heatmap` — the exact
same heatmap rendering used as panel a of the leaf 3-panel composite — but driven by the
root result JSONs under ``results/final_root`` instead of the leaf ones. Root has no
random/replicate/cross *scenario* data the way leaf does, so the b/c panels are omitted and
only the heatmap is drawn.

- rows    = representation variants (raw, CLS, WM, ENS, PCA 1/2/3, finetuned),
- columns = model heads (LR, XGB, MLP), repeated per dataset, datasets grouped by
  experiment setup (Random / Replicate split),
- each cell shows the AUROC (color + printed number), with the standard deviation printed
  beneath it in smaller type as ``±std``.

Run from the repo root so the `evaluate` and `figures` packages are importable:
    python -m figures.create_root_heatmap --help
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# Ensure repository root is importable when running this file directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate.eval_finetune_json import (  # noqa: E402
    DEFAULT_LEAF_GROUPS,
    find_json_files,
    load_all_results,
)
from figures.create_3panel_figure_heatmap import plot_panel_a_heatmap  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create ROOT heatmap figure (panel-a only)")
    p.add_argument(
        "--final-root-dir",
        type=str,
        default="results/final_root",
        help="Directory containing ROOT JSON result folders",
    )
    p.add_argument(
        "--metric",
        type=str,
        default="aucroc",
        choices=["accuracy", "f1_macro", "f1_weighted", "aucroc"],
        help="Metric used in the heatmap",
    )
    p.add_argument(
        "--outpath",
        type=str,
        default="evaluation/figure_creation/root_heatmap_aucroc.png",
        help="Output figure path",
    )
    p.add_argument(
        "--std-bootstrap",
        type=int,
        default=50,
        help="Number of bootstrap resamples used for the per-cell heatmap ±std",
    )
    p.add_argument("--dpi", type=int, default=300, help="Figure resolution")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    final_root_dir = Path(args.final_root_dir)
    outpath = Path(args.outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    root_json_files = find_json_files(str(final_root_dir))
    if not root_json_files:
        raise FileNotFoundError(f"No experiment_results*.json files under {final_root_dir}")

    results_df = load_all_results(root_json_files, hide_pca=False, n_bootstrap=args.std_bootstrap)
    if results_df.empty:
        raise RuntimeError("No valid rows loaded from final_root JSON files")

    dataset_groups = dict(DEFAULT_LEAF_GROUPS)

    fig = plt.figure(figsize=(14, 4.5), constrained_layout=False)
    grid = fig.add_gridspec(1, 1)

    axes_a = plot_panel_a_heatmap(
        fig=fig,
        subspec=grid[0, 0],
        results_df=results_df,
        metric=args.metric,
        dataset_groups=dataset_groups,
    )

    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.18, top=0.80)

    a_left_bbox = axes_a[0].get_position()
    a_right_bbox = axes_a[-1].get_position()
    a_header_top = a_left_bbox.y1 + 0.07 * (a_left_bbox.y1 - a_left_bbox.y0)
    a_title_x = (a_left_bbox.x0 + a_right_bbox.x1) / 2
    a_title_y = a_header_top + 0.05
    fig.text(
        a_title_x, a_title_y,
        "Root prediction tasks overview",
        ha="center", va="bottom", fontsize=14,
    )

    fig.savefig(outpath, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".svg"), dpi=args.dpi, format="svg", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved root heatmap figure: {outpath.resolve()}")


if __name__ == "__main__":
    main()
