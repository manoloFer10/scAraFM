# scAraFM

A single-cell foundation model for *Arabidopsis thaliana*. Two tissue-specific Performer-based masked-language-model backbones (root and leaf) are pretrained on scRNA-seq expression bins, then evaluated by fine-tuning on supervised tasks (cell-type and stress-condition classification).

This repo packages the end-to-end workflow to reproduce pretraining and downstream evaluation.

## 1. Setup

```bash
conda env create -f environment.yml
conda activate scAraFM
```

A CUDA-capable GPU is required for pretraining and embedding generation. Check 5 for hardware requirements.


## 2. Download data

```bash
python data_download/download_data.py
```

This populates [data/](data/) with:

- `scAraFM_Root/` — root pretraining splits + supervised evaluation datasets
- `scAra_Leaf/`   — leaf pretraining splits + supervised evaluation datasets

Each tissue ships a `consensus_hvg.csv` defining the gene vocabulary (root: 4397 genes, leaf: 4054 genes).

The download also places pretrained model weights under [model/weights/](model/weights/), so pretraining is optional (see 3).

## 3. Pretrain

```bash
bash model/train_root.sh   # root backbone
bash model/train_leaf.sh   # leaf backbone
```

Both wrappers `cd` to the repo root and invoke `python -m model.train` with the tissue-appropriate `--gene_num` and data paths. Defaults: depth 3, batch size 20, embedding dim 200, 10 heads, LR 1e-4, mask prob 0.20, 50 epochs.

Pretrained weights are downloaded automatically in 2 and live in [model/weights/](model/weights/). You can skip this step and use them directly:

- [model/weights/scAraFM_root/artifacts/checkpoints/best_model_32.pth](model/weights/scAraFM_root/artifacts/checkpoints/best_model_32.pth)
- [model/weights/scAraFM_leaf/artifacts/checkpoints/best_model_17.pth](model/weights/scAraFM_leaf/artifacts/checkpoints/best_model_17.pth)

MLflow logs are written to `./mlruns/` (file backend). Best-by-val checkpoints are saved in `./checkpoints/{run_id}/best_model_{epoch}.pth`.

If you train your own model and want to evaluate it, update the `CKPT_PATH` variable at the top of [evaluate/finetune_root.sh](evaluate/finetune_root.sh) or [evaluate/finetune_leaf.sh](evaluate/finetune_leaf.sh) to point at your new checkpoint before running 4.

## 4. Fine-tune / evaluate

```bash
bash evaluate/finetune_leaf.sh   # 4 leaf datasets:  4 random split + 1 replicate split + 2 cross-experiment blocks
bash evaluate/finetune_root.sh   # 5 root datasets: 5 random split + 1 replicate split blocks
```

Each wrapper iterates over the per-tissue evaluation datasets and calls `python -m evaluate.finetune` with a `--ckpt_path` pointing at a `best_model_*.pth`. For each dataset the script:

1. Generates CLS + per-gene embeddings via [evaluate/generate_embeddings.py](evaluate/generate_embeddings.py) (single forward pass, written to two `cache/`-backed memmaps).
2. Runs `ModelBattery` from [evaluate/utils.py](evaluate/utils.py): logreg / xgboost / MLP / ensemble / PCA-reduced variants.
3. Runs the scBERT-style fine-tuning head.

Per-dataset results are written under `results/{root,leaf}/{dataset_id}/`.

## 5. Hardware requirements

The dominant resource consumer during fine-tuning is the **per-gene embedding cache** written under `cache/`. It is a dense float32 array of shape `(n_cells, n_genes, embedding_dim)`, so:

```
cache_bytes_per_cell ≈ n_genes × embedding_dim × 4
                     ≈ 3.52 MB / cell  (root, 4397 genes × 200 dim)
                     ≈ 3.24 MB / cell  (leaf, 4054 genes × 200 dim)
```

### GPU

Inference-only forward pass at batch size 8 with a sequence of `n_genes + 1` tokens fits comfortably on a single **24 GB GPU**. Pretraining at batch size 20 with the same architecture is the tighter constraint; we used a single 24 GB GPU.

### Host RAM

Peak host RAM is dominated by the AnnData load and the ModelBattery classifiers (xgboost grid search + MLP). Order of magnitude: ~2× the on-disk `.h5ad` size, plus ~4–8 GB for the classifiers. **32 GB** is comfortable for all evaluation datasets shipped here.

### Disk (cache)

Per-dataset cache sizes are computed exactly from the formula above using cell counts read directly from each `.h5ad` file:

**Root** (4397 genes, 3.52 MB cache / cell):

| Dataset    | Cells  | h5ad on disk | Cache   |
|------------|-------:|-------------:|--------:|
| SRP169576  | 35,665 | 102 MB       | 125 GB  |
| GSE235495  | 24,474 | 69 MB        | 86 GB   |
| SRP148288  | 24,369 | 39 MB        | 86 GB   |
| SRP285817  | 17,553 | 67 MB        | 62 GB   |
| SRP166333  | 16,949 | 85 MB        | 60 GB   |

**Leaf** (4054 genes, 3.24 MB cache / cell):

| Dataset    | Cells  | h5ad on disk | Cache   |
|------------|-------:|-------------:|--------:|
| GSE226826  | 67,961 | 105 MB       | 220 GB  |
| SRP398011  | 22,882 | 173 MB       | 74 GB   |
| GSE273033  |  4,035 | 366 MB       | 13 GB   |
| ERP132245  |  2,018 | 15 MB        | 7 GB    |

**Recommended free disk: ≥130 GB for root evaluation, ≥220 GB for leaf evaluation** (per-dataset peak; the cache is written to a temp file that survives if the process is killed by signal — sweep `cache/` before relaunching).

## 6. Reproduce results 

```bash
conda env create -f environment.yml
conda activate scAraFM

python data_download/download_data.py   # data + pretrained weights

bash evaluate/finetune_root.sh          # root evaluation (5 datasets)
bash evaluate/finetune_leaf.sh          # leaf evaluation (4 datasets)


```

Results land under `results/{root,leaf}/{dataset_id}/`. Pretraining is optional; the downloaded weights are used by default (see 3).


## 7. Generate figures
If evaluate/finetune_leaf.sh is ran, then the replication of paper main figures can be done through:

```bash
FIG_ARGS=(
    --final-leaf-dir results/final_leaf
    --random-results-dir results/leaf/GSE273033
    --random-json-path results/GSE_RANDOM_SPLIT/GSE273033/experiment_results.json
    --replicate-json-path results/final_leaf/GSE273033/experiment_results.json
    --cross-json-path results/final_leaf/GSE273033_2_ERP132245/experiment_results_GSE273033_to_ERP132245.json
)

python -m figures.create_composite_figure "${FIG_ARGS[@]}" \
    --outpath figures/composite_leaf_4panel_aucroc.png

python -m figures.create_3panel_figure "${FIG_ARGS[@]}" \
    --outpath figures/composite_leaf_3panel_aucroc.png

```
