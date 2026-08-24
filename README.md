# scAraFM

This repository contains the source code and download the data required to reproduce the results presented in “On the robustness of scRNA-seq foundation models for plant perturbation response prediction under cross-experiment shift” (under consideration for publication). A preprint of the manuscript is available at [![bioRxiv](https://img.shields.io/badge/bioRxiv-2025.08.31.672925-b31b1b.svg)](https://doi.org/10.64898/2026.08.21.746324)

This repo packages the end-to-end workflow to reproduce pretraining and downstream evaluation.

## 1. Setup
First, clone the repository and create the Conda environment:

```bash
git clone https://github.com/manoloFer10/scAraFM.git
cd scAraFM
conda env create -f environment.yml
conda activate scAraFM
```

A CUDA-capable GPU is required for pretraining and embedding generation. Check 5 for hardware requirements.


## 2. Download data
Run the following command to download the data and pretrained weights. Note that approximately 15 GB of free disk space is required.

```bash
python data/download_data.py
```

This populates [data/](data/) with pretraining splits and supervised evaluation datasets.

Each tissue ships a `consensus_hvg.csv` defining the gene vocabulary.

The download also places pretrained model weights under [model/weights/](model/weights/).

## 3. Pretrain
This pretraining step is optional and may require significant computational time depending on your hardware. Given that the pretrained weights were already downloaded in Step 2, you can safely skip this step and proceed directly to Section 4. To reproduce the pretraining of model backbones from scratch, execute the commands below

```bash
bash model/train_leaf.sh   
bash model/train_root.sh   
```

## 4. Fine-tune and evaluate

The following scripts are provided to fine-tune the downstream classifiers on top of the scAraFM representations and evaluate their performance across the different experimental splits (random, replicate, and cross-experiment)

```bash
bash evaluate/finetune_leaf.sh  
bash evaluate/finetune_root.sh   
```

For each dataset the script:

1. Generates CLS + per-gene embeddings via [evaluate/generate_embeddings.py](evaluate/generate_embeddings.py).
2. Runs `ModelBattery` from [evaluate/utils.py](evaluate/utils.py): LogReg / XGBoost / MLP / ensemble / PCA-reduced variants.
3. Runs the scBERT-style fine-tuning head.

Per-dataset results are written under `results/{root,leaf}/{dataset_id}/`.

Note: If you pretrain your own model and want to evaluate it, update the CKPT_PATH variable at the top of evaluate/finetune_*.sh to point at your new checkpoint.

## 5. Hardware requirements


### GPU

Inference-only forward pass at batch size 8 with a sequence of `n_genes + 1` tokens fits comfortably on a single **24 GB GPU**. Pretraining at batch size 20 with the same architecture is the tighter constraint; we used a single 24 GB GPU.

### Host RAM

**32 GB** is comfortable for all evaluation datasets shipped here.

### Disk (cache)

The dominant resource consumer during fine-tuning is the **per-gene embedding cache** written under `cache/`. It is a dense float32 array of shape `(n_cells, n_genes, embedding_dim)`, so:

```
cache_bytes_per_cell ≈ n_genes × embedding_dim × 4
                     ≈ 3.52 MB / cell  (root, 4397 genes × 200 dim)
                     ≈ 3.24 MB / cell  (leaf, 4054 genes × 200 dim)
```

**Leaf**:

| Dataset    | Cells  | h5ad on disk | Cache   |
|------------|-------:|-------------:|--------:|
| GSE226826  | 67,961 | 105 MB       | 220 GB  |
| SRP398011  | 22,882 | 173 MB       | 74 GB   |
| GSE273033  |  4,035 | 366 MB       | 13 GB   |
| ERP132245  |  2,018 | 15 MB        | 7 GB    |

**Root**:

| Dataset    | Cells  | h5ad on disk | Cache   |
|------------|-------:|-------------:|--------:|
| SRP169576  | 35,665 | 102 MB       | 125 GB  |
| GSE235495  | 24,474 | 69 MB        | 86 GB   |
| SRP148288  | 24,369 | 39 MB        | 86 GB   |
| SRP285817  | 17,553 | 67 MB        | 62 GB   |
| SRP166333  | 16,949 | 85 MB        | 60 GB   |


**Recommended free disk: ≥130 GB for root evaluation, ≥220 GB for leaf evaluation**.

## 6. Generate figures

If evaluate/finetune_leaf.sh is run, then the replication of paper main figures can be done through:

```bash
FIG_ARGS=(
    --final-leaf-dir results/final_leaf
    --random-results-dir results/leaf/GSE273033
    --random-json-path results/GSE_RANDOM_SPLIT/GSE273033/experiment_results.json
    --replicate-json-path results/final_leaf/GSE273033/experiment_results.json
    --cross-json-path results/final_leaf/GSE273033_2_ERP132245/experiment_results_GSE273033_to_ERP132245.json
)

python -m figures.create_3panel_figure_heatmap "${FIG_ARGS[@]}" \
    --outpath figures/composite_leaf_3panel_aucroc_heatmap.png
```

