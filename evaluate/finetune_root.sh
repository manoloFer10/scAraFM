#!/bin/bash
# Run from repo root: `bash evaluate/finetune_root.sh`
set -euo pipefail
cd "$(dirname "$0")/.."

GENE_NUM="4397"
GENE_CSV="data/scAraFM_Root/consensus_hvg.csv"
CKPT_PATH="model/weights/scAraFM_root/artifacts/checkpoints/best_model_32.pth"
DATA_DIR="data/scAraFM_Root/test/subset_h5ad"

export CUDA_VISIBLE_DEVICES=0

# ============ RANDOM SPLIT ============

TRAIN_IDS="GSE235495 SRP148288 SRP166333 SRP169576 SRP285817"
for TRAIN_DATASET_ID in $TRAIN_IDS
do
    TRAIN_DATASET_PATH="${DATA_DIR}/${TRAIN_DATASET_ID}_subset.h5ad"
    echo "Fine-tuning ROOT model on dataset $TRAIN_DATASET_ID (random split)"
    for COUNT in ALL
    do
        echo "... with train_count=$COUNT"
        python -m evaluate.finetune \
            --dataset_path $TRAIN_DATASET_PATH \
            --dataset_id $TRAIN_DATASET_ID \
            --ckpt_path $CKPT_PATH \
            --split_strategy random \
            --lr_frozen 1e-3 \
            --lr_finetune_head 1e-4 \
            --lr_finetune_all 1e-5 \
            --lr_full 1e-4 \
            --model_style Root_$TRAIN_DATASET_ID \
            --train_count $COUNT \
            --n_genes $GENE_NUM \
            --gene_csv $GENE_CSV \
            --tissue Root \
            --seed 42 \
            --do_grid_search_cv \
            --results_dir "results/root/$TRAIN_DATASET_ID"
    done
done

# ============ REPLICATE SPLIT ============
# Datasets with a 'Repeat' column in obs; selected via --split_strategy replicate.

TRAIN_IDS="GSE235495"
for TRAIN_DATASET_ID in $TRAIN_IDS
do
    TRAIN_DATASET_PATH="${DATA_DIR}/${TRAIN_DATASET_ID}_subset.h5ad"
    echo "Fine-tuning ROOT model on dataset $TRAIN_DATASET_ID (replicate split)"
    for COUNT in ALL
    do
        echo "... with train_count=$COUNT"
        python -m evaluate.finetune \
            --dataset_path $TRAIN_DATASET_PATH \
            --dataset_id $TRAIN_DATASET_ID \
            --ckpt_path $CKPT_PATH \
            --split_strategy replicate \
            --lr_frozen 1e-3 \
            --lr_finetune_head 1e-4 \
            --lr_finetune_all 1e-5 \
            --lr_full 1e-4 \
            --model_style Root_${TRAIN_DATASET_ID}_replicate \
            --train_count $COUNT \
            --n_genes $GENE_NUM \
            --gene_csv $GENE_CSV \
            --tissue Root \
            --seed 42 \
            --do_grid_search_cv \
            --results_dir "results/root_replicate/${TRAIN_DATASET_ID}_replicate"
    done
done

# ============ EXTRACT JSON RESULTS ============
# Writes <results_dir>/experiment_results.json for each finetune output above.

for TRAIN_DATASET_ID in GSE235495 SRP148288 SRP166333 SRP169576 SRP285817; do
    python -m evaluate.extract_results --results_dir "results/root/$TRAIN_DATASET_ID"
done

for TRAIN_DATASET_ID in GSE235495; do
    python -m evaluate.extract_results --results_dir "results/root_replicate/${TRAIN_DATASET_ID}_replicate"
done
