#!/bin/bash
# Run from repo root: `bash evaluate/finetune_leaf.sh`
set -euo pipefail
cd "$(dirname "$0")/.."

GENE_NUM="4054"
GENE_CSV="data/scAra_Leaf/pretrain/subset/consensus_hvg.csv"
CKPT_PATH="model/weights/scAraFM_leaf/artifacts/checkpoints/best_model_17.pth"
DATA_DIR="data/scAra_Leaf/test/subset"

export CUDA_VISIBLE_DEVICES=0

# ============ RANDOM SPLIT ============

TRAIN_IDS="GSE226826 GSE273033 ERP132245 SRP398011"

for TRAIN_DATASET_ID in $TRAIN_IDS
do
    TRAIN_DATASET_PATH="${DATA_DIR}/${TRAIN_DATASET_ID}_subset.h5ad"
    echo "Fine-tuning Leaf model on dataset $TRAIN_DATASET_ID (random split)"
    for COUNT in 1000 5000 10000 20000 ALL
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
            --model_style Leaf_$TRAIN_DATASET_ID \
            --train_count $COUNT \
            --n_genes $GENE_NUM \
            --gene_csv $GENE_CSV \
            --tissue Leaf \
            --seed 42 \
            --do_grid_search_cv \
            --results_dir "results/leaf/$TRAIN_DATASET_ID"
    done
done

# ============ REPLICATE SPLIT ============

TRAIN_IDS="GSE273033"

for TRAIN_DATASET_ID in $TRAIN_IDS
do
    TRAIN_DATASET_PATH="${DATA_DIR}/${TRAIN_DATASET_ID}_subset.h5ad"
    echo "Fine-tuning Leaf model on dataset $TRAIN_DATASET_ID (random split)"
    for COUNT in 1000 5000 10000 20000 ALL
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
            --model_style Leaf_${TRAIN_DATASET_ID}_replicate \
            --train_count $COUNT \
            --n_genes $GENE_NUM \
            --gene_csv $GENE_CSV \
            --tissue Leaf \
            --seed 42 \
            --do_grid_search_cv \
            --results_dir "results/leaf/${TRAIN_DATASET_ID}_replicate"
    done
done



# ============ CROSS EXPERIMENT SPLIT ============

# P. syringae
TRAIN_DATASET_PATH="${DATA_DIR}/SRP398011_subset.h5ad"
TEST_DATASET_PATH="${DATA_DIR}/GSE226826_subset.h5ad"
TRAIN_DATASET_ID="SRP398011"
TEST_DATASET_ID="GSE226826"

echo "Fine-tuning Leaf model on dataset $TRAIN_DATASET_ID (cross-experiment -> $TEST_DATASET_ID)"
for COUNT in 1000 5000 10000 20000 ALL
do
    echo "... with train_count=$COUNT"
    python -m evaluate.finetune \
        --dataset_path $TRAIN_DATASET_PATH \
        --dataset_id $TRAIN_DATASET_ID \
        --ckpt_path $CKPT_PATH \
        --lr_frozen 1e-3 \
        --lr_finetune_head 1e-4 \
        --lr_finetune_all 1e-5 \
        --lr_full 1e-4 \
        --model_style Leaf_${TRAIN_DATASET_ID}_cross \
        --train_count $COUNT \
        --n_genes $GENE_NUM \
        --gene_csv $GENE_CSV \
        --tissue Leaf \
        --seed 42 \
        --do_grid_search_cv \
        --results_dir "results/leaf_cross/$TRAIN_DATASET_ID" \
        --test_dataset_path $TEST_DATASET_PATH \
        --test_dataset_id $TEST_DATASET_ID
done

# Drought
TRAIN_DATASET_PATH="${DATA_DIR}/GSE273033_subset.h5ad"
TEST_DATASET_PATH="${DATA_DIR}/ERP132245_subset.h5ad"
TRAIN_DATASET_ID="GSE273033"
TEST_DATASET_ID="ERP132245"

echo "Fine-tuning Leaf model on dataset $TRAIN_DATASET_ID (cross-experiment -> $TEST_DATASET_ID)"
for COUNT in 1000 5000 10000 20000 ALL
do
    echo "... with train_count=$COUNT"
    python -m evaluate.finetune \
        --dataset_path $TRAIN_DATASET_PATH \
        --dataset_id $TRAIN_DATASET_ID \
        --ckpt_path $CKPT_PATH \
        --lr_frozen 1e-3 \
        --lr_finetune_head 1e-4 \
        --lr_finetune_all 1e-5 \
        --lr_full 1e-4 \
        --model_style Leaf_${TRAIN_DATASET_ID}_cross \
        --train_count $COUNT \
        --n_genes $GENE_NUM \
        --gene_csv $GENE_CSV \
        --tissue Leaf \
        --seed 42 \
        --do_grid_search_cv \
        --results_dir "results/leaf_cross/$TRAIN_DATASET_ID" \
        --test_dataset_path $TEST_DATASET_PATH \
        --test_dataset_id $TEST_DATASET_ID
done


# ============ EXTRACT JSON RESULTS ============
# Output paths are chosen so the figure scripts pick up each run with the right
# dataset_id (parsed from JSON filename / parent dir, see eval_finetune_json.extract_dataset_id_from_path).
# DEFAULT_LEAF_GROUPS maps "GSE273033" -> Replicate Split, so the random GSE273033 run
# lives under results/GSE_RANDOM_SPLIT/ and the figure scripts alias it as GSE273033_random.

mkdir -p results/final_leaf results/GSE_RANDOM_SPLIT/GSE273033

# Random splits (non-GSE273033) -> Random Split group via DEFAULT_LEAF_GROUPS
for TRAIN_DATASET_ID in GSE226826 ERP132245 SRP398011; do
    python -m evaluate.extract_results \
        --results_dir "results/leaf/$TRAIN_DATASET_ID" \
        --output_path "results/final_leaf/$TRAIN_DATASET_ID/experiment_results.json"
done

# Random split for GSE273033 (kept separate; consumed via --random-json-path)
python -m evaluate.extract_results \
    --results_dir results/leaf/GSE273033 \
    --output_path results/GSE_RANDOM_SPLIT/GSE273033/experiment_results.json

# Replicate split for GSE273033 -> dataset_id "GSE273033" -> Replicate Split group
python -m evaluate.extract_results \
    --results_dir results/leaf/GSE273033_replicate \
    --output_path results/final_leaf/GSE273033/experiment_results.json

# Cross-experiment runs -> dataset_id parsed from "experiment_results_<train>_to_<test>.json"
python -m evaluate.extract_results \
    --results_dir results/leaf_cross/GSE273033 \
    --output_path results/final_leaf/GSE273033_2_ERP132245/experiment_results_GSE273033_to_ERP132245.json
python -m evaluate.extract_results \
    --results_dir results/leaf_cross/SRP398011 \
    --output_path results/final_leaf/SRP398011_2_GSE226826/experiment_results_SRP398011_to_GSE226826.json
