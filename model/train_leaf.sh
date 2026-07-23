#!/bin/bash

GENE_NUM=${GENE_NUM:=4054}
DEPTH=${DEPTH:=3}
BATCH_SIZE=${BATCH_SIZE:=20}
EMBEDDING_DIM=${EMBEDDING_DIM:=200}
HEADS=${HEADS:=10}
LEARNING_RATE=${LEARNING_RATE:=1e-4}
MASK_PROBABILITY=${MASK_PROBABILITY:=0.20}
MAX_EPOCHS=${MAX_EPOCHS:=50}


GPU=${GPU:=0}
export CUDA_VISIBLE_DEVICES=$GPU

echo "Running training on gpu=${GPU} with depth=${DEPTH}, n_heads=${HEADS}, embedding_dim =${EMBEDDING_DIM}, batch_size=${BATCH_SIZE}, learning_rate=${LEARNING_RATE}, mask_probability=${MASK_PROBABILITY}..."

export EXPERIMENT_NAME="TRAIN_SCARA_LEAF"

# Ensure we always run from the repo root so that `python -m model.train`
# resolves package imports (model.*, data paths) consistently.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python -m model.train \
    --data_path "data/scAra_Leaf/pretrain/subset" \
    --compile \
    --depth $DEPTH \
    --gene_num $GENE_NUM \
    --heads $HEADS \
    --embedding_dim $EMBEDDING_DIM \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --num_workers 8 \
    --mask_probability $MASK_PROBABILITY \
    --max_epochs $MAX_EPOCHS \
    --binning_style "default" \
    --gene_csv "data/scAra_Leaf/pretrain/subset/consensus_hvg.csv" \
    "$@"
