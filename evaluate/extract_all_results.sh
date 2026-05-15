#!/bin/bash
# Run from repo root: `bash evaluate/extract_all_results.sh`
# Extracts JSON result tables from all finetune experiment folders (root + leaf).
#
# The extraction uses stored test_labels from battery files and loads scBERT
# predictions from scbert_predictions.json saved during training.
set -euo pipefail
cd "$(dirname "$0")/.."

# ============================================
# Process LEAF cross-experiment evaluations
# ============================================
echo "=== Processing LEAF cross-experiment evaluations ==="

# GSE273033 -> ERP132245
TRAIN_ID="GSE273033"
TEST_ID="ERP132245"
RESULTS_DIR="results/final_leaf/${TRAIN_ID}_2_${TEST_ID}"

if [ -d "$RESULTS_DIR" ]; then
    echo "Processing LEAF $TRAIN_ID -> $TEST_ID..."
    python -m evaluate.extract_results \
        --results_dir "$RESULTS_DIR" \
        --output_path "${RESULTS_DIR}/experiment_results_${TRAIN_ID}_to_${TEST_ID}.json"
else
    echo "Skipping LEAF $TRAIN_ID -> $TEST_ID (directory not found)"
fi

# SRP398011 -> GSE226826
TRAIN_ID="SRP398011"
TEST_ID="GSE226826"
RESULTS_DIR="results/final_leaf/${TRAIN_ID}_2_${TEST_ID}"

if [ -d "$RESULTS_DIR" ]; then
    echo "Processing LEAF $TRAIN_ID -> $TEST_ID..."
    python -m evaluate.extract_results \
        --results_dir "$RESULTS_DIR" \
        --output_path "${RESULTS_DIR}/experiment_results_${TRAIN_ID}_to_${TEST_ID}.json"
else
    echo "Skipping LEAF $TRAIN_ID -> $TEST_ID (directory not found)"
fi

# ============================================
# Process LEAF single-dataset experiments
# ============================================
echo ""
echo "=== Processing LEAF experiments ==="

LEAF_IDS="SRP398011 GSE226826 GSE273033 ERP132245"

for DATASET_ID in $LEAF_IDS; do
    RESULTS_DIR="results/final_leaf/$DATASET_ID"
    
    if [ -d "$RESULTS_DIR" ]; then
        echo "Processing LEAF $DATASET_ID..."
        python -m evaluate.extract_results \
            --results_dir "$RESULTS_DIR"
    else
        echo "Skipping LEAF $DATASET_ID (directory not found)"
    fi
done

# ============================================
# Process ROOT single-dataset experiments
# ============================================
echo ""
echo "=== Processing ROOT experiments ==="

ROOT_IDS="GSE235495 SRP148288 SRP166333 SRP169576 SRP285817"

for DATASET_ID in $ROOT_IDS; do
    RESULTS_DIR="results/final_root/$DATASET_ID"
    
    if [ -d "$RESULTS_DIR" ]; then
        echo "Processing ROOT $DATASET_ID..."
        python -m evaluate.extract_results \
            --results_dir "$RESULTS_DIR"
    else
        echo "Skipping ROOT $DATASET_ID (directory not found)"
    fi
done

echo ""
echo "=== Extraction complete ==="
echo "JSON files have been saved in their respective results directories"
