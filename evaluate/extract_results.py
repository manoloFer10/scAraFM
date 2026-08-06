#!/usr/bin/env python
"""
Script to extract test results from finetune experiments and save them as JSON.

This script loads the joblib battery files and the stored predictions/labels
and outputs a structured JSON with:
- Train count as the top-level key
- Experiment metadata
- Individual predictions from each model + ground truth

The script uses stored test_labels from battery files as ground truth and
loads scBERT predictions from scbert_predictions.json saved during training.

Usage (run from repo root):
    python -m evaluate.extract_results \
        --results_dir results/final_root/GSE235495 \
        --output_path results/final_root/GSE235495/experiment_results.json
"""

import argparse
import glob
import json
import os
from collections import OrderedDict

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_battery_predictions(battery_path):
    """Load predictions and test labels from a battery joblib file."""
    battery = joblib.load(battery_path)
    
    result = {
        'mode': battery.mode,
        'predictions': {},
        'model_info': {},
        'test_labels': None
    }
    
    # Get predictions (already computed and stored)
    if hasattr(battery, 'preds') and battery.preds:
        result['predictions'] = {k: [int(p) for p in v] for k, v in battery.preds.items()}
    
    # Get test labels (stored during training - these are the ground truth)
    if hasattr(battery, 'test_labels') and battery.test_labels is not None:
        result['test_labels'] = [int(y) for y in battery.test_labels]
    
    # Get model info
    for model_name, model in battery.models.items():
        model_info = {'type': type(model).__name__}
        if hasattr(model, 'best_params_'):
            model_info['best_params'] = model.best_params_
        elif hasattr(model, 'get_params'):
            try:
                params = model.get_params()
                # Filter to relevant params
                if 'logreg' in model_name.lower():
                    model_info['params'] = {'C': params.get('C')}
                elif 'xgboost' in model_name.lower():
                    model_info['params'] = {
                        'n_estimators': params.get('n_estimators'),
                        'learning_rate': params.get('learning_rate'),
                        'max_depth': params.get('max_depth')
                    }
            except:
                pass
        result['model_info'][model_name] = model_info
    
    return result


def load_scbert_predictions(scbert_dir, sample_dir):
    """
    Load scBERT predictions from the scbert_predictions.json file saved during training.
    
    Args:
        scbert_dir: Directory containing scbert model (scbert_classifier/)
        sample_dir: Parent directory (e.g., 40672_samples/) - fallback location for predictions
        
    Returns:
        Dictionary with y_pred, y_proba, y_true
    """
    # Check both possible locations for the predictions file
    predictions_file = os.path.join(scbert_dir, 'scbert_predictions.json')
    if not os.path.exists(predictions_file):
        # Fallback: check in the parent sample directory
        predictions_file = os.path.join(sample_dir, 'scbert_predictions.json')
    
    if not os.path.exists(predictions_file):
        return None
    
    with open(predictions_file, 'r') as f:
        data = json.load(f)
    
    return {
        'y_pred': data.get('y_pred', []),
        'y_proba': data.get('y_proba', []),
        'y_true': data.get('y_true', [])
    }


def extract_experiment_results(results_dir):
    """
    Extract all experiment results from a results directory.
    
    Args:
        results_dir: Directory containing experiment results
    
    Returns a dictionary with train_count as keys.
    """
    all_results = OrderedDict()
    
    # Find all sample directories
    sample_dirs = sorted(glob.glob(os.path.join(results_dir, '*_samples')))
    
    for sample_dir in tqdm(sample_dirs, desc="Processing sample directories"):
        dir_name = os.path.basename(sample_dir)
        train_count_str = dir_name.replace('_samples', '')
        
        print(f"Processing {sample_dir}...")
        
        experiment_data = {
            'train_count_str': train_count_str,
            'metadata': {
                'sample_dir': sample_dir
            },
            'batteries': {},
            'ground_truth': None
        }
        
        # Load all battery files to get stored test_labels and predictions
        battery_files = glob.glob(os.path.join(sample_dir, 'battery_*.joblib'))
        stored_test_labels = None
        
        for battery_file in battery_files:
            battery_name = os.path.basename(battery_file).replace('.joblib', '')
            try:
                battery_results = load_battery_predictions(battery_file)
                experiment_data['batteries'][battery_name] = battery_results
                print(f"  Loaded {battery_name}")
                
                # Use test_labels from first battery that has them
                if stored_test_labels is None and battery_results.get('test_labels') is not None:
                    stored_test_labels = battery_results['test_labels']
                    print(f"    -> Using test_labels from {battery_name} ({len(stored_test_labels)} samples)")
                    
            except Exception as e:
                print(f"  Warning: Could not load {battery_name}: {e}")
        
        # Use stored test_labels as ground truth
        if stored_test_labels is not None:
            experiment_data['ground_truth'] = {
                'labels': stored_test_labels
            }
            experiment_data['metadata']['n_test_samples'] = len(stored_test_labels)
            print(f"  Ground truth: {len(stored_test_labels)} samples from battery")
        else:
            print(f"  Warning: No stored test_labels found in any battery file")
        
        # Check for scbert classifier results - load from saved predictions JSON
        scbert_dir = os.path.join(sample_dir, 'scbert_classifier')
        if os.path.exists(scbert_dir):
            # Load hyperparameters
            scbert_hyperparams_file = os.path.join(scbert_dir, 'model_hyperparameters.json')
            if os.path.exists(scbert_hyperparams_file):
                try:
                    with open(scbert_hyperparams_file, 'r') as f:
                        scbert_params = json.load(f)
                    
                    scbert_entry = {
                        'mode': 'scbert',
                        'model_info': scbert_params,
                        'predictions': {},
                        'test_labels': None
                    }
                    
                    # Load predictions from scbert_predictions.json
                    scbert_preds = load_scbert_predictions(scbert_dir, sample_dir)
                    if scbert_preds is not None:
                        scbert_entry['predictions'] = {
                            'scbert': [int(p) for p in scbert_preds['y_pred']]
                        }
                        scbert_entry['probabilities'] = scbert_preds['y_proba']
                        scbert_entry['test_labels'] = [int(y) for y in scbert_preds['y_true']]
                        print(f"  Loaded scBERT predictions: {len(scbert_preds['y_pred'])} samples")
                        
                        # If we didn't get ground truth from batteries, use scbert's
                        if experiment_data['ground_truth'] is None:
                            experiment_data['ground_truth'] = {
                                'labels': scbert_entry['test_labels']
                            }
                            experiment_data['metadata']['n_test_samples'] = len(scbert_entry['test_labels'])
                            print(f"    -> Using test_labels from scBERT ({len(scbert_entry['test_labels'])} samples)")
                    else:
                        print(f"  Warning: scbert_predictions.json not found in {scbert_dir}")
                    
                    experiment_data['batteries']['scbert_classifier'] = scbert_entry
                    
                except Exception as e:
                    print(f"  Warning: Could not load scbert data: {e}")
        
        # Load timing log
        timing_logs = glob.glob(os.path.join(sample_dir, 'timing_log_*.log'))
        if timing_logs:
            with open(timing_logs[0], 'r') as f:
                experiment_data['metadata']['timing_log'] = f.read()
        
        all_results[train_count_str] = experiment_data
    
    return all_results


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    else:
        return obj


def main():
    parser = argparse.ArgumentParser(
        description="Extract test results from finetune experiments and save as JSON"
    )
    parser.add_argument(
        "--results_dir", 
        type=str, 
        required=True,
        help="Directory containing experiment results (e.g., results/final_root/GSE235495)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output JSON file path. Defaults to <results_dir>/experiment_results.json"
    )
    
    args = parser.parse_args()
    
    if args.output_path is None:
        args.output_path = os.path.join(args.results_dir, 'experiment_results.json')
    
    print(f"Extracting results from: {args.results_dir}")
    print(f"Output will be saved to: {args.output_path}")
    
    # Count total sample directories for progress bar
    sample_dirs = sorted(glob.glob(os.path.join(args.results_dir, '*_samples')))
    print(f"Found {len(sample_dirs)} sample directories to process.\n")
    
    results = extract_experiment_results(results_dir=args.results_dir)
    
    # Convert to JSON-serializable format
    results_serializable = convert_to_serializable(results)
    
    # Save to JSON
    out_dir = os.path.dirname(os.path.abspath(args.output_path))
    os.makedirs(out_dir, exist_ok=True)

    with open(args.output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\nResults saved to: {args.output_path}")
    
    # Print summary
    print("\nSummary:")
    for train_count, data in results.items():
        n_batteries = len(data['batteries'])
        n_test = data['metadata'].get('n_test_samples', 'N/A')
        print(f"  {train_count}: {n_batteries} batteries, {n_test} test samples")


if __name__ == "__main__":
    main()
