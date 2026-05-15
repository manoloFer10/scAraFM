import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import mlflow
import time
import logging
import json

from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from model.performer.performer import PerformerLM
from tqdm import tqdm
from contextlib import contextmanager
from evaluate.utils import (
    train_logreg, train_ensemble, train_mlp_classifier, train_xgboost,
    train_model_with_dimensionality_reduction, ModelBattery,
    train_scbert_classifier, evaluate_scbert
)
from evaluate.generate_embeddings import generate_both_embeddings_for_dataset

import scanpy as sc
import gc

N_CLASSES = 7

def split_data(X, y, train_count, test_size=0.2, seed=42, is_all=False):
    # Calculate absolute test size
    n_total = len(X)
    n_test = int(n_total * test_size)
    
    if n_test > 2000:
        print(f"Test set size ({n_test}) exceeds 2000. Capping to 2000.")
        n_test = 2000

    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y,
        test_size=n_test,
        stratify=y,
        random_state=seed,
    )

    # sanity check
    n_pool = len(X_pool)
    val_count = 0.1
    if is_all:
        train_count = 0.9
    else:
        need = train_count + val_count* n_pool
        if need > n_pool:
            raise ValueError(
                f"Need {need} rows for train+val but only {n_pool} remain after test split."
            )

    X_train, X_val, y_train, y_val = train_test_split(
        X_pool, y_pool,
        train_size=train_count,
        test_size=val_count,    
        stratify=y_pool,
        random_state=seed,
    )


    print(f"Train/Val/Test sizes: {len(X_train)}/{len(X_val)}/{len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def split_data_replicate(X, y, metadata, train_count, seed=42, is_all=False):
    replicates = metadata['Repeat'].unique()
    test_replicate = replicates[0]  # Choose the first replicate as test
    
    mask_test = metadata['Repeat'] == test_replicate
    X_test = X[mask_test]
    y_test = y[mask_test]

    if len(X_test) > 2000:
        X_test, _, y_test, _ = train_test_split(
            X_test, y_test,
            train_size=2000,
            stratify=y_test,
            random_state=seed,
        )
    
    X_pool = X[~mask_test]
    y_pool = y[~mask_test]
    
    # Sanity check
    n_pool = len(X_pool)
    val_count = 0.1
    if is_all:
        train_count = 0.9
    else:
        need = train_count + val_count* n_pool
        if need > n_pool:
            raise ValueError(
                f"Need {need} rows for train+val but only {n_pool} remain after test split."
            )

    
    X_train, X_val, y_train, y_val = train_test_split(
        X_pool, y_pool,
        train_size=train_count,
        test_size=val_count,
        stratify=y_pool,
        random_state=seed,
    )
    print(f"Train/Val/Test sizes: {len(X_train)}/{len(X_val)}/{len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_data(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled



def get_balanced_indices(y, seed):
    """
    Calculates indices to undersample the majority classes to match the minority class count.
    """
    unique_classes, counts = np.unique(y, return_counts=True)
    min_count = np.min(counts)

    indices = []
    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        if len(cls_indices) > min_count:
            cls_indices = np.random.choice(cls_indices, min_count, replace=False)
        print(f"Class {cls} has been undersampled to {len(cls_indices)} samples.")
        indices.extend(cls_indices)

    return np.array(indices)


def log_metrics_mlflow(metrics, run_name, parent_run_id=None, params=None):
    with mlflow.start_run(run_name=run_name, nested=bool(parent_run_id), parent_run_id=parent_run_id):
        #params
        if params:
            for k, v in params.items():
                mlflow.log_param(k, v)
        
        # metrics
        for k, v in metrics.items():
            if isinstance(v, (int, float, np.floating)):
                mlflow.log_metric(k, float(v))   

        # overall and per-cell
        for k, v in metrics.items():
            if isinstance(v, dict):             
                for subk, subv in v.items():    
                    if isinstance(subv, (int, float, np.floating)):
                        mlflow.log_metric(f"{k}_{subk}", float(subv))



def get_weighted_mean_embeddings(X_emb_per_gene, X_raw, batch_size=500):
    """
    Compute weighted mean embeddings per cell, processing in batches.
    """
    n_samples = X_emb_per_gene.shape[0]
    n_dim = X_emb_per_gene.shape[2]
    X_emb_wm = np.zeros((n_samples, n_dim), dtype=np.float32)
    
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        
        # Load only the batch
        emb_batch = np.array(X_emb_per_gene[start:end])  # Force load this batch
        raw_batch = X_raw.iloc[start:end].values
        
        for i, (emb, weights) in enumerate(zip(emb_batch, raw_batch)):
            weights_sum = np.sum(weights)
            if weights_sum > 0:
                weights = weights / weights_sum
                X_emb_wm[start + i] = np.sum(emb * weights[:, np.newaxis], axis=0)
            else:
                X_emb_wm[start + i] = np.mean(emb, axis=0)
        
        del emb_batch  # Explicitly free
        
    return X_emb_wm


def upload_to_mlflow(predicted, probas, true_labels, run_name, parent_run_id, params):
    metrics = {
        "accuracy": accuracy_score(true_labels, predicted),
        "f1_macro": f1_score(true_labels, predicted, average='macro'),
        "f1_weighted": f1_score(true_labels, predicted, average='weighted'),
        "report": classification_report(true_labels, predicted, output_dict=True)
    }

    if probas is not None:
        try:
            # Check if binary or multiclass
            if probas.shape[1] == 2:
                 # For binary, roc_auc_score expects probability of the positive class
                 metrics["roc_auc"] = roc_auc_score(true_labels, probas[:, 1])
            else:
                 metrics["roc_auc_ovr"] = roc_auc_score(true_labels, probas, multi_class='ovr', average='macro')
                 metrics["roc_auc_ovo"] = roc_auc_score(true_labels, probas, multi_class='ovo', average='macro')
        except Exception as e:
            print(f"Could not calculate ROC AUC for {run_name}: {e}")

    log_metrics_mlflow(metrics, run_name=run_name, parent_run_id=parent_run_id, params=params)


def upload_battery_results(battery_results, battery_probas, true_labels, dataset_id, parent_run_id, params, mode):
    if mode == 'per_gene':
        for model_name, results in battery_results.items():
            model_method = model_name.split('_')[-1] #logreg, xgboost or mlp
            run_name = None
            if 'ensemble' in model_name:
                run_name = f'{dataset_id}_{model_method}_embeddings_ENS'
            elif 'PCA' in model_name:
                pca_dim = model_name.split('_')[1]
                run_name = f'{dataset_id}_{model_method}_embeddings_RD_pca_{pca_dim}'
            
            if run_name:
                params_ = {**params, "method": model_method, "input": "embeddings"}
                probas = battery_probas.get(model_name) if battery_probas else None
                upload_to_mlflow(np.array(results), probas, true_labels, run_name, parent_run_id, params_)
            else:
                print(f"Warning: Unrecognized model name format for 'per_gene' mode: {model_name}. Skipping MLflow upload.")
    else:
        for model_name, results in battery_results.items():
            model_method = model_name #logreg, xgboost or mlp
            run_name = f'{dataset_id}_{model_method}_{mode}'
            input_ = 'raw'
            if mode != 'raw': input_ = 'embeddings'
            params_ = {**params, "method": model_method, "input": input_}
            probas = battery_probas.get(model_name) if battery_probas else None
            upload_to_mlflow(np.array(results), probas, true_labels, run_name, parent_run_id, params_)

def create_indexed_memmap(source_mmap, indices, cache_dir='cache'):
    """
    Create a new memmap file containing only the indexed rows.
    This avoids loading everything into RAM.
    Returns (memmap, temp_path) - caller is responsible for cleanup.
    """
    import tempfile
    
    os.makedirs(cache_dir, exist_ok=True)
    indices = np.asarray(indices)
    n_samples = len(indices)
    shape = (n_samples,) + source_mmap.shape[1:]
    
    fd, temp_path = tempfile.mkstemp(dir=cache_dir, suffix='.npy')
    os.close(fd)
    
    result_mmap = np.lib.format.open_memmap(
        temp_path,
        mode='w+',
        dtype=source_mmap.dtype,
        shape=shape
    )
    
    # Copy in batches to avoid loading everything at once
    batch_size = 500
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_indices = indices[start:end]
        result_mmap[start:end] = source_mmap[batch_indices]
    
    result_mmap.flush()
    
    return result_mmap, temp_path

def subset_genes_memmap(source_mmap, gene_indices, cache_dir='cache'):
    """
    Create a new memmap with only selected genes (axis 1).
    Returns (memmap, temp_path).
    """
    import tempfile
    
    os.makedirs(cache_dir, exist_ok=True)
    gene_indices = np.asarray(gene_indices)
    n_samples, _, n_dims = source_mmap.shape
    n_genes_new = len(gene_indices)
    shape = (n_samples, n_genes_new, n_dims)
    
    fd, temp_path = tempfile.mkstemp(dir=cache_dir, suffix='.npy')
    os.close(fd)
    
    result_mmap = np.lib.format.open_memmap(
        temp_path,
        mode='w+',
        dtype=source_mmap.dtype,
        shape=shape
    )
    
    # Process in sample batches
    batch_size = 500
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        # Load batch, subset genes, write back
        batch = np.array(source_mmap[start:end])  # Load batch into RAM
        result_mmap[start:end] = batch[:, gene_indices, :]
        del batch
    
    result_mmap.flush()
    return result_mmap, temp_path


def run_pipeline(scbert_model, dataset_path, dataset_id, data_dict, results_dir, batch_size = 16, parent_run=None, device="cuda",
                 lr_frozen=1e-3, lr_finetune_head=1e-3, lr_finetune_all=1e-4, lr_full=1e-4, train_count=None, seed=42, do_grid_search_cv=False, is_all=False,
                 split_strategy='random'):
    
    temp_files_to_cleanup = []
    pipeline_start_time = time.time()

    has_separate_test_set = 'test_raw' in data_dict

    # Unpack training data from the data_dict
    X_emb_per_gene_full_train = data_dict['train_embeddings_per_gene']
    X_emb_cls_full_train = data_dict['train_embeddings_cls']
    X_raw_full_train = data_dict['train_raw']
    y_full_train = data_dict['train_labels']
    metadata_full_train = data_dict['train_metadata']

    # Balance training data to get indices for sampling
    try:
        balanced_indices = get_balanced_indices(y_full_train, seed=seed)
        print("Balanced training data.")
    except Exception as e:
        print(f"Couldn't balance classes due to: {e}. Proceeding without balancing.")
        balanced_indices = np.arange(len(y_full_train))

    y_balanced = y_full_train[balanced_indices]
    metadata_balanced = metadata_full_train.iloc[balanced_indices]
    indices_for_split = np.arange(len(y_balanced))

    if has_separate_test_set:
        print("Using provided test dataset for evaluation.")
        # Use the full balanced training data for train/val split
        X_pool, y_pool = indices_for_split[:, None], y_balanced
        
        # Sanity check for train/val split
        n_pool = len(X_pool)
        val_count = 0.1
        if is_all:
            train_count = 0.9
        else:
            need = train_count + val_count* n_pool
            if need > n_pool:
                raise ValueError(
                    f"Need {need} rows for train+val but only {n_pool} remain after test split."
                )

        train_idx_b, val_idx_b, y_train, y_val = train_test_split(
            X_pool, y_pool,
            train_size=train_count,
            test_size=val_count,
            stratify=y_pool,
            random_state=seed,
        )

        # Unpack the separate test data
        X_test_raw = data_dict['test_raw']
        y_test = data_dict['test_labels']
        test_metadata_raw = data_dict['test_metadata']
        X_test_emb_cls = data_dict['test_embeddings_cls']
        X_test_emb_per_gene = data_dict['test_embeddings_per_gene']

        # Cap test set size to the minimum condition count, stratified
        capped_indices = []
        original_indices = np.arange(len(y_test))
        
        # Determine the minimum count among all conditions
        condition_counts = test_metadata_raw['Condition'].value_counts()
        min_count = condition_counts.min()
        if min_count > 1000: min_count = 1000
        print(f"Capping all test conditions to {min_count}.")
        
        for condition in test_metadata_raw['Condition'].unique():
            condition_mask = (test_metadata_raw['Condition'] == condition).values
            condition_indices = original_indices[condition_mask]
            
            if len(condition_indices) > min_count:
                print(f"Condition '{condition}' has {len(condition_indices)} samples. Capping to {min_count}.")
                capped_cond_indices, _ = train_test_split(
                    condition_indices,
                    train_size=min_count,
                    stratify=y_test[condition_indices],
                    random_state=seed
                )
                capped_indices.extend(capped_cond_indices)
            else:
                print(f"Condition '{condition}' has {len(condition_indices)} samples. No capping needed.")
                capped_indices.extend(condition_indices)
        
        capped_indices = np.sort(np.array(capped_indices))

        # Apply capping
        X_test_raw = X_test_raw.iloc[capped_indices]
        y_test = y_test[capped_indices]
        test_metadata_raw = test_metadata_raw.iloc[capped_indices]
        X_test_emb_cls = X_test_emb_cls[capped_indices]
        X_test_emb_per_gene, tmp = create_indexed_memmap(X_test_emb_per_gene, capped_indices)
        temp_files_to_cleanup.append(tmp)

        print(f"Train/Val/Test sizes: {len(train_idx_b)}/{len(val_idx_b)}/{len(y_test)}")

        # Map train/val indices back to original training data indices
        original_train_locs = balanced_indices[train_idx_b.flatten()]
        original_val_locs = balanced_indices[val_idx_b.flatten()]

        # Create train/val data splits
        X_train_raw, X_val_raw = X_raw_full_train.iloc[original_train_locs], X_raw_full_train.iloc[original_val_locs]
        X_train_emb_cls, X_val_emb_cls = X_emb_cls_full_train[original_train_locs], X_emb_cls_full_train[original_val_locs]
        X_train_emb_per_gene,tmp = create_indexed_memmap(X_emb_per_gene_full_train, original_train_locs)
        temp_files_to_cleanup.append(tmp)
        X_val_emb_per_gene,tmp = create_indexed_memmap(X_emb_per_gene_full_train, original_val_locs)
        temp_files_to_cleanup.append(tmp)

        # Align variable names for the rest of the pipeline
        y_train_raw, y_val_raw, y_test_raw = y_train, y_val, y_test
        y_train_emb, y_val_emb, y_test_emb = y_train, y_val, y_test
        test_metadata_emb = test_metadata_raw

    else:
        print(f"No separate test set provided. Splitting training data with strategy='{split_strategy}'.")
        if split_strategy == 'replicate':
            if 'Repeat' not in metadata_balanced.columns:
                raise ValueError("split_strategy='replicate' requires a 'Repeat' column in dataset metadata.")
            train_idx_b, val_idx_b, test_idx_b, y_train, y_val, y_test = split_data_replicate(
                X=indices_for_split[:, None], y=y_balanced, metadata=metadata_balanced,
                train_count=train_count, seed=seed, is_all=is_all,
            )
        else:
            train_idx_b, val_idx_b, test_idx_b, y_train, y_val, y_test = split_data(
                X=indices_for_split[:, None], y=y_balanced,
                train_count=train_count, seed=seed, is_all=is_all,
            )

        # Map back to original indices
        original_train_locs = balanced_indices[train_idx_b.flatten()]
        original_val_locs = balanced_indices[val_idx_b.flatten()]
        original_test_locs = balanced_indices[test_idx_b.flatten()]

        # Create data splits for raw and cls embeddings from memory
        X_train_raw, X_val_raw, X_test_raw = X_raw_full_train.iloc[original_train_locs], X_raw_full_train.iloc[original_val_locs], X_raw_full_train.iloc[original_test_locs]
        y_train_raw, y_val_raw, y_test_raw = y_train, y_val, y_test

        X_train_emb_cls, X_val_emb_cls, X_test_emb_cls = X_emb_cls_full_train[original_train_locs], X_emb_cls_full_train[original_val_locs], X_emb_cls_full_train[original_test_locs]
        y_train_emb, y_val_emb, y_test_emb = y_train, y_val, y_test

        # Load subsets for per-gene embeddings from disk
        X_train_emb_per_gene, tmp = create_indexed_memmap(X_emb_per_gene_full_train, original_train_locs)
        temp_files_to_cleanup.append(tmp)
        X_val_emb_per_gene, tmp = create_indexed_memmap(X_emb_per_gene_full_train, original_val_locs)
        temp_files_to_cleanup.append(tmp)
        X_test_emb_per_gene, tmp = create_indexed_memmap(X_emb_per_gene_full_train, original_test_locs)
        temp_files_to_cleanup.append(tmp)

        test_idx = X_test_raw.index
        test_metadata_raw = metadata_full_train.loc[test_idx]
        test_metadata_emb = test_metadata_raw

    params = {"dataset_path": dataset_path, "train_count": train_count}
    
    # Align columns before scaling if they differ
    if not X_train_raw.columns.equals(X_test_raw.columns):
        print("Train and test sets have different feature sets. Aligning to common features.")
        common_genes = X_train_raw.columns.intersection(X_test_raw.columns)
        print(f"Number of common genes: {len(common_genes)}")
        
        # Get indices for per-gene embeddings
        train_gene_indices = X_train_raw.columns.get_indexer_for(common_genes)
        test_gene_indices = X_test_raw.columns.get_indexer_for(common_genes)

        # Subset raw data
        X_train_raw = X_train_raw[common_genes]
        X_val_raw = X_val_raw[common_genes]
        X_test_raw = X_test_raw[common_genes]

        # Subset per-gene embeddings
        X_train_emb_per_gene, tmp = subset_genes_memmap(X_train_emb_per_gene, train_gene_indices)
        temp_files_to_cleanup.append(tmp)
        X_val_emb_per_gene, tmp = subset_genes_memmap(X_val_emb_per_gene, train_gene_indices)
        temp_files_to_cleanup.append(tmp)
        X_test_emb_per_gene, tmp = subset_genes_memmap(X_test_emb_per_gene, test_gene_indices)
        temp_files_to_cleanup.append(tmp)

    #X_train_raw, X_val_raw, X_test_raw = scale_data(X_train_raw, X_val_raw, X_test_raw)
    test_metadata_emb = test_metadata_raw
    
    start_time = time.time()
    X_train_emb_per_gene_wm = get_weighted_mean_embeddings(X_train_emb_per_gene, X_train_raw)
    X_val_emb_per_gene_wm = get_weighted_mean_embeddings(X_val_emb_per_gene, X_val_raw)
    X_test_emb_per_gene_wm = get_weighted_mean_embeddings(X_test_emb_per_gene, X_test_raw)
    end_time = time.time()
    
    # Setup logging and save dir
    if is_all:
        train_count = len(X_train_raw)  

    save_dir = os.path.join(results_dir, f'{train_count}_samples')
    os.makedirs(save_dir, exist_ok=True)

    log_dir = os.path.join(results_dir, f'{train_count}_samples')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'timing_log_{dataset_id}_{train_count}.log')
    logger = logging.getLogger(f'PipelineLogger_{dataset_id}_{train_count}')
    logger.setLevel(logging.INFO)
    if not logger.handlers: # prevent duplicate logs if function is called multiple times
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(ch)

    logger.info("Starting battery training...")

    # Train model batteries
    
    # # 1. Train raw battery
    start = time.time()
    battery_raw = ModelBattery(X_train_raw, y_train, mode='raw', do_grid_search_cv=do_grid_search_cv, random_state=seed)
    results_raw = battery_raw.predict(X_test_raw)
    probas_raw = battery_raw.predict_proba(X_test_raw)
    # REMOVED: battery_raw.test_samples = X_test_raw (Prevents saving data inside model file)
    battery_raw.test_labels = y_test_raw
    end = time.time()
    logger.info(f"Battery raw took {end - start:.2f} seconds.")
    upload_battery_results(results_raw, probas_raw, y_test_raw, dataset_id, parent_run, params, mode='raw')
    
    # Extract best params
    best_params_raw = {}
    for name, model in battery_raw.models.items():
        if hasattr(model, 'best_params_'):
             best_params_raw[name] = model.best_params_
        elif hasattr(model, 'get_params'):
             p = model.get_params()
             if name == 'logreg':
                 best_params_raw[name] = {'C': p['C']}
             elif name == 'xgboost':
                 param_grid_keys = ['n_estimators', 'learning_rate', 'max_depth']
                 best_params_raw[name] = {k: p[k] for k in param_grid_keys if k in p}
    
    # # SAVE AND RELEASE RAW
    battery_raw.save(os.path.join(save_dir, f"battery_raw.joblib"))
    del battery_raw
    gc.collect()

    # # 2. Train CLS battery
    # start = time.time()
    # battery_embeddings_cls = ModelBattery(X_train_emb_cls, y_train, mode='cls', do_grid_search_cv=do_grid_search_cv, random_state=seed)
    # results_embeddings_cls = battery_embeddings_cls.predict(X_test_emb_cls)
    # probas_embeddings_cls = battery_embeddings_cls.predict_proba(X_test_emb_cls)
    # #battery_embeddings_cls.test_samples = X_test_emb_cls
    # battery_embeddings_cls.test_labels = y_test_emb
    # end = time.time()
    # logger.info(f"Battery embeddings cls took {end - start:.2f} seconds.")
    # upload_battery_results(results_embeddings_cls, probas_embeddings_cls, y_test_emb, dataset_id, parent_run, params, mode='cls')
    
    # # SAVE AND RELEASE CLS
    # battery_embeddings_cls.save(os.path.join(save_dir, f"battery_embeddings_cls.joblib"))
    # del battery_embeddings_cls
    # gc.collect()

    # 3. Train Per Gene battery
    start = time.time()
    battery_embeddings_per_gene = ModelBattery(X_train_emb_per_gene, y_train, mode='per_gene', do_grid_search_cv=do_grid_search_cv, random_state=seed, params_dict=best_params_raw)
    results_embeddings_per_gene = battery_embeddings_per_gene.predict(X_test_emb_per_gene)
    probas_embeddings_per_gene = battery_embeddings_per_gene.predict_proba(X_test_emb_per_gene)
    #battery_embeddings_per_gene.test_samples = X_test_emb_per_gene
    battery_embeddings_per_gene.test_labels = y_test_emb
    end = time.time()
    logger.info(f"Battery embeddings per gene took {end - start:.2f} seconds.")
    upload_battery_results(results_embeddings_per_gene, probas_embeddings_per_gene, y_test_emb, dataset_id, parent_run, params, mode='per_gene')
    
    # SAVE AND RELEASE PER GENE
    battery_embeddings_per_gene.save(os.path.join(save_dir, f"battery_embeddings_per_gene.joblib"))
    del battery_embeddings_per_gene
    gc.collect()

    # 4. Train WM battery
    start = time.time()
    battery_WM = ModelBattery(X_train_emb_per_gene_wm, y_train, mode='raw', do_grid_search_cv=do_grid_search_cv, random_state=seed)
    results_WM = battery_WM.predict(X_test_emb_per_gene_wm)
    probas_WM = battery_WM.predict_proba(X_test_emb_per_gene_wm)
    #battery_WM.test_samples = X_test_emb_per_gene_wm
    battery_WM.test_labels = y_test_emb
    end = time.time()
    logger.info(f"Battery weighted mean embeddings took {end - start:.2f} seconds.")
    upload_battery_results(results_WM, probas_WM, y_test_emb, dataset_id, parent_run, params, mode='WM')

    # SAVE AND RELEASE WM
    battery_WM.save(os.path.join(save_dir, f"battery_WM.joblib"))
    del battery_WM
    gc.collect()

    # scBERT finetuning
    start_time = time.time()
    logger.info("Starting scBERT finetuning...")
    
    scbert_classifier = train_scbert_classifier(
        scbert_model, 
        X_train_raw, y_train_raw, 
        X_val_raw, y_val_raw, 
        ckpt_dir=os.path.join(results_dir, f'{train_count}_samples', 'scbert_classifier'), 
        device=device, 
        lr=lr_finetune_head, 
        batch_size=batch_size, 
        random_state=seed
    )
    
    y_pred_scbert, y_proba_scbert, y_true_scbert = evaluate_scbert(
        scbert_classifier, 
        X_test_raw, y_test_raw,
        metadata=test_metadata_raw, 
        device=device
    )
    
    params_scbert = {
        "strategy": 'scbert_finetune',
        "dataset_path": dataset_path,
        "train_count": train_count,
        "lr_frozen": lr_frozen,
        "lr_finetune_head": lr_finetune_head,
        "lr_finetune_all": lr_finetune_all,
        "lr_full": lr_full
    }

    #save scBERT predictions
    scbert_predictions_path = os.path.join(os.path.join(save_dir, "scbert_classifier"),"scbert_predictions.json")
    os.makedirs(os.path.dirname(scbert_predictions_path), exist_ok=True)
    with open(scbert_predictions_path, 'w') as f:
        json.dump({
            "y_pred": y_pred_scbert.tolist(),
            "y_proba": y_proba_scbert.tolist(),
            "y_true": y_true_scbert.tolist()
        }, f)

    upload_to_mlflow(y_pred_scbert, y_proba_scbert, y_true_scbert, run_name=f"{dataset_id}_scbert_finetune", parent_run_id=parent_run, params=params_scbert)
    
    end_time = time.time()
    logger.info(f"scBERT finetuning took {end_time - start_time:.2f} seconds.")

    pipeline_end_time = time.time()
    logger.info(f"Total pipeline execution time: {pipeline_end_time - pipeline_start_time:.2f} seconds.")

    # lastly clean tmp files
    for tmp_path in temp_files_to_cleanup:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception as e:
            print(f"Warning: Could not remove temp file {tmp_path}: {e}")


@contextmanager
def init_experiment_run(exp_name: str, run_name: str, tags=None):
    """
    Activates exp_name (creating it if needed) and opens a parent run.
    Yields the run_id so nested runs can attach to it.
    """
    mlflow.set_tracking_uri("file:./mlruns")     
    mlflow.set_experiment(exp_name)              
    with mlflow.start_run(run_name=run_name, tags=tags) as parent:
        yield parent.info.run_id

### CLI
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--test_dataset_path", default=None)
    parser.add_argument("--dataset_id", required=True)
    parser.add_argument("--test_dataset_id", default=None)
    parser.add_argument("--results_dir", type=str, required=True, help="Directory to save trained models.")
    parser.add_argument("--ckpt_path", required=True, help="Path to the pretrained PerformerLM checkpoint (.pth) to finetune from.")
    parser.add_argument("--split_strategy", choices=["random", "replicate"], default="random",
                        help="How to build the train/val/test split when no separate test set is provided. "
                             "'replicate' requires a 'Repeat' column in the h5ad obs metadata.")
    parser.add_argument("--model_style", required=True, help = "What base model is used for finetuning (just for exp name). Eg: from_normal, from_stressed, from_combined, etc.")
    parser.add_argument("--n_genes", type = int, default = 4000)
    parser.add_argument("--lr_frozen", type=float, default=1e-3)
    parser.add_argument("--lr_finetune_head", type=float, default=1e-3)
    parser.add_argument("--lr_finetune_all", type=float, default=1e-4)
    parser.add_argument("--lr_full", type=float, default=1e-4)
    parser.add_argument("--train_count", type=str, default=None, help="Amount of samples to use for train+val (e.g., 10 for 10%)")
    parser.add_argument("--gene_csv", type=str, required=True, help="Path to genes used by the model.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--do_grid_search_cv", action='store_true', help="Whether to do grid search CV for classical models.")
    parser.add_argument("--tissue", choices=["Root", "Leaf"], default="Root",
                        help="Tissue type; selects the matching cell-type markers for embedding generation.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- Load Model ---
    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert os.path.exists(args.ckpt_path), f"--ckpt_path does not exist: {args.ckpt_path}"
    gene_list = list(pd.read_csv(args.gene_csv)['gene_name'])
    n_genes = len(gene_list)
    tissue = args.tissue
    model = PerformerLM.from_checkpoint(args.ckpt_path, n_genes, gene_csv=args.gene_csv, device=device)
    model.eval()
    print(f"Loaded model from {args.ckpt_path}")

    def load_data_dict(datapath, dataset_id, model, device, gene_list, n_classes, mode=''):
        print(f"Loading data and generating embeddings for {datapath}...")
        adata = sc.read_h5ad(datapath)
        
        # From load_raw_expression
        raw_expression = pd.DataFrame(adata.X.toarray(), index=adata.obs_names, columns=adata.var_names)
        # Align to gene_list to ensure consistency with embeddings
        raw_expression = raw_expression.reindex(columns=gene_list, fill_value=0)
        metadata = adata.obs

        # From load_labels
        if 'Genotype' in adata.obs.columns:
            string_labels = adata.obs['Condition'].astype(str) + ' - ' + adata.obs['Genotype'].astype(str)
        else:
            string_labels = adata.obs['Condition'].astype(str)
        print(pd.Series(string_labels.unique(), name="Classes").to_frame())
        labels = pd.get_dummies(string_labels).astype(int).values.argmax(axis=1)

        # Generate CLS and per-gene embeddings in a single forward pass per batch
        emb_cls, emb_per_gene = generate_both_embeddings_for_dataset(
            adata=adata, model=model,
            device=device, gene_list=gene_list, n_classes=n_classes, tissue=tissue
        )

        data_dict = {
            f'{mode}_embeddings_per_gene': emb_per_gene,
            f'{mode}_embeddings_cls': emb_cls,
            f'{mode}_raw': raw_expression,
            f'{mode}_metadata': metadata,
            f'{mode}_labels': labels
        }

        print(f"Loaded data for dataset {dataset_id} with {len(data_dict[f'{mode}_raw'])} samples and {len(np.unique(data_dict[f'{mode}_labels']))} classes.")

        return data_dict

    data = load_data_dict(args.dataset_path, args.dataset_id, model, device, gene_list, N_CLASSES, 'train')

    if args.test_dataset_path is not None:
        print(f"Loading test dataset from {args.test_dataset_path}")
        test_data = load_data_dict(args.test_dataset_path, args.test_dataset_id, model, device, gene_list, N_CLASSES, 'test')
        data.update(test_data)  # merge train and test data dicts

    exp_name = f"{args.model_style}"            
    run_name = f"cross_{args.dataset_id}_supervised_finetuning_pipeline"

    if args.train_count == 'ALL':
        is_all = True
        train_count = 0.9  # will be interpreted as fraction inside the split function
    else:
        is_all = False
        train_count = int(args.train_count)

    with init_experiment_run(exp_name, run_name,
                            tags={"dataset": args.dataset_id}) as parent_id:
        run_pipeline(
            scbert_model=model,
            dataset_path=args.dataset_path,
            dataset_id=args.test_dataset_id,
            data_dict=data,
            parent_run=parent_id,                     
            lr_frozen=args.lr_frozen,
            lr_finetune_head=args.lr_finetune_head,
            lr_finetune_all=args.lr_finetune_all,
            lr_full=args.lr_full,
            train_count=train_count,
            seed=int(args.seed),
            is_all=is_all,
            do_grid_search_cv=args.do_grid_search_cv,
            results_dir=args.results_dir,
            split_strategy=args.split_strategy,
        )
