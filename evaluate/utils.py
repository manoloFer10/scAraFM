from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import torch
import joblib
import math
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.decomposition import PCA
from evaluate.generate_embeddings import SCDatasetFromDataFrame
from model.train import save_ckpt
from torch.optim.lr_scheduler import _LRScheduler
import os
import glob
import json

torch.manual_seed(42)
np.random.seed(42)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, n_classes, hidden_dim=256, dropout=0.2, depth=2, act_fn='relu'):
        super().__init__()
        
        act_fn_map = {
            'relu': nn.ReLU,
            'sigmoid': nn.Sigmoid
        }
        activation = act_fn_map[act_fn]

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(activation())
        layers.append(nn.Dropout(dropout))
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, n_classes))
        self.model = nn.Sequential(*layers)


    def forward(self, x):
        return self.model(x)
    
    def predict(self, X):
        """
        predict class labels for all samples in X. As X is given in its entirety, we process it in batches to avoid OOM
        we check if X is already a tensor.
        """
        if not isinstance(X, torch.Tensor):
            X_values = X.values if hasattr(X, 'values') else X
            X = torch.tensor(X_values, dtype=torch.float32)
        
        test_ds = TensorDataset(X)
        test_loader = DataLoader(test_ds, batch_size=32)

        self.eval()
        all_preds = []
        with torch.no_grad():
            for (xb,) in test_loader:
                xb = xb.to(next(self.parameters()).device)
                logits = self.forward(xb)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.append(preds)
        return np.concatenate(all_preds)

    def predict_proba(self, X):
        """
        predict class probabilities for all samples in X. As X is given in its entirety, we process it in batches to avoid OOM
        we check if X is already a tensor.
        """
        if not isinstance(X, torch.Tensor):
            X_values = X.values if hasattr(X, 'values') else X
            X = torch.tensor(X_values, dtype=torch.float32)

        test_ds = TensorDataset(X)
        test_loader = DataLoader(test_ds, batch_size=32)

        self.eval()
        all_probas = []
        with torch.no_grad():
            for (xb,) in test_loader:
                xb = xb.to(next(self.parameters()).device)
                logits = self.forward(xb)
                probas = torch.softmax(logits, dim=1).cpu().numpy()
                all_probas.append(probas)
        return np.concatenate(all_probas)        


class Identity(torch.nn.Module):
    def __init__(self, dropout = 0., h_dim = 100, out_dim = 10, SEQ_LEN=4000): #use performerlm.max_seq_len for SEQ_LEN
        super(Identity, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, (1, 200))
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(in_features=SEQ_LEN, out_features=512, bias=True)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(in_features=512, out_features=h_dim, bias=True)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(in_features=h_dim, out_features=out_dim, bias=True)

    def forward(self, x):
        x = x[:,None,:,:]
        x = self.conv1(x)
        x = self.act(x)
        x = x.view(x.shape[0],-1)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

def preprocess_for_scbert(input_data, input_labels, gene_list, batch_size, random_state, shuffle=True):
    """
    Preprocess expression matrices for SCBERT input.
    
    Args:
        input_data: Expression DataFrame
        input_labels: Labels (Series or array)
        gene_list: List of genes
        batch_size: Batch size for DataLoader
        random_state: Random state for reproducibility
        shuffle: Whether to shuffle the data. Set to True for training, False for evaluation.
    """
    input_labels_series = input_labels if isinstance(input_labels, pd.Series) else pd.Series(input_labels, index=input_data.index)

    train_dataset = SCDatasetFromDataFrame(input_data, input_labels_series, gene_list, n_classes=7)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2, shuffle=shuffle)
    return train_dataloader


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


def train_scbert_classifier(scbert, X_train, y_train, X_val, y_val, batch_size = 16, device="cuda", lr=1e-3, epochs=100, gradient_accumulation=60, ckpt_dir='./checkpoints', random_state=42):
    """
    Train a SCBERT classifier.
    scbert: pretrained SCBERT model
    X_train: AnnData object with training data
    """
    #Data setup
    train_loader = preprocess_for_scbert(X_train, y_train, scbert.gene_list, batch_size, random_state)
    val_loader = preprocess_for_scbert(X_val, y_val, scbert.gene_list, batch_size, random_state)    
    
    #Model setup
    for param in scbert.parameters():
        param.requires_grad = False
    for param in scbert.norm.parameters():
        param.requires_grad = True
    for param in scbert.performer.net.layers[-2].parameters():
        param.requires_grad = True

    n_classes = len(np.unique(y_train))
    h_dim = 128
    dropout = 0.2
    scbert.to_out = Identity(dropout = dropout, h_dim = h_dim, out_dim = n_classes, SEQ_LEN=scbert.max_seq_len)
    scbert = scbert.to(device)

    os.makedirs(ckpt_dir, exist_ok=True)
    hyperparameters = {
        "dropout": dropout,
        "h_dim": h_dim,
        "out_dim": int(n_classes),
        "seq_len": int(scbert.max_seq_len)
    }
    with open(os.path.join(ckpt_dir, 'model_hyperparameters.json'), 'w') as f:
        json.dump(hyperparameters, f, indent=4)

    optimizer = torch.optim.Adam(scbert.parameters(), lr=lr)
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=15,
        cycle_mult=2,
        max_lr=lr,
        min_lr=1e-6,
        warmup_steps=5,
        gamma=0.9
    )

    loss_fn = nn.CrossEntropyLoss(weight=None)
    
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

    # Train loop
    os.makedirs(ckpt_dir, exist_ok=True)
    VALIDATE_EVERY = 1
    PATIENCE = 5
    trigger_times = 0
    max_acc = 0.0
    best_ckpt_path = None
    for i in tqdm(range(1, epochs+1), desc="Training SCBERT classifier"):
        scbert.train()
        running_loss = 0.0
        cum_acc = 0.0
        for index, (data, labels) in enumerate(train_loader):
            index += 1
            data, labels = data.to(device), labels.to(device)
            # if index % gradient_accumulation != 0:
            #     with scbert.no_sync():
            #         logits = scbert(data)
            #         loss = loss_fn(logits, labels)
            #         loss.backward()
            #         #manu: this bit (".no_sync()") was intended for DDP, but we are not using it now.
            if index % gradient_accumulation != 0 and index != len(train_loader):
                logits = scbert(data)
                loss = loss_fn(logits, labels)
                loss.backward()
            if index % gradient_accumulation == 0 or index == len(train_loader):
                logits = scbert(data)
                loss = loss_fn(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(scbert.parameters(), int(1e6))
                optimizer.step()
                optimizer.zero_grad()
            running_loss += loss.item()
            softmax = nn.Softmax(dim=-1)
            final = softmax(logits)
            final = final.argmax(dim=-1)
            pred_num = labels.size(0)
            correct_num = torch.eq(final, labels).sum(dim=-1)
            cum_acc += torch.true_divide(correct_num, pred_num).mean().item()
        epoch_loss = running_loss / index
        train_losses.append(epoch_loss)
        epoch_acc = 100 * cum_acc / index
        train_accuracies.append(epoch_acc)

        print(f'    ==  Epoch: {i} | Training Loss: {epoch_loss:.6f} | Accuracy: {epoch_acc:6.4f}%  ==')
        scheduler.step()

        if i % VALIDATE_EVERY == 0:
            scbert.eval()
            running_loss = 0.0
            predictions = []
            truths = []
            with torch.no_grad():
                for index, (data_v, labels_v) in enumerate(val_loader):
                    index += 1
                    data_v, labels_v = data_v.to(device), labels_v.to(device)
                    logits = scbert(data_v)
                    loss = loss_fn(logits, labels_v)
                    running_loss += loss.item()
                    softmax = nn.Softmax(dim=-1)
                    final_prob = softmax(logits)
                    final = final_prob.argmax(dim=-1)
                    #final[np.amax(np.array(final_prob.cpu()), axis=-1) < UNASSIGN_THRES] = -1
                    predictions.append(final)
                    truths.append(labels_v)
                del data_v, labels_v, logits, final_prob, final
                # gather
                predictions = torch.cat(predictions, dim=0)
                truths = torch.cat(truths, dim=0)
                no_drop = predictions != -1
                predictions = np.array((predictions[no_drop]).cpu())
                truths = np.array((truths[no_drop]).cpu())
                cur_acc = accuracy_score(truths, predictions)
                val_accuracies.append(cur_acc)
                f1 = f1_score(truths, predictions, average='macro')
                val_loss = running_loss / index
                val_losses.append(val_loss)
                if cur_acc > max_acc:
                    max_acc = cur_acc
                    print(f'New best at Epoch: {i} | Val Loss: {val_loss:.6f} | Accuracy: {100*cur_acc:6.4f}%  ==')
                    trigger_times = 0
                    best_ckpt_path = save_ckpt(i, scbert, optimizer, scheduler, val_loss, 'best_scbert_classifier', ckpt_dir)
                    if best_ckpt_path is None:
                        pattern = os.path.join(ckpt_dir, 'best_scbert_classifier*')
                        candidates = [path for path in glob.glob(pattern) if os.path.isfile(path)]
                        if candidates:
                            best_ckpt_path = max(candidates, key=os.path.getmtime)
                else:
                    trigger_times += 1
                    if trigger_times > PATIENCE:
                        break

    training_metrics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }
    with open(os.path.join(ckpt_dir, 'training_metrics.json'), 'w') as f:
        json.dump(training_metrics, f, indent=4)

    if not best_ckpt_path or not os.path.exists(best_ckpt_path):
        pattern = os.path.join(ckpt_dir, 'best_scbert_classifier*')
        candidates = [path for path in glob.glob(pattern) if os.path.isfile(path)]
        if candidates:
            best_ckpt_path = max(candidates, key=os.path.getmtime)
    if not best_ckpt_path or not os.path.exists(best_ckpt_path):
        raise FileNotFoundError(f"Best checkpoint not found in {ckpt_dir}")
    checkpoint = torch.load(best_ckpt_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict') or checkpoint.get('state_dict')
    if state_dict is None:
        raise KeyError("Checkpoint missing model state dict.")
    scbert.load_state_dict(state_dict)
    scbert = scbert.to(device)
    scbert.eval()
    return scbert

def evaluate_scbert(scbert, X_test, y_test, metadata, device=None, batch_size=16, random_state=42):
    if hasattr(scbert, 'eval'):
        scbert.eval()

    # Use shuffle=False for evaluation to preserve sample order
    dataloader = preprocess_for_scbert(X_test, y_test, scbert.gene_list, batch_size=batch_size, random_state=random_state, shuffle=False)

    preds, trues, probas = [], [], []
    with torch.no_grad():
        for xb, yb in dataloader:
            if device:
                xb = xb.to(device)
            out = scbert(xb)
            
            # Get probabilities
            probs = torch.softmax(out, dim=1).cpu().numpy()
            probas.append(probs)

            if isinstance(out, torch.Tensor):
                preds.append(out.argmax(dim=1).cpu().numpy())
            else:
                preds.append(out)
            trues.append(yb.numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(trues)
    y_proba = np.concatenate(probas)

    return y_pred, y_proba, y_true


def train_logreg(X_train, y_train, do_grid_search_cv=False, random_state=42, params=None):
    if params:
        return LogisticRegression(max_iter=10000, random_state=random_state, **params).fit(X_train, y_train)

    if X_train.shape[0] > 40 and do_grid_search_cv:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        param_grid = {'C': [ 0.1, 1, 10]}
        grid_search = GridSearchCV(LogisticRegression(max_iter=10000, random_state=random_state), param_grid, cv=cv)
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_
    return LogisticRegression(max_iter=10000, random_state=random_state).fit(X_train, y_train)


def train_xgboost(X_train, y_train, params=None, do_grid_search_cv=False, random_state=42):
    default_params = {
        'objective': 'multi:softmax',
        'eval_metric': 'mlogloss',
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': random_state
    }
    
    if params:
        default_params.update(params)
    
    default_params['num_class'] = len(np.unique(y_train))
    
    if params is None and X_train.shape[0] > 40 and do_grid_search_cv:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 6]
        }
        
        base_model_params = default_params.copy()
        for key in param_grid.keys():
            base_model_params.pop(key, None)

        model = XGBClassifier(**base_model_params)
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_
    
    model = XGBClassifier(**default_params)
    model.fit(X_train, y_train)
    return model


def train_mlp_classifier(X_train, y_train, batch_size = 16, device="cuda", lr=1e-3, epochs=20, do_grid_search_cv=False, random_state=42, params=None):
    input_dim = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    
    best_params = None
    if params:
        final_lr = params.get('lr', lr)
        final_dropout = params.get('dropout', 0.2)
        final_depth = params.get('depth', 2)
        final_hidden_dim = params.get('hidden_dim', 256)
        final_act_fn = params.get('act_fn', 'relu')
    else:
        final_lr = lr
        final_dropout = 0.2
        final_depth = 2
        final_hidden_dim = 256
        final_act_fn = 'relu'

        if X_train.shape[0] > 40 and do_grid_search_cv:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
            
            param_grid = {
                'lr': [1e-4, 1e-3], 
                'dropout': [0.2, 0.5],
                'depth' : [1, 3, 5],
                'hidden_dim' : [128, 256],
                'act_fn' : ['relu', 'sigmoid']
                } # mirar la profundidad del mlp, la # de capas, probar la sigmoidea como variante también
            best_params = {}
            best_score = -1

            for lr_candidate in param_grid['lr']:
                for dropout_candidate in param_grid['dropout']:
                    for depth_candidate in param_grid['depth']:
                        for hidden_dim_candidate in param_grid['hidden_dim']:
                            for act_fn_candidate in param_grid['act_fn']:
                                scores = []
                                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
                                for train_idx, val_idx in skf.split(X_train, y_train):
                                    if hasattr(X_train, 'iloc'):
                                        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
                                        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
                                    else:
                                        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
                                        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

                                    model = MLPClassifier(input_dim=input_dim, 
                                                            n_classes=n_classes, 
                                                            dropout=dropout_candidate, 
                                                            depth=depth_candidate, 
                                                            hidden_dim=hidden_dim_candidate, 
                                                            act_fn=act_fn_candidate).to(device)
                                    optimizer = torch.optim.Adam(model.parameters(), lr=lr_candidate)
                                    criterion = nn.CrossEntropyLoss()

                                    X_train_fold_values = X_train_fold.values if hasattr(X_train_fold, 'values') else X_train_fold
                                    train_ds_fold = TensorDataset(torch.tensor(X_train_fold_values, dtype=torch.float32), torch.tensor(y_train_fold, dtype=torch.long))
                                    train_loader_fold = DataLoader(train_ds_fold, batch_size=batch_size, shuffle=True)

                                    model.train()
                                    for _ in range(epochs):
                                        for xb, yb in train_loader_fold:
                                            xb, yb = xb.to(device), yb.to(device)
                                            optimizer.zero_grad()
                                            loss = criterion(model(xb), yb)
                                            loss.backward()
                                            optimizer.step()
                                    
                                    model.eval()
                                    with torch.no_grad():
                                        preds = model.predict(X_val_fold)
                                        accuracy = np.mean(preds == y_val_fold)
                                        scores.append(accuracy)
                                
                                avg_score = np.mean(scores)
                                if avg_score > best_score:
                                    best_score = avg_score
                                    best_params = {
                                        'lr': lr_candidate, 
                                        'dropout': dropout_candidate,
                                        'depth': depth_candidate,
                                        'hidden_dim': hidden_dim_candidate,
                                        'act_fn': act_fn_candidate
                                    }
        
            if best_params:
                final_lr = best_params.get('lr', lr)
                final_dropout = best_params.get('dropout', 0.2)
                final_depth = best_params.get('depth', 2)
                final_hidden_dim = best_params.get('hidden_dim', 256)
                final_act_fn = best_params.get('act_fn', 'relu')

    model = MLPClassifier(input_dim=input_dim, n_classes=n_classes, dropout=final_dropout, depth=final_depth, hidden_dim=final_hidden_dim, act_fn=final_act_fn).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=final_lr)
    criterion = nn.CrossEntropyLoss()

    X_train_values = X_train.values if hasattr(X_train, 'values') else X_train
    train_ds = TensorDataset(torch.tensor(X_train_values, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    if best_params:
        model.best_params_ = best_params
    return model


class Ensemble:
    def __init__(self, classifiers):
        self.classifiers = classifiers

    def __call__(self, X):
        return self.predict(X)

    def predict(self, X):
        dimensions = X.shape[2]
        if dimensions != len(self.classifiers):
            raise ValueError("Number of classifiers must match the number of dimensions in X")
        
        # predictions from each classifier
        preds = [clf[1].predict(X[:, :, dim]) for dim, clf in enumerate(self.classifiers)]

        # majority vote for each sample, majority vote per dimension
        preds = np.array(preds)  # Shape: (n_classifiers, n_samples)
        majority_votes = []
        for i in range(preds.shape[1]):
            counts = np.bincount(preds[:, i])
            majority_votes.append(np.argmax(counts))
        return np.array(majority_votes)
    
    def predict_proba(self, X):
        # average the probabilities from each classifier, soft voting
        probas = [clf[1].predict_proba(X[:, :, dim]) for dim, clf in enumerate(self.classifiers)]
        return np.mean(probas, axis=0)



def train_ensemble(ensemble_model_fn, X_train, y_train, do_grid_search_cv=False, random_state=42, params=None):
    """
    train an ensemble of models on each embedding dimension, X_train.shape[2].
    If grid search is enabled, it's performed only on the first dimension to find
    the best hyperparameters, which are then reused for all other dimensions to save time.
    """ 
    dimensions = X_train.shape[2]
    classifiers = []
    best_params = params

    # # if grid search is enabled, run it once on the first dimension to get the best parameters
    # if do_grid_search_cv and dimensions > 0:
    #     first_dim_data = X_train[:, :, 0]
    #     model_for_params = ensemble_model_fn(first_dim_data, y_train, do_grid_search_cv=True, random_state=random_state)

    #     if hasattr(model_for_params, 'best_params_'): 
    #         best_params = model_for_params.best_params_
    #     elif hasattr(model_for_params, 'get_params'):  
    #         params = model_for_params.get_params()
    #         if ensemble_model_fn.__name__ == 'train_logreg':
    #             best_params = {'C': params['C']}
    #         elif ensemble_model_fn.__name__ == 'train_xgboost':
    #             param_grid_keys = ['n_estimators', 'learning_rate', 'max_depth']
    #             best_params = {k: params[k] for k in param_grid_keys if k in params}
        
    for dim in tqdm(range(dimensions), total=dimensions, desc=f"Training ensemble classifiers for: {ensemble_model_fn.__name__}"):
        dimension_data = X_train[:, :, dim]
        # use best_params if found, and disable grid search for the actual training loop
        model = ensemble_model_fn(dimension_data, y_train, do_grid_search_cv=do_grid_search_cv, random_state=random_state, params=best_params)
        classifiers.append((f'{ensemble_model_fn.__name__}_dim_{dim}', model))

    # as the models of the ensemble are already pretrained, we don't use the fit method of the ensemble
    ensemble = Ensemble(classifiers=classifiers) #majority voting

    return ensemble


def train_model_with_dimensionality_reduction(model_fn, X_train, y_train, target_dim=50, do_grid_search_cv=False, random_state=42, batch_size=500):
    """
    Train a model after reducing the dimensionality of the input data using PCA.
    Loads data in batches to avoid OOM issues.
    Fits PCA on a manageable stratified subset of the data due to memory constraints.
    """
    n_samples, n_genes, n_dims = X_train.shape

    # X_train_reshaped = X_train.reshape(-1, n_dims)

    # pca = PCA(n_components=target_dim)
    # X_train_pca_reshaped = pca.fit_transform(X_train_reshaped)

    # X_train_final = X_train_pca_reshaped.reshape(n_samples, -1)

    # model = model_fn(X_train_final, y_train, do_grid_search_cv=do_grid_search_cv, random_state=random_state)

    # return model, pca
    #.reshape copies the mmap to RAM, so we process in batches below:

    max_pca_samples = min(10000, n_samples)
    if max_pca_samples < n_samples:
        sample_indices = train_test_split(np.arange(n_samples), 
                                            train_size=max_pca_samples,
                                            stratify=y_train, 
                                            random_state=random_state)[0]
    else:
        sample_indices = np.arange(n_samples) 
    #np.random.choice(n_samples, max_pca_samples, replace=False)
    
    # Load only the sample for PCA fitting
    X_sample = np.empty((max_pca_samples * n_genes, n_dims), dtype=np.float32)
    for i, idx in enumerate(sample_indices):
        X_sample[i * n_genes : (i + 1) * n_genes] = X_train[idx]
    
    pca = PCA(n_components=target_dim)
    pca.fit(X_sample)
    del X_sample
    
    # Second pass: transform all data in batches
    X_train_final = np.empty((n_samples, n_genes * target_dim), dtype=np.float32)
    
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch = np.array(X_train[start:end])  # Load batch: (batch_size, n_genes, n_dims)
        batch_reshaped = batch.reshape(-1, n_dims)  # (batch_size * n_genes, n_dims)
        batch_pca = pca.transform(batch_reshaped)  # (batch_size * n_genes, target_dim)
        X_train_final[start:end] = batch_pca.reshape(end - start, -1)
        del batch, batch_reshaped, batch_pca

    model = model_fn(X_train_final, y_train, do_grid_search_cv=do_grid_search_cv, random_state=random_state)
    del X_train_final

    return model, pca
    

class ModelBattery:
    def __init__(self, inputs, labels, mode, do_grid_search_cv=False, random_state=42, params_dict=None):
        if params_dict is None:
            params_dict = {}

        if mode == 'per_gene':
            models = {
                'ensemble_logreg': train_ensemble(train_logreg, inputs, labels, do_grid_search_cv=do_grid_search_cv, random_state=random_state, params=params_dict.get('logreg')),
                'ensemble_xgboost': train_ensemble(train_xgboost, inputs, labels, do_grid_search_cv=do_grid_search_cv, random_state=random_state, params=params_dict.get('xgboost')),
                'ensemble_mlp': train_ensemble(train_mlp_classifier, inputs, labels, do_grid_search_cv=do_grid_search_cv, random_state=random_state, params=params_dict.get('mlp')),
                'PCA_1_logreg': train_model_with_dimensionality_reduction(train_logreg, inputs, labels, target_dim=1, do_grid_search_cv=do_grid_search_cv, random_state=random_state),
                'PCA_2_logreg': train_model_with_dimensionality_reduction(train_logreg, inputs, labels, target_dim=2, do_grid_search_cv=do_grid_search_cv, random_state=random_state),
                'PCA_3_logreg': train_model_with_dimensionality_reduction(train_logreg, inputs, labels, target_dim=3, do_grid_search_cv=do_grid_search_cv, random_state=random_state),
                'PCA_1_xgboost': train_model_with_dimensionality_reduction(train_xgboost, inputs, labels, target_dim=1, do_grid_search_cv=do_grid_search_cv, random_state=random_state),
                'PCA_2_xgboost': train_model_with_dimensionality_reduction(train_xgboost, inputs, labels, target_dim=2, do_grid_search_cv=do_grid_search_cv, random_state=random_state),
                'PCA_3_xgboost': train_model_with_dimensionality_reduction(train_xgboost, inputs, labels, target_dim=3, do_grid_search_cv=do_grid_search_cv, random_state=random_state),
                'PCA_1_mlp': train_model_with_dimensionality_reduction(train_mlp_classifier, inputs, labels, target_dim=1, do_grid_search_cv=do_grid_search_cv, random_state=random_state),
                'PCA_2_mlp': train_model_with_dimensionality_reduction(train_mlp_classifier, inputs, labels, target_dim=2, do_grid_search_cv=do_grid_search_cv, random_state=random_state),
                'PCA_3_mlp': train_model_with_dimensionality_reduction(train_mlp_classifier, inputs, labels, target_dim=3, do_grid_search_cv=do_grid_search_cv, random_state=random_state),
            }
        if mode == 'cls':
            models = {
                'logreg': train_logreg(inputs, labels, do_grid_search_cv=do_grid_search_cv, random_state=random_state),
                'xgboost': train_xgboost(inputs, labels, do_grid_search_cv=do_grid_search_cv, random_state=random_state),
                'mlp': train_mlp_classifier(inputs, labels, do_grid_search_cv=do_grid_search_cv, random_state=random_state),
            }
        if mode == 'raw':
            models = {
                'logreg': train_logreg(inputs, labels, do_grid_search_cv=do_grid_search_cv, random_state=random_state),
                'xgboost': train_xgboost(inputs, labels, do_grid_search_cv=do_grid_search_cv, random_state=random_state),
                'mlp': train_mlp_classifier(inputs, labels, do_grid_search_cv=do_grid_search_cv, random_state=random_state),
            }

        self.models = models
        self.mode = mode
        self.do_grid_search_cv = do_grid_search_cv
        self.random_state = random_state
        #self.inputs = inputs
        #self.labels = labels

    def __call__(self, X):
        return self.predict(X)

    def predict(self, X, batch_size=500):
        # predictions from each model
        preds = {}
        for model_name, model in self.models.items():
            if 'PCA' in model_name:
                pca_model, pca = model
                n_samples, n_genes, n_dims = X.shape
                target_dim = pca.n_components_
                
                # process in batches to keep memory usage low: don't copy mmaps to RAM
                X_final = np.empty((n_samples, n_genes * target_dim), dtype=np.float32)
                for start in range(0, n_samples, batch_size):
                    end = min(start + batch_size, n_samples)
                    batch = np.array(X[start:end])
                    batch_reshaped = batch.reshape(-1, n_dims)
                    batch_pca = pca.transform(batch_reshaped)
                    X_final[start:end] = batch_pca.reshape(end - start, -1)
                    del batch, batch_reshaped, batch_pca
                
                preds[model_name] = [int(result) for result in pca_model.predict(X_final)]
                del X_final
            else:
                preds[model_name] = [int(result) for result in model.predict(X)]

        self.preds = preds
        return preds
    
    def predict_proba(self, X, batch_size=500):
        # average the probabilities from each model, soft voting
        probas = {}
        for model_name, model in self.models.items():
            if 'PCA' in model_name:
                pca_model, pca = model
                n_samples, n_genes, n_dims = X.shape
                target_dim = pca.n_components_
                
                # Process in batches like predict()
                X_final = np.empty((n_samples, n_genes * target_dim), dtype=np.float32)
                for start in range(0, n_samples, batch_size):
                    end = min(start + batch_size, n_samples)
                    batch = np.array(X[start:end])
                    batch_reshaped = batch.reshape(-1, n_dims)
                    batch_pca = pca.transform(batch_reshaped)
                    X_final[start:end] = batch_pca.reshape(end - start, -1)
                    del batch, batch_reshaped, batch_pca
                
                probas[model_name] = pca_model.predict_proba(X_final)
                del X_final
            else:
                probas[model_name] = model.predict_proba(X)
        return probas
    
    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)