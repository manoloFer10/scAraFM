import os 
import time
import logging
import argparse
import numpy as np
import random

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Subset

from model.multi_file_dataset import MultiScRNADataset as scRNADataset
from model.performer.performer import PerformerLM, filter_gene2vec_weight, read_gene_list

from torch.optim import Adam
from model.masking import data_mask
from pathlib import Path
import mlflow

from tqdm import tqdm


def set_seed(seed):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
torch.set_float32_matmul_precision('high')

set_seed(seed := os.environ.get('SEED', 42))  # Use environment variable if present, or default to 42

EXPERIMENT_NAME = os.environ.get("EXPERIMENT_NAME", "scFoundationModel")

if (os.environ.get("LOG_TO_DAGSHUB", False)):
    import dagshub
    logging.info("Logging to DagsHub...")
    dagshub.init(EXPERIMENT_NAME, "rbonazzola", mlflow=True)
else:
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(EXPERIMENT_NAME)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def save_ckpt(epoch, model, optimizer, scheduler, losses, model_name, ckpt_folder, model_args=None):
    """
    save checkpoint
    """
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)

    ckpt_file = f'{ckpt_folder}/{model_name}_{epoch}.pth'

    torch.save({
        'epoch': epoch,            
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'losses': losses,
        'model_args': model_args
    }, ckpt_file)

    return ckpt_file

class Trainer:
    
    def __init__(self, args, g2v_embeddings=None, binning_style=None):
        logging.info("Initializing Trainer")
        if binning_style is None:
            raise ValueError("binning_style must be provided")
        self.binning_style = binning_style
        logging.info(f"Using binning style: {self.binning_style}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_half_precision = args.half_precision and self.device.type == "cuda"
        self.SEQ_LEN = args.gene_num + 1
        self.LEARNING_RATE = args.lr
        self.MAX_EPOCHS = args.max_epochs
        self.max_samples = args.max_samples
        self.VALIDATE_EVERY = 1
        self.GRADIENT_ACCUMULATION = args.gradient_accumulation or 1
        self.BATCH_SIZE = args.batch_size or 1
        self.TOP_N_GENES = args.top_n_genes
        self.NUM_WORKERS = args.num_workers or 0
        self.USING_COMPILE = args.compile
        self.MASK_PROBABILITY = args.mask_probability
        self.best_val_loss = float("inf")
        self.patience = args.early_stopping_patience or 10
        self.early_stopping_counter = 0
        self.resume_mode = args.resume_mode
        self.resume_run_id = None
        self.start_epoch = 1
        self.resume_ckpt = args.resume_ckpt
        self.gene_list = read_gene_list(args.gene_csv)

        self._load_data(args.data_path)

        self.model_args = {
            "num_tokens": self.N_CLASSES+1,  # +1 for mask token
            "dim": args.embedding_dim,
            "depth": args.depth,
            "max_seq_len": self.SEQ_LEN,
            "heads": args.heads,
            "local_attn_heads": args.local_attn_heads,
            "g2v_embeddings": g2v_embeddings,
            "gene_list": self.gene_list
        }

        self._initialize_model()
        self._initialize_training_components()
    

    def _load_data(self, data_path):
        
        logging.info("Loading data")
        start_time = time.time()

        self.train_dataset = scRNADataset(f"{data_path}/train", self.binning_style)
        self.val_dataset = scRNADataset(f"{data_path}/val", self.binning_style, bin_edges=self.train_dataset.bin_edges) # use the bin edges from training set

        self.N_CLASSES = self.train_dataset.N_CLASSES

        if self.max_samples is not None:
            self.train_dataset = Subset(self.train_dataset, np.random.choice(range(len(self.train_dataset)), self.max_samples, replace=False))
            self.val_dataset   = Subset(self.val_dataset,   np.random.choice(range(len(self.val_dataset)), self.max_samples, replace=False))

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.BATCH_SIZE, num_workers=self.NUM_WORKERS)
        self.val_loader   = DataLoader(self.val_dataset,   batch_size=self.BATCH_SIZE, num_workers=self.NUM_WORKERS)

        logging.info(f"Data loaded in {time.time() - start_time:.2f} seconds")
        logging.info(f"Data loaded in {time.time() - start_time:.2f} seconds.")
        logging.info(f"Training with {len(self.train_dataset)} samples.")
        logging.info(f"Validatig with {len(self.val_dataset)} samples.")
    

    def _initialize_model(self):
        logging.info("Initializing model")
        start_time = time.time()
        self.model = PerformerLM(**self.model_args).to(self.device)
        
        if self.use_half_precision:
            self.model = self.model.half()
        
        if self.USING_COMPILE:
            self.model = torch.compile(self.model)

        if self.resume_ckpt and os.path.exists(self.resume_ckpt):
            logging.info(f"Resuming from checkpoint: {self.resume_ckpt}")
            checkpoint = torch.load(self.resume_ckpt, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            if self.resume_mode == "same":
                self.start_epoch = checkpoint["epoch"] + 1
                self.best_val_loss = checkpoint.get("losses", float("inf"))
            self._ckpt_opt_state = checkpoint

            # Try to infer run_id
            ckpt_path = Path(self.resume_ckpt).resolve()
            if "mlruns" in str(ckpt_path):
                try:
                    run_id = ckpt_path.parents[2].name
                    self.resume_run_id = run_id
                    logging.info(f"Inferred MLflow run_id: {run_id}")
                except Exception as e:
                    logging.warning(f"Could not infer MLflow run_id: {e}")

        n_params = count_parameters(self.model)
        logging.info(f"Model initialized with {n_params:,} parameters in {time.time() - start_time:.2f} seconds")


    def _initialize_training_components(self):
        self.optimizer = Adam(self.model.parameters(), lr=self.LEARNING_RATE)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=1)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.N_CLASSES, reduction='mean')
        self.softmax = nn.Softmax(dim=-1)

        # if resuming, restore optimizer + scheduler
        if hasattr(self, "_ckpt_opt_state"):
            self.optimizer.load_state_dict(self._ckpt_opt_state["optimizer_state_dict"])
            self.scheduler.load_state_dict(self._ckpt_opt_state["scheduler_state_dict"])
    
    def train(self):
        logging.info("Starting training")
        if self.resume_mode == "same" and self.resume_run_id:
            mlflow.start_run(run_id=self.resume_run_id)
            logging.info(f"Continuing MLflow run {self.resume_run_id}")
        else:
            mlflow.start_run()
            logging.info("Starting a new MLflow run")
        mlflow.log_params({
           "max_epochs": self.MAX_EPOCHS, 
           "learning_rate": self.LEARNING_RATE, 
           "batch_size": self.BATCH_SIZE,
           # "n_batches": self.MAX_BATCHES,
           "samples_per_epoch": len(self.train_dataset), #if self.MAX_BATCHES is None else min(self.MAX_BATCHES * self.BATCH_SIZE, len(self.train_dataset)),
           "embedding_dim": self.model_args["dim"],
           "depth": self.model_args["depth"],
           "heads": self.model_args["heads"],
           "local_attn_heads": self.model_args["local_attn_heads"],
           "platform": os.uname().nodename,
           "top_n_genes": self.TOP_N_GENES,
           "num_workers": self.NUM_WORKERS,
           "using_compile": self.USING_COMPILE,
           "n_parameters": count_parameters(self.model),
           "mask_probability": self.MASK_PROBABILITY,
           "seed": seed
        })

        for epoch in tqdm(range(self.start_epoch, self.MAX_EPOCHS + 1), desc='Epochs'):
            
            epoch_start_time = time.time()
            self.model.train()
            
            batch_times, processing_times, mask_times = [], [], []
            
            total_batches = 0
            running_loss = 0.0
            
            # for index, data in tqdm(enumerate(self.train_loader)):
            for index, data in tqdm(enumerate(self.train_loader),desc = "Batches", total=len(self.train_loader)):

                # if self.MAX_BATCHES and index >= self.MAX_BATCHES:
                    # break
            
                batch_start_time = time.time()           
                
                data = data.to(self.device)
            
                # ──────── MASKING ────────
                mask_start_time = time.time()
                data, labels = data_mask(data, mask_prob=self.MASK_PROBABILITY)
                mask_time = time.time() - mask_start_time
                # torch.cuda.synchronize()
            
                # ──────── FORWARD + LOSS ────────
                processing_start_time = time.time()
                
                with torch.amp.autocast('cuda', enabled=self.use_half_precision):                                       
                    logits = self.model(data)
                    # torch.cuda.synchronize()
                loss = self.loss_fn(logits.transpose(1, 2), labels) / self.GRADIENT_ACCUMULATION
            
                # ──────── BACKWARD ────────
                loss.backward()
            
                # ──────── OPTIMIZER STEP ────────
                if index % self.GRADIENT_ACCUMULATION == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
                # ──────── TIME COLLECTION ────────
                processing_time = time.time() - processing_start_time
                batch_time = time.time() - batch_start_time
                
                batch_times.append(batch_time)
                processing_times.append(processing_time)
                mask_times.append(mask_time)
                
                running_loss += loss.item()
                total_batches += 1

            
            epoch_time = time.time() - epoch_start_time
            avg_batch_time = sum(batch_times) / total_batches if total_batches > 0 else 0
            avg_processing_time = sum(processing_times) / total_batches if total_batches > 0 else 0
            data_loading_time = epoch_time - sum(batch_times)
            avg_mask_time = sum(mask_times) / total_batches if total_batches > 0 else 0

            epoch_loss = running_loss / total_batches

            logging.info(f'Epoch {epoch}: Loss = {running_loss / total_batches:.6f}')
            # logging.info(f'Epoch {epoch}: Time = {epoch_time:.2f} sec, Avg batch = {avg_batch_time:.4f} sec')
            # logging.info(f'Processing time: {avg_processing_time:.4f} sec, Data loading time: {data_loading_time:.4f} sec')
            
            mlflow.log_metrics({
                "epoch_time": epoch_time,
                "avg_batch_time": avg_batch_time,
                "avg_processing_time": avg_processing_time,
                "data_loading_time": data_loading_time,
                "data_loading_time_per_sample": data_loading_time / total_batches / self.BATCH_SIZE,
                "time_per_sample": epoch_time / total_batches / self.BATCH_SIZE,
                "avg_mask_time": avg_mask_time,
                "train_loss": epoch_loss}, step=epoch)
            
            self.scheduler.step()
            
            if epoch % self.VALIDATE_EVERY == 0:
                self.validate(epoch)
               
            run_id = mlflow.active_run().info.run_id            
            # ckpt_file = save_ckpt(epoch, self.model, self.optimizer, self.scheduler, epoch_loss, f"performer_model", "./checkpoints/{run_id}")
            # mlflow.log_artifact(ckpt_file, artifact_path="checkpoints")

            if self.early_stopping_counter >= self.patience:
                logging.info(f"Early stopping triggered at epoch {epoch} after {self.early_stopping_counter} epochs with no improvement")
                break

        
        mlflow.end_run()
        logging.info(f"Training complete, saved model to {run_id=}")


    def validate(self, epoch):
        logging.info(f"Validating at epoch {epoch}")
        self.model.eval()
        running_loss_val = 0.0
        correct_num, val_num = 0, 0
        
        with torch.no_grad():
            for data in self.val_loader:
                data = data.to(self.device)
                data, labels = data_mask(data)
                logits = self.model(data)
                loss = self.loss_fn(logits.transpose(1, 2), labels)
                running_loss_val += loss.item()
                final = self.softmax(logits)[..., 1:-1].argmax(dim=-1) + 1
                correct_num += ((labels != self.N_CLASSES) * (final == labels)).sum().item()
                val_num += (labels != self.N_CLASSES).sum().item()
        
        val_loss = running_loss_val / len(self.val_loader)
        val_acc = 100.0 * correct_num / val_num
        logging.info(f'Epoch {epoch}: Validation Loss = {val_loss:.6f}, Accuracy = {val_acc:.4f}%')
        
        if val_loss < self.best_val_loss and abs(val_loss-self.best_val_loss) > 0.001:
            self.best_val_loss = val_loss
            self.early_stopping_counter = 0  # Reset counter
            run_id = mlflow.active_run().info.run_id
            save_ckpt(epoch, self.model, self.optimizer, self.scheduler, val_loss, "best_model", f"./checkpoints/{run_id}", self.model_args)
            mlflow.log_artifact(f"./checkpoints/{run_id}/best_model_{epoch}.pth", artifact_path="checkpoints")
            logging.info(f"Best model updated at epoch {epoch}")
        else:
            self.early_stopping_counter += 1
            logging.info(f"Early stopping counter: {self.early_stopping_counter} / {self.patience}")


        mlflow.log_metrics({"val_loss": val_loss, "val_acc": val_acc}, step=epoch)


def main(args):
    # Pre-load embeddings before initializing Trainer to avoid I/O conflicts
    g2v_embeddings = None
    if args.gene_embedding_path:
        logging.info(f"Loading gene list from {args.gene_csv}")
        gene_list = read_gene_list(args.gene_csv)
        logging.info(f"Loading and filtering gene embeddings from {args.gene_embedding_path}...")
        g2v_embeddings = filter_gene2vec_weight(args.gene_embedding_path, gene_list, args.embedding_dim)
        logging.info("Embeddings loaded.")

    Trainer(args, g2v_embeddings=g2v_embeddings, binning_style=args.binning_style).train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to the preprocessed data')
    parser.add_argument('--gene_csv', type=str, required=True, help='Path to the gene list csv')
    parser.add_argument('--gene_embedding_path', type=str, default='model/gene2vec/gene2vec_80.txt', help='Path to gene2vec embedding file. If not provided, embeddings will be random.')
    parser.add_argument('--gene_num', type=int, required=True, help='Number of genes')
    parser.add_argument('--depth', type=int, default=3, help='Model depth')
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--compile', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gradient_accumulation', type=int, default=1)
    parser.add_argument('--half_precision', action='store_true', default=False, help='Use FP16 for faster training')
    parser.add_argument('--max_samples', type=int, default=None, help='Limit training toand validation  a given number of basamples (the same for both)')
    parser.add_argument('--use-flash-attention', '--use_flash_attention', action='store_true', default=False, help='Limit training to a given number of batches')
    parser.add_argument('--mask-probability', '--mask_probability', default=0.15, help='Masking probability during training', type=float)
    parser.add_argument('--embedding_dim', type=int, default=200, help='Embedding dimension')
    parser.add_argument('--early_stopping_patience', type=int, default=5, help="Patience for early stopping")
    parser.add_argument('--heads', type=int, default=10, help='Number of attention heads')
    parser.add_argument('--local_attn_heads', type=int, default=0, help='Number of local attention heads')
    parser.add_argument('--top_n_genes', type=int, default=None, help="Number of genes to use")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers to use for data loading.")
    parser.add_argument('--resume_ckpt', type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument('--resume_mode', choices=['same','new'], default='new', help="Whether to continue logging in the same MLflow run or start a new one")
    parser.add_argument('--binning_style', type=str, choices=['quantile', 'default'], default='default', help="Binning style for expression values")

    args = parser.parse_args()
    main(args)