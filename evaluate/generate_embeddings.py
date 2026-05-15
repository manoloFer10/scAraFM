import os, sys
import numpy as np
import pandas as pd
import scanpy as sc
import glob
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from model.performer.performer import PerformerLM
import tempfile

sys.path.append("..")
torch.multiprocessing.set_sharing_strategy('file_system')

class SCDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, gene_list, data_prefix, n_classes, mask_token=6, tissue='Root'):

        if os.path.isdir(data_prefix):
            h5ad_files = sorted([f for f in os.listdir(data_prefix) if f.endswith('.h5ad')])
            file_paths = [os.path.join(data_prefix, f) for f in h5ad_files]
        else:
            file_paths = [data_path]

        # MANU: this is not necessary right? we already selected genes in preprocess_dataset
        # Find common genes
        # gene_sets = []
        # for path in file_paths:
        #     adata = sc.read_h5ad(path, backed='r')
        #     gene_sets.append(set(adata.var_names))
        #     adata.file.close()
        # common_genes = set.intersection(*gene_sets)

        first_adata = sc.read_h5ad(file_paths[0], backed='r')
        # not really, genes were already subsetted in preprocess_dataset
        # selected_genes = [g for g in first_adata.var_names if g in common_genes][:4000]
        first_adata.file.close()

        adata = sc.read_h5ad(data_path)

        # Perform cell type prediction on the original data first
        markers_df = pd.read_csv("data/arabidopsis_thaliana.marker_fd.csv.gz", compression="gzip")
        markers_df = markers_df.query("tissue == @tissue")
        markers_df = markers_df[markers_df['gene'].isin(adata.var_names)]
        
        celltype_scores = {}
        for cell_type in markers_df['clusterName'].unique():        
            sub = markers_df[markers_df['clusterName'] == cell_type]
            if sub.empty: # if there are no markers for this cell type, skip it.
                continue
            expr_matrix = adata[:, sub['gene']].X.toarray()
            weights = sub['avg_log2FC'].values
            weighted_score = expr_matrix @ weights
            celltype_scores[cell_type] = weighted_score
        
        score_df = pd.DataFrame(celltype_scores, index=adata.obs_names)
        adata.obs["predicted_celltype"] = score_df.idxmax(axis=1)
        
        # Now, prepare the data for the model: binning and then reindexing
        # 1. Bin the data
        data = np.clip(adata.X.toarray(), a_min=0, a_max=n_classes - 2).astype(np.int32)
        
        # 2. Create a DataFrame to handle gene lists
        data_df = pd.DataFrame(data, index=adata.obs_names, columns=adata.var_names)
        
        # 3. Reindex to match the model's gene list, filling missing genes with the mask token
        data_df = data_df.reindex(columns=gene_list, fill_value=mask_token)
        
        self.gene_names = data_df.columns.tolist()
        data = data_df.values
        
        # 4. Add <EOS> token
        eos_col = np.full((data.shape[0], 1), n_classes - 1, dtype=np.uint32)
        data = np.concatenate([data, eos_col], axis=1)

        self.data = torch.Tensor(data)
        
        self.cell_type_labels = pd.get_dummies(adata.obs["predicted_celltype"]).astype(float)
        self.label_names = self.cell_type_labels.columns
        self.cell_type_labels = torch.Tensor(self.cell_type_labels.values)

    def __getitem__(self, index):
        
        return self.cell_type_labels[index], self.data[index].long()
    
    def __len__(self):
        return len(self.data)

class SCDatasetFromAdata(torch.utils.data.Dataset):
    """A PyTorch Dataset class that takes an AnnData object."""
    def __init__(self, adata, gene_list, n_classes, mask_token=6, tissue='Root'):
        """
        Initializes the dataset.

        Args:
            adata (anndata.AnnData): The AnnData object containing the single-cell data.
            gene_list (list): The list of genes the model was trained on.
            n_classes (int): The number of bins for expression clipping.
            mask_token (int, optional): The token to use for missing genes. Defaults to 6.
            tissue (str, optional): Tissue name used to filter cell-type markers ('Root' or 'Leaf'). Defaults to 'Root'.
        """
        # Perform cell type prediction on the original data first
        markers_df = pd.read_csv("data/arabidopsis_thaliana.marker_fd.csv.gz", compression="gzip")
        markers_df = markers_df.query("tissue == @tissue")
        markers_df = markers_df[markers_df['gene'].isin(adata.var_names)]
        
        celltype_scores = {}
        for cell_type in markers_df['clusterName'].unique():        
            sub = markers_df[markers_df['clusterName'] == cell_type]
            if sub.empty: # if there are no markers for this cell type, skip it.
                continue
            expr_matrix = adata[:, sub['gene']].X.toarray()
            weights = sub['avg_log2FC'].values
            weighted_score = expr_matrix @ weights
            celltype_scores[cell_type] = weighted_score
        
        score_df = pd.DataFrame(celltype_scores, index=adata.obs_names)
        adata.obs["predicted_celltype"] = score_df.idxmax(axis=1)

        # Now, prepare the data for the model: binning and then reindexing
        # 1. Bin the data
        data = np.clip(adata.X.toarray(), a_min=0, a_max=n_classes - 2).astype(np.int8)
        
        # 2. Create a DataFrame to handle gene lists
        data_df = pd.DataFrame(data, index=adata.obs_names, columns=adata.var_names)
        
        # 3. Reindex to match the model's gene list, filling missing genes with the mask token
        data_df = data_df.reindex(columns=gene_list, fill_value=mask_token)
        
        self.gene_names = data_df.columns.tolist()
        data = data_df.values
        
        # 4. Add <EOS> token
        eos_col = np.full((data.shape[0], 1), n_classes - 1, dtype=np.uint8)
        data = np.concatenate([data, eos_col], axis=1)

        self.data = torch.Tensor(data)
        
        self.cell_type_labels = pd.get_dummies(adata.obs["predicted_celltype"]).astype(float)
        self.label_names = self.cell_type_labels.columns
        self.cell_type_labels = torch.Tensor(self.cell_type_labels.values)

    def __getitem__(self, index):
        
        return self.cell_type_labels[index], self.data[index].long()
    
    def __len__(self):
        return len(self.data)

class SCDatasetFromDataFrame(torch.utils.data.Dataset):
    """A PyTorch Dataset class that takes a prebuilt expression DataFrame."""
    def __init__(self, expression_df: pd.DataFrame, labels: pd.Series, gene_list, n_classes, mask_token=6, tissue='Root'):
        markers_df = pd.read_csv("data/arabidopsis_thaliana.marker_fd.csv.gz", compression="gzip")
        markers_df = markers_df.query("tissue == @tissue")
        markers_df = markers_df[markers_df['gene'].isin(expression_df.columns)]

        celltype_scores = {}
        for cell_type in markers_df['clusterName'].unique():
            sub = markers_df[markers_df['clusterName'] == cell_type]
            if sub.empty:
                continue
            expr_matrix = expression_df.loc[:, sub['gene']].values
            weights = sub['avg_log2FC'].values
            weighted_score = expr_matrix @ weights
            celltype_scores[cell_type] = weighted_score

        score_df = pd.DataFrame(celltype_scores, index=expression_df.index)
        predicted_celltype = score_df.idxmax(axis=1)

        #gene_list = gene_list[1:] #RETRAIN WITH THIS IN MIND: read_gene_list in performer.py incorrectly loads the header as a gene.

        data = np.clip(expression_df.values, a_min=0, a_max=n_classes - 2).astype(np.int32)
        data_df = pd.DataFrame(data, index=expression_df.index, columns=expression_df.columns)
        data_df = data_df.reindex(columns=gene_list, fill_value=mask_token)

        self.gene_names = data_df.columns.tolist()
        data = data_df.values
        eos_col = np.full((data.shape[0], 1), n_classes - 1, dtype=np.uint32)
        data = np.concatenate([data, eos_col], axis=1)

        self.data = torch.Tensor(data)
        self.cell_type_labels = pd.get_dummies(predicted_celltype).astype(float)
        self.label_names = self.cell_type_labels.columns
        self.cell_type_labels = torch.Tensor(self.cell_type_labels.values)
        label_array = labels.values if hasattr(labels, "values") else np.asarray(labels)
        self.labels = torch.tensor(label_array, dtype=torch.long)

    def __getitem__(self, index):
        return self.data[index].long(), self.labels[index]

    def __len__(self):
        return len(self.data)


def get_epoch_from_ckpt(ckpt_path):
    return int(ckpt_path.split("_")[-1].split(".")[0])


def direct_embed(model, x): #, inference_indices_in_training):
    output    = model(x, return_encodings=True)[:,-1,:] #embedding from the [EOS] token
    embedding = output.to("cpu")
    return embedding


def per_gene_embed(model, x): #, inference_indices_in_training):
    full_output = model(x, return_encodings=True)[:,:-1,:]
    embedding = full_output.to("cpu")
    return embedding


def both_embed(model, x):
    """Single forward pass; returns (cls_embedding, per_gene_embedding)."""
    output = model(x, return_encodings=True)
    cls_emb = output[:, -1, :].to("cpu")
    per_gene_emb = output[:, :-1, :].to("cpu")
    return cls_emb, per_gene_emb


def get_embedding_fn(mode):
    if mode == 'direct':
        return direct_embed
    elif mode == 'per_gene':
        return per_gene_embed
    else:
        raise ValueError(f"Unknown embedding mode: {mode}")
    
def generate_embeddings_for_dataset(adata, model, embedding_mode, device, gene_list, n_classes, cache_path=None, tissue='Root'):
    '''Generate embeddings for all cells in the AnnData object using the provided model.'''

    if cache_path and os.path.exists(cache_path):
        print(f"Loading embeddings from cache: {cache_path}")
        return np.lib.format.open_memmap(cache_path, mode='r')

    embedding_function = get_embedding_fn(embedding_mode)
    dataset = SCDatasetFromAdata(adata, gene_list, n_classes, tissue=tissue)
    
    total_samples = len(dataset)
    if total_samples == 0:
        return np.array([])

    def generate(output_file):
        dataloader = DataLoader(dataset, batch_size=32, num_workers=2, shuffle=False)
        
        # Determine embedding dimensions from the first batch
        _, x = next(iter(dataloader))
        x = x.to(device)
        with torch.no_grad():
            sample_embedding = embedding_function(model, x)
        embedding_shape = sample_embedding.shape[1:]
        
        # Create memory-mapped array
        embeddings_mmap = np.lib.format.open_memmap(
            output_file, 
            mode='w+', 
            dtype=np.float16,
            shape=(total_samples, *embedding_shape)
        )
        
        # Re-initialize dataloader to iterate from the beginning
        dataloader = DataLoader(dataset, batch_size=32, num_workers=2, shuffle=False)
        
        start_idx = 0
        for _, x in tqdm(dataloader):
            x = x.to(device)
            
            with torch.no_grad():
                embedding = embedding_function(model, x)
            
            batch_size = embedding.shape[0]
            end_idx = start_idx + batch_size
            embeddings_mmap[start_idx:end_idx] = embedding.cpu().numpy()
            start_idx = end_idx
            
        embeddings_mmap.flush()
        return embeddings_mmap

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        print(f"Generating embeddings and saving to cache: {cache_path}")
        return generate(cache_path)
    else:
        # Create a temp file in the cache directory (to ensure disk space)
        # We use mkstemp to get a file, then close the FD because open_memmap needs a path.
        os.makedirs('cache', exist_ok=True)
        fd, temp_path = tempfile.mkstemp(dir='cache', suffix='.npy')
        os.close(fd)
        
        print(f"Generating temporary embeddings backed by {temp_path} (will be deleted from filesystem immediately)")
        
        try:
            embeddings_mmap = generate(temp_path)
            # On Linux, we can unlink the file and keep the mmap open.
            # The space will be reclaimed when the process exits or the mmap is closed.
            os.remove(temp_path)
            return embeddings_mmap
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e


def generate_both_embeddings_for_dataset(adata, model, device, gene_list, n_classes, tissue='Root'):
    """Run a single forward pass per batch and persist both CLS and per-gene embeddings.

    Returns (cls_mmap, per_gene_mmap) as numpy memmaps (float16). Both arrays are backed
    by temp files in ./cache that are unlinked immediately after creation; space is
    reclaimed when the mmaps are closed or the process exits.
    """
    dataset = SCDatasetFromAdata(adata, gene_list, n_classes, tissue=tissue)
    total_samples = len(dataset)
    if total_samples == 0:
        return np.array([]), np.array([])

    dataloader = DataLoader(dataset, batch_size=32, num_workers=2, shuffle=False)
    _, x = next(iter(dataloader))
    x = x.to(device)
    with torch.no_grad():
        sample_cls, sample_pg = both_embed(model, x)
    cls_shape = sample_cls.shape[1:]
    pg_shape = sample_pg.shape[1:]

    os.makedirs('cache', exist_ok=True)
    fd_cls, cls_path = tempfile.mkstemp(dir='cache', suffix='.npy')
    os.close(fd_cls)
    fd_pg, pg_path = tempfile.mkstemp(dir='cache', suffix='.npy')
    os.close(fd_pg)

    print(f"Generating CLS and per-gene embeddings in a single pass; temp files {cls_path} and {pg_path}")
    try:
        cls_mmap = np.lib.format.open_memmap(
            cls_path, mode='w+', dtype=np.float16, shape=(total_samples, *cls_shape)
        )
        pg_mmap = np.lib.format.open_memmap(
            pg_path, mode='w+', dtype=np.float16, shape=(total_samples, *pg_shape)
        )

        dataloader = DataLoader(dataset, batch_size=32, num_workers=2, shuffle=False)
        start_idx = 0
        for _, x in tqdm(dataloader):
            x = x.to(device)
            with torch.no_grad():
                cls_emb, pg_emb = both_embed(model, x)
            bs = cls_emb.shape[0]
            end_idx = start_idx + bs
            cls_mmap[start_idx:end_idx] = cls_emb.cpu().numpy()
            pg_mmap[start_idx:end_idx] = pg_emb.cpu().numpy()
            start_idx = end_idx

        cls_mmap.flush()
        pg_mmap.flush()

        os.remove(cls_path)
        os.remove(pg_path)
        return cls_mmap, pg_mmap
    except Exception:
        for p in (cls_path, pg_path):
            if os.path.exists(p):
                os.remove(p)
        raise


def main(ckpt_path, gene_csv, emb_mode, data_dir, n_classes, n_genes, tissue='Root'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert os.path.exists(ckpt_path), f"--ckpt_path does not exist: {ckpt_path}"

    if os.path.isdir(data_dir):
        datapaths = glob.glob(f"{data_dir}/*.h5ad")
    else:
        datapaths = [data_dir]

    gene_list = list(pd.read_csv(gene_csv)['gene_name'])
    if n_genes is None:
        if not datapaths:
            raise FileNotFoundError(f"No .h5ad files found in {data_dir}")
        print("Inferring n_genes from gene_csv...")
        n_genes = len(gene_list)
        print(f"Inferred n_genes: {n_genes}")

    model = PerformerLM.from_checkpoint(ckpt_path, n_genes, gene_csv=gene_csv, device=device)
    model.eval()
    print(f"loaded model from {ckpt_path=}")

    os.makedirs("cache",exist_ok=True)

    print(f"{datapaths=}")

    embedding_function = get_embedding_fn(emb_mode)
    ckpt_tag = os.path.splitext(os.path.basename(ckpt_path))[0]

    for data_path in tqdm(datapaths):
        DATASET_ID = os.path.basename(data_path).split("_")[0]
        OUTPUT_EMBEDDING_FILE = f"cache/embeddings__{emb_mode}__{ckpt_tag}__{DATASET_ID}__{tissue}.npy"

        print(f"Processing {DATASET_ID=}...")

        dataset = SCDataset(data_path=data_path, gene_list=gene_list, data_prefix=data_dir, n_classes=n_classes, tissue=tissue)
        model.inference_gene_list = dataset.gene_names
        # training_gene_indices = {gene: i for i, gene in enumerate(model.gene_list)}
        # inference_indices_in_training = [training_gene_indices[gene] for gene in model.inference_gene_list]  
        dataloader = DataLoader(dataset, batch_size=32, num_workers=2, shuffle=False)
        
        # First, determine the total number of samples and embedding dimensions
        total_samples = len(dataset)
        
        # Get embedding dimensions from the first batch
        y, x = next(iter(dataloader))
        x = x.to(device)
        with torch.no_grad():
            sample_embedding = embedding_function(model, x)#, inference_indices_in_training)
        embedding_shape = sample_embedding.shape[1:]  # Get dimensions excluding batch
        
        # Create memory-mapped arrays for embeddings 
        embeddings_mmap = np.lib.format.open_memmap(
            OUTPUT_EMBEDDING_FILE, 
            mode='w+', 
            dtype=np.float16,
            shape=(total_samples, *embedding_shape)
        )
        
        # Reset dataloader
        dataloader = DataLoader(dataset, batch_size=32, num_workers=2, shuffle=False)
        
        # Process batches and write directly to mmap files
        start_idx = 0
        for batch in tqdm(dataloader):
            y, x = batch
            x = x.to(device)
            
            batch_size = x.shape[0]
            end_idx = start_idx + batch_size
            
            # Process embeddings
            with torch.no_grad():
                embedding = embedding_function(model, x)#, inference_indices_in_training)
            
            # Write batch to mmap files
            embeddings_mmap[start_idx:end_idx] = embedding.numpy()
            
            # Update index for next batch
            start_idx = end_idx
            
        # Ensure data is written to disk
        embeddings_mmap.flush()

    return embeddings_mmap # intended to return something when used for a single data file. Else it will return only the last one.

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate embeddings from a trained scAraFM model.")
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to the pretrained PerformerLM checkpoint (.pth).')
    parser.add_argument('--n_classes', type=int, default=7, help='Number of classes for clipping expression values.')
    parser.add_argument('--gene_csv', type=str, required=True, help='Path to the CSV file containing the gene list of the model.')
    parser.add_argument('--emb-mode', type=str, default='direct', choices=['direct', 'per_gene'], help='Embedding mode: [EOS] token (direct) or per-gene encodings.')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to a .h5ad file or a directory containing .h5ad files.')
    parser.add_argument('--n_genes', type=int, default=None, help='Number of genes. If not provided, inferred from gene_csv.')
    parser.add_argument('--tissue', choices=['Root', 'Leaf'], default='Root', help='Tissue used to filter cell-type marker genes.')

    args = parser.parse_args()
    main(ckpt_path=args.ckpt_path,
         gene_csv=args.gene_csv,
         emb_mode=args.emb_mode,
         data_dir=args.data_dir,
         n_classes=args.n_classes,
         n_genes=args.n_genes,
         tissue=args.tissue)

