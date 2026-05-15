import scanpy as sc
import torch
from torch.utils.data import Dataset
import os
import numpy as np

N_CLASSES = 7

class MultiScRNADataset(Dataset):
    def __init__(self, folder_path, binning_style, bin_edges=None, top_n_genes=None):
        super().__init__()

        self.tensor_list = []
        self.lengths = []
        self.N_CLASSES = N_CLASSES
        self.binning_style = binning_style
        self.bin_edges = bin_edges

        h5ad_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.h5ad')])
        self.file_paths = [os.path.join(folder_path, f) for f in h5ad_files]

        # Find common genes
        gene_sets = []
        for path in self.file_paths:
            adata = sc.read_h5ad(path, backed='r')
            gene_sets.append(set(adata.var_names))
            adata.file.close()
        common_genes = set.intersection(*gene_sets)

        # Use order from first file
        first_adata = sc.read_h5ad(self.file_paths[0], backed='r')
        self.selected_genes = [g for g in first_adata.var_names if g in common_genes][:top_n_genes]
        first_adata.file.close()

        all_X = []
        for path in self.file_paths:
            adata = sc.read_h5ad(path)
            adata = adata[:, self.selected_genes]

            # Extract matrix and convert
            X_file = adata.X
            if hasattr(X_file, "toarray"):
                X_file = X_file.toarray()
            elif hasattr(X_file, "A"):
                X_file = X_file.A
            else:
                X_file = np.asarray(X_file)

            all_X.append(X_file)
            
            del adata

        X = np.concatenate(all_X, axis=0)
        del all_X

        # bin at dataset level    
        X = self._binning_fn()(X)

        eos_column = np.full((X.shape[0], 1), N_CLASSES - 1, dtype=np.uint32) #Manu: add eos token as a particular id

        X = np.concatenate([X, eos_column], axis=1)

        tensor_data = torch.from_numpy(X).long()  # dtype=torch.uint8
        self.tensor_list.append(tensor_data)
        self.lengths.append(tensor_data.shape[0])

        del X  # explicit cleanup

        # Calculate cumulative lengths across files (needed for indexing)
        self.cumulative_lengths = np.cumsum([0] + self.lengths)

    def _binning_fn(self):

        def default_binning_fn(X):
            return np.clip(X, a_min=0, a_max=self.N_CLASSES-2).astype(np.uint32)  # values 0-5 Supposes adata contains normalized values
        
        def quantile_binning_fn(X):
            n_bins = self.N_CLASSES - 2  # 5 bins for gene expression levels
            X_flat = X.flatten()
            X_flat = X_flat[X_flat > 0]  

            
            # quantile edges for n_bins
            if self.bin_edges is None:
                quantiles = np.linspace(0, 100, n_bins + 1)
                bin_edges = np.percentile(X_flat, quantiles)
                bin_edges[0] = 0  
                bin_edges[-1] = np.inf  # maximum captures all values
                self.bin_edges = bin_edges
            
            X = np.digitize(X, self.bin_edges[1:-1]).astype(np.uint32)
            X = np.clip(X, 0, n_bins - 1) 
            
            return X

        if self.binning_style == 'default':
            return default_binning_fn
        elif self.binning_style == 'quantile':
            return quantile_binning_fn
        else:
            raise ValueError(f"Unknown binning style: {self.binning_style}")

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, index):

        file_idx = np.searchsorted(self.cumulative_lengths, index, side='right') - 1 
        local_idx = index - self.cumulative_lengths[file_idx]
        datapoint = self.tensor_list[file_idx][local_idx]
        return datapoint
        