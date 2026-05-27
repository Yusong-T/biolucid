"""
Utility functions for data preprocessing and analysis.

This module contains functions for data preprocessing, validation,
and general utility functions used throughout the analysis pipeline.
"""

import logging
from typing import Tuple, Dict
import numpy as np
import pandas as pd
import scipy.sparse
import itertools
import scanpy as sc
from anndata import AnnData

# Set up logging
logger = logging.getLogger(__name__)

def setup_logging(verbose: bool = True):
    """
    Configure logging settings.
    
    Args:
        verbose: Whether to show INFO level logs
    """
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def validate_and_filter_celltypes(adata: AnnData, params: Dict) -> AnnData:
    """
    Validate batch and cell type information and filter cell types.
    Ensures each cell type has sufficient cells in EACH batch.
    
    Args:
        adata: Input data
        params: Analysis parameters
        
    Returns:
        Filtered AnnData with valid cell types
        
    Raises:
        ValueError: If validation fails
    """
    # Check batch information
    n_batches = adata.obs[params['batch_key']].nunique()
    if n_batches < 2:
        raise ValueError(f"Found only {n_batches} batch(es). At least 2 batches are required for analysis")
    
    # Calculate cell counts for each batch-celltype combination
    batch_celltype_counts = pd.crosstab(
        adata.obs[params['batch_key']], 
        adata.obs[params['celltype_key']]
    )
    
    min_cells = params.get('min_cells', 20)
    
    # 1. Find top 10 most frequent cell types globally
    top_celltypes = batch_celltype_counts.sum(axis=0).nlargest(10).index.tolist()
    
    total_samples = len(batch_celltype_counts.index)
    
    best_score = -1
    best_comb = None
    best_samples = None
    
    # 2. Iterate through power set of top 10 cell types (combinations of size >= 2)
    for r in range(2, len(top_celltypes) + 1):
        for comb in itertools.combinations(top_celltypes, r):
            comb_list = list(comb)
            
            # 3. Find samples having >= min_cells for all cell types in this combination
            sub_counts = batch_celltype_counts[comb_list]
            valid_samples_mask = (sub_counts >= min_cells).all(axis=1)
            valid_samples = batch_celltype_counts.index[valid_samples_mask].tolist()
            
            if len(valid_samples) < 2:
                continue
                
            # 4. Calculate maximum selection score
            # Motivation: both score components are bounded between 0 and 1 and monotonous in the desirable properties (# samples kept, # cell types kept). 
            # This score over-penalizes having few cell types (1 cell type -> score = 0, 2 cell types -> score = 0.5)
            score = 0.5 * (1 - 1 / len(comb_list)) + 0.5 * (len(valid_samples) / total_samples)

            if score > best_score:
                best_score = score
                best_comb = comb_list
                best_samples = valid_samples
                
    if best_comb is None:
        raise ValueError(
            f"Could not find any combination of >=2 cell types present in >=2 samples "
            f"with at least {min_cells} cells each."
        )
        
    valid_celltypes = best_comb
    
    # 5. Filter data to keep only selected cell types AND selected samples
    mask = (adata.obs[params['celltype_key']].isin(valid_celltypes)) & \
           (adata.obs[params['batch_key']].isin(best_samples))
    adata_ct_selection = adata[mask].copy()
    
    # Store dropped samples in .uns just in case downstream tools want to access them
    dropped_samples = set(batch_celltype_counts.index) - set(best_samples)
    adata_ct_selection.uns['biolucid_dropped_samples'] = list(dropped_samples)
    
    # Logging
    logger.info(f"Optimization finished: Selected {len(valid_celltypes)} cell types and {len(best_samples)} samples (Score: {best_score:.4f})")
    logger.info(f"Retained {len(adata_ct_selection)} / {len(adata)} cells after filtering")
    logger.info(f"Retained cell types: {valid_celltypes}")
    logger.info(f"Retained samples: {best_samples}")
    
    if dropped_samples:
        logger.warning("--- EARLY REJECTION REPORT ---")
        for ds in dropped_samples:
            logger.warning(f"Sample {ds}: b_score = N/A | Recommendation = Drop (Missing core biological populations)")
        logger.warning("------------------------------")
        
    logger.info("Cell counts per batch for retained cell types:")
    retained_counts = batch_celltype_counts.loc[best_samples, valid_celltypes]
    for batch in retained_counts.index:
        logger.info(f"Batch {batch}:")
        for celltype in valid_celltypes:
            count = retained_counts.loc[batch, celltype]
            logger.info(f"  {celltype}: {count} cells")
    
    return adata_ct_selection

def select_abundant_genes(adata: AnnData, params: Dict) -> Tuple[AnnData, np.ndarray]:
    """
    Select genes with sufficient expression across all cells.
    
    Args:
        adata: Input data after cell type filtering
        params: Analysis parameters
        
    Returns:
        Tuple of (filtered AnnData, abundant gene indices)
        
    Raises:
        ValueError: If insufficient abundant genes found
    """
    # Get raw counts
    counts = get_counts_matrix(adata)
    
    # Calculate mean UMI per cell for each gene
    gene_means = np.array(counts.mean(axis=0)).flatten()
    abundant_genes = np.where(gene_means >= params.get('abundant_gene_threshold', 1))[0]
    
    min_genes = params.get('min_abundant_genes', 300)
    if len(abundant_genes) < min_genes:
        raise ValueError(
            f"Found only {len(abundant_genes)} abundant genes (>= {params['abundant_gene_threshold']} UMI/cell). "
            f"At least {min_genes} genes are required for meaningful analysis"
        )
    
    # Create filtered AnnData
    adata_ct_genes_selection = adata[:, abundant_genes].copy()
    
    logger.info(f"Selected {len(abundant_genes)} abundant genes")
    return adata_ct_genes_selection, abundant_genes

def get_counts_matrix(adata: AnnData) -> np.ndarray:
    """
    Locate and return raw counts matrix.
    
    Args:
        adata: Input data
        
    Returns:
        Raw counts matrix
        
    Raises:
        ValueError: If no counts matrix is found
    """
    def is_counts(X) -> bool:
        """Check if matrix contains count data."""
        if scipy.sparse.issparse(X):
            return (np.issubdtype(X.data.dtype, np.integer) or 
                    np.all(np.mod(X.data, 1) == 0))
        else:
            return (np.issubdtype(X.dtype, np.integer) or 
                    np.all(np.mod(X, 1) == 0))
    
    # Check possible locations
    locations = {
        'layers["counts"]': (
            'counts' in adata.layers,
            lambda: adata.layers['counts']
        ),
        'raw.X': (
            adata.raw is not None and is_counts(adata.raw.X),
            lambda: adata.raw.X
        ),
        'X': (
            is_counts(adata.X),
            lambda: adata.X
        )
    }
    
    for loc_name, (exists, getter) in locations.items():
        if exists:
            logger.info(f"Found counts in {loc_name}")
            counts = getter()
            return scipy.sparse.csr_matrix(counts).toarray()
            
    raise ValueError("No raw counts found in data")

def normalize_data(adata: AnnData) -> AnnData:
    """
    Normalize data using median counts and log transform.
    
    Args:
        adata: Input data
        
    Returns:
        Normalized data with 'logTPM' layer
    """
    adata = adata.copy()
    
    # Get counts
    adata.X = get_counts_matrix(adata)

    # Store in layers['counts']
    logger.info("Storing raw counts in 'layers[\"counts\"]'")
    adata.layers['counts'] = adata.X.copy()

    logger.debug(f"Data range after getting counts: [{adata.X.min()}, {adata.X.max()}]")
    
    # Normalize to median counts
    total_counts = np.sum(adata.X, axis=1)
    median_counts = np.median(total_counts)
    logger.debug(f"Median counts: {median_counts}")
    
    sc.pp.normalize_total(adata, target_sum=median_counts)
    logger.debug(f"Data range after normalization: [{adata.X.min()}, {adata.X.max()}]")
    
    # Log transform
    adata.layers['logTPM'] = np.log1p(adata.X)
    logger.debug(f"Data range after log1p: [{adata.layers['logTPM'].min()}, {adata.layers['logTPM'].max()}]")
    
    return adata

def preprocess_data(adata: AnnData, params: Dict) -> AnnData:
    """
    Complete preprocessing pipeline.
    
    Workflow:
    1. Validate batches and filter cell types
    2. Select abundant genes
    3. Create and normalize one dataset:
       - Abundant genes dataset for batch effects analysis
    
    Args:
        adata: Input data
        params: Analysis parameters
        
    Returns:
        normalized data with abundant genes for batch scores,
    """
    # 1. Validate and filter cell types
    adata_ct_selection = validate_and_filter_celltypes(adata, params)
    
    # 2. Select abundant genes
    adata_ct_genes_selection, abundant_genes = select_abundant_genes(adata_ct_selection, params)
    
    # 3. Normalize both datasets
    adata_batch = normalize_data(adata_ct_genes_selection)  # abundant genes for batch scores
    
    return adata_batch