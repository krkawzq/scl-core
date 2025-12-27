# Preprocessing

Data preprocessing and normalization functions.

::: tip Status
This section is under construction.
:::

## Overview

The `scl.pp` module provides functions for preprocessing single-cell data, including normalization, filtering, and feature selection.

## Functions

### Normalization
- `normalize_total()` - Normalize counts per cell
- `log1p()` - Logarithm with pseudocount
- `scale()` - Z-score normalization

### Filtering
- `filter_cells()` - Remove cells by criteria
- `filter_genes()` - Remove genes by criteria

### Feature Selection
- `highly_variable_genes()` - Select highly variable genes

## Coming Soon

Detailed API reference will be auto-generated from Python docstrings.

## Example

```python
import scl

# Normalize to 10,000 counts per cell
scl.pp.normalize_total(adata, target_sum=1e4)

# Log transform
scl.pp.log1p(adata)

# Select highly variable genes
scl.pp.highly_variable_genes(adata, n_top_genes=2000)
```

