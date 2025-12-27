# Neighbors

Neighborhood graph construction and analysis.

::: tip Status
This section is under construction.
:::

## Overview

The `scl.neighbors` module provides functions for computing k-nearest neighbors graphs, essential for clustering and visualization.

## Functions

### Graph Construction
- `compute_neighbors()` - Standard KNN
- `bbknn()` - Batch-balanced KNN

### Graph Operations
- `neighbors_to_connectivities()` - Convert to connectivity graph
- `neighbors_to_distances()` - Extract distance matrix

## Coming Soon

Detailed API reference will be auto-generated from Python docstrings.

## Example

```python
import scl

# Compute 15 nearest neighbors
scl.neighbors.compute_neighbors(adata, n_neighbors=15)

# Or use BBKNN for batch correction
scl.neighbors.bbknn(adata, batch_key='batch', n_neighbors=15)
```

