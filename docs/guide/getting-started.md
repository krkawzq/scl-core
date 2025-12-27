# Getting Started

Welcome to SCL-Core! This guide will help you get up and running quickly.

## Installation

SCL-Core can be installed via pip:

```bash
pip install scl-core
```

Or install from source for the latest development version:

```bash
git clone https://github.com/krkawzq/scl-core.git
cd scl-core
pip install -e .
```

## Requirements

- Python 3.8+
- NumPy 1.20+
- SciPy 1.7+
- C++17 compatible compiler (for building from source)

## First Steps

### Basic Usage

```python
import scl_core as scl
import numpy as np
from scipy.sparse import csr_matrix

# Create a sparse matrix
data = np.random.randn(1000, 2000)
sparse_data = csr_matrix(data)

# Normalize to total counts
scl.normalize_total(sparse_data, target_sum=1e4)

# Log transform
scl.log1p(sparse_data)

print("Data shape:", sparse_data.shape)
print("Non-zero elements:", sparse_data.nnz)
```

### Working with AnnData

SCL-Core integrates seamlessly with the AnnData ecosystem:

```python
import scanpy as sc
import scl_core as scl

# Load data
adata = sc.datasets.pbmc3k()

# Use SCL-Core for preprocessing
scl.normalize_total(adata.X, target_sum=1e4)
scl.log1p(adata.X)

# Compute neighbors with BBKNN
scl.bbknn(
    adata,
    batch_key='batch',
    neighbors_within_batch=3,
    n_pcs=50
)
```

## Core Concepts

### Sparse Matrix Operations

SCL-Core is optimized for sparse matrices in CSR (Compressed Sparse Row) and CSC (Compressed Sparse Column) formats:

::: tip Format Selection
- Use **CSR** for row-based operations (e.g., normalizing cells, computing cell-wise statistics)
- Use **CSC** for column-based operations (e.g., normalizing genes, computing gene-wise statistics)
:::

### Zero-Copy Integration

SCL-Core uses zero-copy data sharing with NumPy and SciPy, meaning operations are performed **in-place** without copying data:

```python
import numpy as np
from scipy.sparse import csr_matrix
import scl_core as scl

# Original data
X = csr_matrix(np.random.randn(100, 50))
print("Before:", X.data[0])

# In-place operation (no copy)
scl.normalize_total(X, target_sum=100)
print("After:", X.data[0])  # Data is modified in-place
```

### Parallelization

All SCL-Core operations are automatically parallelized across available CPU cores:

```python
import scl_core as scl

# Automatically uses all available cores
scl.neighbors(data, n_neighbors=15)

# Control number of threads
scl.set_num_threads(4)
scl.neighbors(data, n_neighbors=15)
```

## Next Steps

- Learn about architecture and design principles (coming soon)
- Explore the [API Reference](/api/)
- Check out performance tips (coming soon)
- Browse examples (coming soon)

## Getting Help

If you encounter any issues:

1. Check the [API Reference](/api/)
2. Search [GitHub Issues](https://github.com/krkawzq/scl-core/issues)
3. Ask in [GitHub Discussions](https://github.com/krkawzq/scl-core/discussions)
4. Report bugs in the [Issue Tracker](https://github.com/krkawzq/scl-core/issues/new)

