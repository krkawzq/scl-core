# Python API Reference

The SCL-Core Python API provides a high-level, Pythonic interface for biological data analysis with single-cell data.

## Design Principles

### Pythonic Interface
- Follows PEP 8 style guidelines
- Type hints for all public functions
- Comprehensive docstrings with examples
- Integration with standard scientific Python stack

### NumPy/SciPy Integration
- Native support for NumPy arrays
- SciPy sparse matrices (CSR/CSC)
- Zero-copy where possible
- Automatic format conversion when needed

### Scanpy Compatibility
- Works seamlessly with AnnData objects
- Follows scanpy API conventions
- Can be used as drop-in replacement for scanpy functions
- Maintains metadata and observations

### High Performance
- Automatic parallelization
- Optimized memory usage
- Leverages C++ kernels under the hood
- Minimal Python overhead

---

## API Organization

### Preprocessing (`scl.pp`)
Data preprocessing and normalization:
- Normalization (L1, L2, log)
- Feature selection
- Dimensionality reduction preparation

### Neighbors (`scl.neighbors`)
Neighborhood graph construction:
- K-nearest neighbors
- BBKNN (batch-balanced KNN)
- Connectivity graphs
- Distance calculations

### Statistics (`scl.stats`)
Statistical analysis:
- Differential expression (Mann-Whitney U)
- Wilcoxon rank-sum test
- T-tests and variance tests
- Multiple testing correction

### Utilities (`scl.utils`)
Helper functions:
- Sparse matrix operations
- Data conversion
- Validation and diagnostics

---

## Quick Start

### Installation

```bash
pip install scl-core
```

### Basic Usage

```python
import scl
import numpy as np
from scipy.sparse import csr_matrix

# Create sparse matrix
data = csr_matrix(np.random.randn(1000, 2000))

# Normalize (L2 norm per cell)
scl.pp.normalize_total(data, norm='l2')

# Find neighbors
scl.neighbors.compute_neighbors(data, n_neighbors=15, metric='euclidean')

# Run statistical test
results = scl.stats.mannwhitneyu(data, group_labels)
```

### AnnData Integration

```python
import scanpy as sc
import scl

# Load data
adata = sc.read_h5ad('data.h5ad')

# Use scl for preprocessing (faster than scanpy)
scl.pp.normalize_total(adata, target_sum=1e4)
scl.pp.log1p(adata)

# Compute neighbors with BBKNN
scl.neighbors.bbknn(adata, batch_key='batch')

# Statistical testing
scl.stats.rank_genes_groups(adata, groupby='cell_type')
```

---

## Installation

### Requirements
- Python >= 3.8
- NumPy >= 1.20
- SciPy >= 1.7
- (Optional) scanpy >= 1.9 for AnnData support

### From PyPI
```bash
pip install scl-core
```

### From Source
```bash
git clone https://github.com/krkawzq/scl-core.git
cd scl-core
pip install -e .
```

### Development Installation
```bash
pip install -e ".[dev]"  # Includes testing and documentation tools
```

---

## Configuration

### Parallelization

```python
# Set number of threads
scl.set_num_threads(8)

# Auto-detect (default)
scl.set_num_threads(0)

# Get current setting
n_threads = scl.get_num_threads()
```

### Memory Management

```python
# Enable memory profiling
scl.config.profile_memory = True

# Set memory limit for operations
scl.config.max_memory = '8GB'
```

### Verbosity

```python
# Set logging level
scl.config.verbosity = 2  # 0=error, 1=warning, 2=info, 3=debug

# Disable all output
scl.config.verbosity = 0
```

---

## API Conventions

### Function Naming
- `compute_*` - Returns new data structure
- `*_inplace` - Modifies data in-place
- Without suffix - Usually modifies AnnData in-place

### Parameters
- `adata` - AnnData object (modified in-place unless `copy=True`)
- `copy` - If True, return modified copy instead of modifying in-place
- `n_jobs` - Number of parallel jobs (-1 = all cores)

### Return Values
- Functions operating on AnnData usually return None (modify in-place)
- Use `copy=True` to get a modified copy
- Computational functions return results directly

---

## Performance Tips

1. **Use Sparse Matrices**: Dense arrays are slower for sparse data
2. **Preallocate**: Specify output shapes when possible
3. **Batch Operations**: Process multiple samples together
4. **Thread Count**: Tune `n_jobs` for your hardware
5. **Memory Layout**: CSR format is generally fastest

---

## Next Steps

1. Explore [Preprocessing](/api/python/preprocessing/) for data preparation
2. Learn [Neighbors](/api/python/neighbors/) for graph construction
3. Check [Statistics](/api/python/stats/) for differential analysis
4. See examples for complete workflows (coming soon)

