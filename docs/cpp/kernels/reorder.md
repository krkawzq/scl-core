# Matrix Reordering

Sparse matrix reordering and permutation operations.

## Overview

Reordering kernels provide:

- **Row Reordering** - Permute rows of sparse matrix
- **Column Reordering** - Permute columns of sparse matrix
- **Parallel Processing** - Efficient reordering for large matrices
- **Memory Efficient** - Optimized for sparse structures

## Row Reordering

### reorder_rows

Reorder rows of sparse matrix according to permutation:

```cpp
#include "scl/kernel/reorder.hpp"

Sparse<Real, true> matrix = /* ... */;      // Input CSR matrix
Array<const Index> permutation = /* ... */;  // Row permutation [n_rows]
Index n_rows = matrix.rows();

Sparse<Real, true> output;
output = Sparse<Real, true>::create(n_rows, matrix.cols(), /* estimated nnz */);

scl::kernel::reorder::reorder_rows(matrix, permutation, n_rows, output);
```

**Parameters:**
- `matrix`: Input sparse matrix (CSR format)
- `permutation`: Row permutation array, size = `n_rows`
- `n_rows`: Number of rows
- `output`: Output reordered matrix (must be pre-allocated)

**Postconditions:**
- `output[i]` contains row `permutation[i]` from input
- Matrix structure (columns, values) preserved for each row
- Input matrix unchanged

**Algorithm:**
- For each row in parallel:
  1. Read row from input matrix at position `permutation[i]`
  2. Copy values and indices to output at position `i`
  3. Update indptr array

**Complexity:**
- Time: O(nnz) - linear in number of non-zeros
- Space: O(nnz) auxiliary for output matrix

**Thread Safety:**
- Safe - parallelized over rows
- Each thread processes independent rows
- No shared mutable state

**Use cases:**
- Clustering result visualization
- Sorting by metadata (cell type, batch, etc.)
- Data organization for downstream analysis
- Matrix transformation for algorithms

## Column Reordering

### reorder_columns

Reorder columns of sparse matrix according to permutation:

```cpp
Sparse<Real, true> matrix = /* ... */;      // Input matrix
Array<const Index> permutation = /* ... */;  // Column permutation [n_cols]
Index n_cols = matrix.cols();

Sparse<Real, true> output;
output = Sparse<Real, true>::create(matrix.rows(), n_cols, /* estimated nnz */);

scl::kernel::reorder::reorder_columns(matrix, permutation, n_cols, output);
```

**Parameters:**
- `matrix`: Input sparse matrix (CSR or CSC format)
- `permutation`: Column permutation array, size = `n_cols`
- `n_cols`: Number of columns
- `output`: Output reordered matrix (must be pre-allocated)

**Postconditions:**
- Output has columns in permuted order
- Column `j` in output corresponds to column `permutation[j]` in input
- Row structure preserved
- Input matrix unchanged

**Algorithm:**
- For CSR: Requires value remapping (columns change indices)
- For CSC: Direct row permutation of column slices
- Parallel processing over primary dimension

**Complexity:**
- Time: O(nnz) - linear in number of non-zeros
- Space: O(nnz) auxiliary for output matrix

**Thread Safety:**
- Safe - parallelized processing
- No shared mutable state

**Use cases:**
- Gene ordering (by variance, expression level)
- Feature selection result organization
- Matrix transformation for column-based algorithms
- Data visualization (sorted heatmaps)

## Examples

### Sort Cells by Cluster Assignment

```cpp
#include "scl/kernel/reorder.hpp"
#include "scl/core/sparse.hpp"

Sparse<Real, true> expression = /* ... */;
Array<Index> cluster_labels = /* ... */;  // Cluster assignment per cell

// Create permutation: sort by cluster labels
Index n_cells = expression.rows();
std::vector<Index> cell_indices(n_cells);
std::iota(cell_indices.begin(), cell_indices.end(), 0);

// Sort indices by cluster labels
std::sort(cell_indices.begin(), cell_indices.end(),
    [&](Index i, Index j) { return cluster_labels[i] < cluster_labels[j]; });

// Create permutation array
Array<Index> permutation(cell_indices.data(), n_cells);

// Reorder rows (cells)
Sparse<Real, true> sorted_expression;
sorted_expression = Sparse<Real, true>::create(n_cells, expression.cols(),
                                               expression.nnz());
scl::kernel::reorder::reorder_rows(expression, permutation, n_cells,
                                   sorted_expression);
```

### Sort Genes by Variance

```cpp
// Compute gene variances
Array<Real> gene_vars(expression.cols());
// ... compute variances ...

// Create permutation: sort by variance (descending)
std::vector<Index> gene_indices(expression.cols());
std::iota(gene_indices.begin(), gene_indices.end(), 0);
std::sort(gene_indices.begin(), gene_indices.end(),
    [&](Index i, Index j) { return gene_vars[i] > gene_vars[j]; });

Array<Index> permutation(gene_indices.data(), expression.cols());

// Reorder columns (genes)
Sparse<Real, true> sorted_by_variance;
sorted_by_variance = Sparse<Real, true>::create(expression.rows(),
                                                expression.cols(),
                                                expression.nnz());
scl::kernel::reorder::reorder_columns(expression, permutation,
                                     expression.cols(),
                                     sorted_by_variance);
```

### Batch Ordering

```cpp
Array<Index> batch_labels = /* ... */;  // Batch ID per cell

// Sort cells by batch
std::vector<Index> cell_indices(n_cells);
std::iota(cell_indices.begin(), cell_indices.end(), 0);
std::sort(cell_indices.begin(), cell_indices.end(),
    [&](Index i, Index j) { return batch_labels[i] < batch_labels[j]; });

Array<Index> permutation(cell_indices.data(), n_cells);

Sparse<Real, true> batch_ordered;
batch_ordered = Sparse<Real, true>::create(n_cells, expression.cols(),
                                           expression.nnz());
scl::kernel::reorder::reorder_rows(expression, permutation, n_cells,
                                   batch_ordered);
```

## Performance Considerations

### Parallelization

- Operations are parallelized over primary dimension
- Threshold: `PARALLEL_THRESHOLD = 256` rows/columns
- Small matrices may use sequential processing for better cache behavior

### Memory Efficiency

- Output matrix must be pre-allocated
- Estimate nnz for allocation (usually same as input)
- Sparse structure preserved (no densification)

### Permutation Validation

- Caller must ensure permutation is valid:
  - All values in range [0, n-1]
  - No duplicates
  - For row permutation: size = n_rows
  - For column permutation: size = n_cols

---

::: tip Pre-allocation
Always pre-allocate the output matrix with appropriate dimensions. For most cases, nnz will be the same as input, but consider edge cases where permutation might affect structure.
:::

