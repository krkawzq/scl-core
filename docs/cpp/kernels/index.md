# Kernels

The `scl/kernel/` directory contains 400+ computational operators organized by functionality.

## Overview

Kernels provide:

- **Sparse Tools** - Matrix conversion, validation, cleanup, optimization
- **Linear Algebra** - Sparse matrix-vector multiplication, Gram matrices
- **Preprocessing** - Normalization, scaling, log transforms, standardization
- **Feature Selection** - Highly variable genes, quality control metrics
- **Statistics** - Statistical tests, correlation, multiple testing correction
- **Neighbors** - KNN, batch-balanced KNN
- **Clustering** - Leiden, Louvain community detection
- **Spatial Analysis** - Spatial patterns, hotspot detection, tissue analysis
- **Enrichment** - Gene set enrichment analysis
- **Graph Operations** - Diffusion, propagation, centrality
- **Single-Cell Analysis** - Markers, doublet detection, velocity, pseudotime
- **Data Manipulation** - Merging, slicing, reordering, sampling, permutation

## Categories

### Sparse Tools

Matrix infrastructure and utilities:

```cpp
#include "scl/kernel/sparse.hpp"
#include "scl/kernel/sparse_opt.hpp"

// Convert to contiguous CSR/CSC
auto arrays = scl::kernel::sparse::to_contiguous_arrays(matrix);

// Convert to COO format
auto coo = scl::kernel::sparse::to_coo_arrays(matrix);

// Validate matrix structure
auto result = scl::kernel::sparse::validate(matrix);

// Get memory info
auto info = scl::kernel::sparse::memory_info(matrix);

// Eliminate zeros
scl::kernel::sparse::eliminate_zeros(matrix, tolerance);

// Prune small values
scl::kernel::sparse::prune(matrix, threshold, keep_structure);

// Sparse optimization (Lasso, elastic net)
scl::kernel::sparse_opt::lasso_coordinate_descent(
    X, y, alpha, coefficients, max_iter, tol
);
scl::kernel::sparse_opt::elastic_net_coordinate_descent(
    X, y, alpha, l1_ratio, coefficients, max_iter, tol
);
scl::kernel::sparse_opt::fista(X, y, alpha, coefficients, max_iter, tol);
```

### Normalization

Row/column normalization and scaling:

```cpp
#include "scl/kernel/normalize.hpp"

// Compute row norms
scl::kernel::normalize::row_norms(matrix, NormMode::L2, output);

// Normalize rows in-place
scl::kernel::normalize::normalize_rows_inplace(matrix, NormMode::L2);

// Scale matrix
scl::kernel::normalize::scale(matrix, factor);
```

### Softmax

Softmax normalization with temperature scaling:

```cpp
#include "scl/kernel/softmax.hpp"

// Standard softmax
scl::kernel::softmax::softmax_inplace(values, len);

// With temperature
scl::kernel::softmax::softmax_inplace(values, len, 0.5);

// Log-softmax
scl::kernel::softmax::log_softmax_inplace(values, len);

// Sparse matrix
scl::kernel::softmax::softmax_inplace(matrix);
```

### Scaling and Transforms

Standardization, scaling, and logarithmic transforms:

```cpp
#include "scl/kernel/scale.hpp"
#include "scl/kernel/log1p.hpp"

// Standardize matrix
scl::kernel::scale::standardize(matrix, means, stds, max_value, zero_center);

// Scale rows
scl::kernel::scale::scale_rows(matrix, scales);

// Log1p transform
scl::kernel::log1p::log1p_inplace(matrix);

// Log2(1+x) transform
scl::kernel::log1p::log2p1_inplace(matrix);

// Reverse transform
scl::kernel::log1p::expm1_inplace(matrix);
```

### Linear Algebra

Sparse matrix-vector multiplication and Gram matrices:

```cpp
#include "scl/kernel/algebra.hpp"
#include "scl/kernel/gram.hpp"

// Sparse matrix-vector: y = alpha * A * x + beta * y
scl::kernel::algebra::spmv(A, x, y, alpha, beta);

// Simple: y = A * x
scl::kernel::algebra::spmv_simple(A, x, y);

// Gram matrix: G[i,j] = dot(row_i, row_j)
scl::kernel::gram::gram(matrix, output);
```

### Statistics

Statistical tests, correlation, and multiple testing correction:

```cpp
#include "scl/kernel/ttest.hpp"
#include "scl/kernel/mwu.hpp"
#include "scl/kernel/correlation.hpp"
#include "scl/kernel/multiple_testing.hpp"

// T-test
scl::kernel::ttest::ttest(matrix, group_ids, t_stats, p_values, log2_fc);

// Mann-Whitney U test
scl::kernel::mwu::mwu_test(matrix, group_ids, u_stats, p_values, log2_fc);

// Pearson correlation
scl::kernel::correlation::pearson(matrix, output);

// Multiple testing correction
scl::kernel::multiple_testing::benjamini_hochberg(p_values, adjusted_p_values);
scl::kernel::multiple_testing::bonferroni(p_values, adjusted_p_values);
```

### Group Aggregation

Per-group statistics:

```cpp
#include "scl/kernel/group.hpp"

// Compute group means and variances
scl::kernel::group::group_stats(
    matrix, group_ids, n_groups, group_sizes, means, vars
);
```

### Matrix Operations

Merging and slicing:

```cpp
#include "scl/kernel/merge.hpp"
#include "scl/kernel/slice.hpp"

// Vertical stack
auto result = scl::kernel::merge::vstack(matrix1, matrix2);

// Horizontal stack
auto result = scl::kernel::merge::hstack(matrix1, matrix2);

// Slice rows
auto sliced = scl::kernel::slice::slice_primary(matrix, keep_indices);

// Filter columns
auto filtered = scl::kernel::slice::filter_secondary(matrix, mask);
```

### Neighbors

K-nearest neighbors:

```cpp
#include "scl/kernel/neighbors.hpp"

// Pre-compute norms
scl::kernel::neighbors::compute_norms(matrix, norms_sq);

// Compute KNN
scl::kernel::neighbors::knn(matrix, norms_sq, k, indices, distances);
```

### Batch Balanced KNN

Batch-aware KNN for integrating data across batches:

```cpp
#include "scl/kernel/bbknn.hpp"

// Pre-compute norms (optional)
scl::kernel::bbknn::compute_norms(matrix, norms_sq);

// Compute BBKNN (k neighbors from each batch)
scl::kernel::bbknn::bbknn(
    matrix, batch_labels, n_batches, k, indices, distances, norms_sq
);
```

### MMD

Maximum Mean Discrepancy for distribution comparison:

```cpp
#include "scl/kernel/mmd.hpp"

// Compare two distributions
scl::kernel::mmd::mmd_rbf(mat_x, mat_y, output, gamma);
```

### Clustering

Community detection:

```cpp
#include "scl/kernel/leiden.hpp"
#include "scl/kernel/louvain.hpp"

// Leiden clustering
auto labels = scl::kernel::leiden::leiden(graph, resolution);

// Louvain clustering
auto labels = scl::kernel::louvain::louvain(graph, resolution);
```

### Spatial Analysis

Spatial autocorrelation and niche analysis:

```cpp
#include "scl/kernel/spatial.hpp"
#include "scl/kernel/niche.hpp"

// Spatial autocorrelation
scl::kernel::spatial::morans_i(graph, features, output);
scl::kernel::spatial::gearys_c(graph, features, output);

// Niche composition
scl::kernel::niche::neighborhood_composition(
    spatial_neighbors, cell_type_labels, n_types, composition
);

// Co-localization analysis
scl::kernel::niche::colocalization_matrix(
    spatial_neighbors, cell_type_labels, n_types, coloc_matrix
);

// Niche diversity (Shannon entropy)
scl::kernel::niche::niche_diversity(
    spatial_neighbors, cell_type_labels, n_types, diversity
);

// Boundary detection
scl::kernel::niche::niche_boundary_score(
    spatial_neighbors, cell_type_labels, n_types, boundary_scores
);
```

### Feature Selection

Highly variable genes and quality control:

```cpp
#include "scl/kernel/hvg.hpp"
#include "scl/kernel/qc.hpp"

// Select highly variable genes by dispersion
scl::kernel::hvg::select_by_dispersion(
    matrix, n_top, out_indices, out_mask, out_dispersions
);

// Quality control metrics
scl::kernel::qc::compute_basic_qc(matrix, out_n_genes, out_total_counts);
scl::kernel::qc::compute_subset_pct(matrix, subset_mask, out_pcts);
```

### Marker Genes

Marker gene identification and specificity scoring:

```cpp
#include "scl/kernel/markers.hpp"

// Find marker genes for each cluster
scl::kernel::markers::find_markers(
    expression, cluster_labels, n_cells, n_genes, n_clusters,
    marker_genes, marker_scores, max_markers, min_fc, max_pval
);
```

### Data Manipulation

Reordering, sampling, and permutation:

```cpp
#include "scl/kernel/reorder.hpp"
#include "scl/kernel/sampling.hpp"
#include "scl/kernel/permutation.hpp"

// Reorder rows
scl::kernel::reorder::reorder_rows(matrix, permutation, n_rows, output);

// Geometric sketching for downsampling
scl::kernel::sampling::geometric_sketching(
    data, target_size, selected_indices, n_selected, seed
);

// Permutation test
scl::kernel::permutation::permutation_test(
    data, labels, test_func, n_permutations, p_value, seed
);
```

## Design Patterns

### Functional API

Pure functions with no side effects:

```cpp
// Pure function - returns result
auto result = compute_something(input);

// In-place modification - clearly named
modify_something_inplace(data);
```

### Template-Based Polymorphism

Works with any compatible type:

```cpp
template <CSRLike MatrixT>
void process_matrix(const MatrixT& matrix) {
    // Works with any CSR-like type
}
```

### Explicit Parallelization

```cpp
// Parallel by default for large inputs
parallel_for(Size(0), n, [&](size_t i) {
    process(data[i]);
});
```

## Performance

### SIMD Optimization

Hot paths use SIMD:

```cpp
namespace s = scl::simd;
const s::Tag d;

for (size_t i = 0; i < n; i += s::Lanes(d)) {
    auto v = s::Load(d, data + i);
    // SIMD operations
}
```

### Minimal Allocations

Use workspaces instead of allocating:

```cpp
// Pre-allocate workspace
WorkspacePool<Real> pool(num_threads, workspace_size);

// Reuse in parallel loop
parallel_for(Size(0), n, [&](size_t i, size_t thread_rank) {
    Real* workspace = pool.get(thread_rank);
    // Use workspace
});
```

## Next Steps

Explore specific kernel categories:

### Core Infrastructure
- [Sparse Tools](/cpp/kernels/sparse-tools) - Matrix utilities and optimization
- [Algebra](/cpp/kernels/algebra) - Sparse linear algebra operations
- [Gram](/cpp/kernels/gram) - Gram matrix computation

### Preprocessing
- [Normalization](/cpp/kernels/normalization) - Normalization and scaling
- [Scale](/cpp/kernels/scale) - Standardization and row scaling
- [Log1p](/cpp/kernels/log1p) - Logarithmic transforms
- [Softmax](/cpp/kernels/softmax) - Softmax normalization with temperature scaling

### Feature Selection
- [HVG](/cpp/kernels/hvg) - Highly variable gene selection
- [QC](/cpp/kernels/qc) - Quality control metrics
- [Markers](/cpp/kernels/markers) - Marker gene identification

### Statistics
- [T-test](/cpp/kernels/ttest) - Parametric statistical test
- [Mann-Whitney U](/cpp/kernels/mwu) - Non-parametric statistical test
- [Correlation](/cpp/kernels/correlation) - Pearson correlation
- [Multiple Testing](/cpp/kernels/multiple_testing) - FDR and FWER correction

### Spatial Analysis
- [Spatial](/cpp/kernels/spatial) - Spatial autocorrelation (Moran's I, Geary's C)
- [Niche Analysis](/cpp/kernels/niche) - Neighborhood composition, co-localization, diversity

### Neighbors and Clustering
- [Neighbors](/cpp/kernels/neighbors) - KNN algorithms
- [BBKNN](/cpp/kernels/bbknn) - Batch Balanced KNN for batch integration
- [Leiden](/cpp/kernels/leiden) - Leiden clustering
- [Louvain](/cpp/kernels/louvain) - Louvain clustering

### Data Manipulation
- [Merge](/cpp/kernels/merge) - Matrix merging operations
- [Slice](/cpp/kernels/slice) - Matrix slicing operations
- [Reorder](/cpp/kernels/reorder) - Matrix reordering
- [Sampling](/cpp/kernels/sampling) - Downsampling methods
- [Permutation](/cpp/kernels/permutation) - Permutation testing

### Advanced Analysis
- [MMD](/cpp/kernels/mmd) - Maximum Mean Discrepancy
- [Group](/cpp/kernels/group) - Group aggregation statistics
- [Sparse Optimization](/cpp/kernels/sparse-opt) - Lasso, elastic net, proximal methods
- [Niche Analysis](/cpp/kernels/niche) - Cellular neighborhood and microenvironment analysis

---

::: tip High Performance
All kernels are optimized for performance with SIMD, parallelization, and minimal allocations.
:::

