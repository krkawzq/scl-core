# Batch 2 Kernels

> Batch 2: Neighbors, Centrality, Clonotype, and Co-expression Analysis

## Overview

This batch contains kernels for:
- **Batch-balanced KNN** - Cross-batch neighbor search
- **Graph Centrality** - Network importance measures
- **Clonotype Analysis** - Immune repertoire analysis
- **Co-expression** - Gene module detection (WGCNA-style)

## Files

| File | Description | Main APIs |
|------|-------------|-----------|
| [bbknn.hpp](./bbknn) | Batch Balanced KNN | `bbknn`, `compute_norms`, `build_batch_groups` |
| [centrality.hpp](./centrality) | Graph Centrality Measures | `pagerank`, `betweenness_centrality`, `degree_centrality` |
| [clonotype.hpp](./clonotype) | Clonotype Analysis | `clonal_diversity`, `clone_expansion`, `clone_phenotype_association` |
| [coexpression.hpp](./coexpression) | Co-expression Modules | `correlation_matrix`, `detect_modules`, `module_eigengene` |

## Quick Start

### Batch-Balanced KNN

```cpp
#include "scl/kernel/bbknn.hpp"

Sparse<Real, true> matrix = /* ... */;
Array<int32_t> batch_labels = /* ... */;
Array<Index> indices(/* ... */);
Array<Real> distances(/* ... */);

scl::kernel::bbknn::bbknn(
    matrix, batch_labels, n_batches, k, indices, distances
);
```

### Graph Centrality

```cpp
#include "scl/kernel/centrality.hpp"

Sparse<Real, true> adjacency = /* ... */;
Array<Real> scores(n_nodes);

scl::kernel::centrality::pagerank(adjacency, scores);
```

### Clonotype Diversity

```cpp
#include "scl/kernel/clonotype.hpp"

Array<Index> clone_ids = /* ... */;
Real shannon, simpson, gini;

scl::kernel::clonotype::clonal_diversity(
    clone_ids, n_cells, shannon, simpson, gini
);
```

### Co-expression Modules

```cpp
#include "scl/kernel/coexpression.hpp"

Sparse<Real, true> expression = /* ... */;
Real* corr_matrix = /* allocate */;
Index* module_labels = /* allocate */;

scl::kernel::coexpression::correlation_matrix(
    expression, n_cells, n_genes, corr_matrix
);
Index n_modules = scl::kernel::coexpression::detect_modules(
    dissim, n_genes, module_labels
);
```

## See Also

- [Neighbors](/cpp/kernels/neighbors) - Standard KNN search
- [Statistics](/cpp/kernels/statistics) - Statistical analysis

