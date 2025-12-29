---
title: Batch-Balanced KNN
description: Batch-aware K-nearest neighbors for multi-batch data
---

# Batch-Balanced KNN

The `bbknn` kernel provides batch-balanced KNN search that ensures balanced representation across batches.

## Overview

BBKNN is essential for:
- Multi-batch data integration
- Removing batch effects in neighbor graphs
- Improving downstream analysis quality

## Functions

### `bbknn`

Compute batch-balanced KNN graph.

```cpp
template <typename T, bool IsCSR>
CSR bbknn(
    const Sparse<T, IsCSR>& matrix,
    Array<const Index> batch_labels,
    Index k = 15
);
```

**Parameters**:
- `matrix` [in]: Expression matrix
- `batch_labels` [in]: Batch ID for each cell
- `k` [in]: Number of neighbors per batch

**Returns**: Batch-balanced KNN graph

**Example**:
```cpp
#include "scl/kernel/bbknn.hpp"

Array<Index> batch_labels = {batches_ptr, n_cells};
CSR knn = kernel::bbknn::bbknn(matrix, batch_labels, k=15);
```

## Configuration

```cpp
namespace scl::kernel::bbknn::config {
    constexpr Size CHUNK_SIZE = 64;
    constexpr Size PREFETCH_DISTANCE = 8;
    constexpr Size MIN_SAMPLES_PARALLEL = 128;
}
```

## Related Documentation

- [Neighbors](./neighbors.md) - General neighbor search
- [Alignment](./alignment.md) - Data alignment
